"""Pre-computed borehole embedding map belief transformer.

PreCompBHMapBeliefTransformer is a reconstruction-pretraining model that
accepts a partially-observed borehole map where the borehole observations have
already been encoded into latent vectors by an external encoder (e.g.
JEPAModel.embed()).  It produces:

  * A single map-level latent (CLS token) summarising the geological belief
    state — the primary output for downstream tasks such as total ore
    prediction, next-drill value estimation, or RL policies.

  * Per-cell ore value predictions via a lightweight reconstruction head —
    used for supervised pretraining to ensure the latent contains meaningful
    geological information.

Input format
------------
  (B, 2 + latent_dim, n_x, n_y)  float32
    channel 0 : sparse observed ore value (0 at unobserved cells)
    channel 1 : binary observation mask   (1 = drilled, 0 = not drilled)
    channels 2+: borehole encoder latent  (zero vector at unobserved cells)

This is the same format produced by the end-to-end models after their scatter
step, and matches the GeologicalBeliefDataset output when latent_dim > 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..map_encoder_components.components import (
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)
from ..model_configs import MapBeliefConfig  # noqa: F401 — re-exported for callers


class PreCompBHMapBeliefTransformer(nn.Module):
    """Transformer-based geological belief encoder with reconstruction pretraining.

    Converts sparse borehole observations (with pre-computed borehole latents)
    into:
      1. A map-level latent belief embedding (CLS token), suitable for use as
         the belief state in downstream RL / planning / prediction heads.
      2. A full-map ore prediction via a per-cell reconstruction head, used
         for supervised pretraining with MSE loss.

    The forward() method is a drop-in replacement for UNetBelief: it accepts
    the same (B, 2 + latent_dim, n_x, n_y) input and returns (B, 1, n_x, n_y).

    For downstream tasks use encode() to obtain the map belief latent:
        latent = model.encode(x)   # (B, d_model)

    To obtain both in a single forward pass (no recomputation) use:
        ore_map, latent = model.forward_with_latent(x)
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = SpatialTokenEmbedding(cfg)
        self.encoder     = MapBeliefEncoder(cfg)
        self.ore_head    = OreReconstructionHead(cfg)

    def _forward_all(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full pipeline and return (ore_map, map_belief_latent)."""
        n_x, n_y = x.shape[2], x.shape[3]
        tokens               = self.token_embed(x)
        cls_out, spatial_out = self.encoder(tokens)
        ore_map              = self.ore_head(spatial_out, n_x, n_y)
        return ore_map, cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the ore map from sparse observations.

        Drop-in replacement for UNetBelief.forward().

        Parameters
        ----------
        x : (B, 2 + latent_dim, n_x, n_y)

        Returns
        -------
        ore_map : (B, 1, n_x, n_y)
        """
        ore_map, _ = self._forward_all(x)
        return ore_map

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the partially observed map into a belief latent vector.

        Parameters
        ----------
        x : (B, 2 + latent_dim, n_x, n_y)

        Returns
        -------
        latent : (B, d_model)
        """
        _, latent = self._forward_all(x)
        return latent

    def forward_with_latent(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both the ore reconstruction and the map belief latent.

        Parameters
        ----------
        x : (B, 2 + latent_dim, n_x, n_y)

        Returns
        -------
        ore_map : (B, 1, n_x, n_y)
        latent  : (B, d_model)
        """
        return self._forward_all(x)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

MapBeliefTransformer = PreCompBHMapBeliefTransformer
MapBeliefModel       = PreCompBHMapBeliefTransformer
