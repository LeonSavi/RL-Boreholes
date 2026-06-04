"""End-to-end geological map belief transformer with raw borehole encoding.

EndToEndMapBeliefTransformer encodes raw drilled boreholes end-to-end and
reconstructs the full ore map.  Unlike MapBeliefTransformer, which requires
pre-computed borehole latents, this model trains the borehole encoder jointly
with the map reconstruction head.

Architecture
------------
  BoreholeTransformerEncoder
      1D CNN backbone compresses depth from D=440 to D/16 tokens.  A small
      transformer captures long-range depth dependencies.  Mean-pool + linear
      projection yields a latent_dim embedding per borehole.
      Padded boreholes are skipped to protect BatchNorm statistics.

  Scatter to dense map tensor
      Drilled-cell positions (normalised [0,1]) are converted to grid indices
      and the borehole embeddings, ore values, and drilled mask are scattered
      into a dense (2 + latent_dim, n_x, n_y) tensor that matches the
      MapBeliefTransformer input format exactly.

  SpatialTokenEmbedding → MapBeliefEncoder → OreReconstructionHead
      Reused directly from MapBeliefTransformer — the map belief encoder and
      reconstruction head are identical to the pre-computed-latent model.

Input (forward)
---------------
boreholes    : (B, K, V, D)  padded standardised raw boreholes
ore_vals     : (B, K)        observed ore values  (0 at padding)
positions    : (B, K, 2)     normalised [0,1] (x, y) of drilled cells
padding_mask : (B, K) bool   True at zero-padded rows

Output
------
ore_map : (B, 1, n_x, n_y)  full reconstructed ore map (normalised space)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..borehole_encoder_components.components import BoreholeTransformerEncoder
from ..map_encoder_components.components import (
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)
from ..model_configs import EndToEndMapBeliefConfig, MapBeliefConfig  # noqa: F401
from .end_to_end_helpers import encode_boreholes, scatter_to_map
from .model_configs import E2EConfig  # noqa: F401

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EndToEndMapBeliefTransformer(nn.Module):
    """End-to-end geological map belief transformer.

    Trains BoreholeTransformerEncoder jointly with the MapBeliefTransformer
    components (SpatialTokenEmbedding, MapBeliefEncoder, OreReconstructionHead)
    so that borehole embeddings learn features useful for global geological map
    reconstruction rather than single-candidate scoring.

    Training objective: full-map MSE reconstruction loss.

    For downstream tasks use encode() to obtain the map belief latent (CLS
    token), or forward_with_latent() to get both outputs in one pass.
    """

    def __init__(self, cfg: EndToEndMapBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.bh_encoder = BoreholeTransformerEncoder(cfg.to_e2e_config())

        map_cfg = cfg.to_map_belief_config()
        self.token_embed = SpatialTokenEmbedding(map_cfg)
        self.map_encoder = MapBeliefEncoder(map_cfg)
        self.ore_head = OreReconstructionHead(map_cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_all(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, map_belief_latent)."""
        latents = encode_boreholes(self.bh_encoder, boreholes, padding_mask, self.cfg.latent_dim)
        x = scatter_to_map(ore_vals, positions, latents, padding_mask, self.cfg.n_x, self.cfg.n_y, self.cfg.latent_dim)
        tokens = self.token_embed(x)
        cls_out, spatial_out = self.map_encoder(tokens)
        ore_map = self.ore_head(spatial_out, self.cfg.n_x, self.cfg.n_y)
        return ore_map, cls_out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the full ore map from raw drilled boreholes.

        Parameters
        ----------
        boreholes    : (B, K, V, D) — padded standardised raw boreholes
        ore_vals     : (B, K)       — observed ore values (0 at padding)
        positions    : (B, K, 2)    — normalised [0,1] (x, y)
        padding_mask : (B, K) bool  — True at padded positions

        Returns
        -------
        ore_map : (B, 1, n_x, n_y)
        """
        ore_map, _ = self._forward_all(boreholes, ore_vals, positions, padding_mask)
        return ore_map

    def encode(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the map-level belief latent (CLS token output).

        Returns
        -------
        latent : (B, d_model)
        """
        _, latent = self._forward_all(boreholes, ore_vals, positions, padding_mask)
        return latent

    def forward_with_latent(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both the ore reconstruction and the map belief latent.

        Returns
        -------
        ore_map : (B, 1, n_x, n_y)
        latent  : (B, d_model)
        """
        return self._forward_all(boreholes, ore_vals, positions, padding_mask)
