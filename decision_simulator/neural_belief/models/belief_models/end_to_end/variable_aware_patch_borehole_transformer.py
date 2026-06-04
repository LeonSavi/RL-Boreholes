"""Variable-aware patch borehole encoder with CLS-token pooling and end-to-end map belief transformer.

Experimental variant of PatchBoreholeCLSEndToEndMapBeliefTransformer that gives
each geological variable its own token stream instead of flattening all variables
into a single patch token.

Architecture comparison
-----------------------
CLS-patch (patch_borehole_cls_transformer.py):
  (B, V, D) → (B, n_patches, V * patch_size) → linear → PE → CLS → transformer → CLS[0] → latent
  Variables are anonymous channels inside each patch token.

Variable-aware (this file):
  (B, V, D) → (B, V, n_patches, patch_size) → (B, V*n_patches, patch_size)
            → linear → + var_embed + depth_PE → prepend CLS → transformer → CLS[0] → latent
  Each variable gets its own token per patch; the transformer can model
  cross-variable interactions across depth explicitly.

Rationale
---------
Flattening V variables into a single patch token treats them as anonymous
channels. Adding a learned variable embedding lets the transformer distinguish
"this is the Fe column at depth 3" from "this is the Al column at depth 3",
enabling explicit cross-variable and cross-depth attention.

Everything downstream (scatter, SpatialTokenEmbedding, MapBeliefEncoder,
OreReconstructionHead) is identical to the CLS-patch variant.

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

from ..borehole_encoder_components.components import (
    VariableAwarePatchBoreholeTransformerEncoder,
)
from ..map_encoder_components.components import (
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)
from ..model_configs import VariableAwarePatchBoreholeEndToEndConfig  # noqa: F401
from .end_to_end_helpers import encode_boreholes, scatter_to_map
from .model_configs import VariableAwarePatchBoreholeConfig  # noqa: F401

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VariableAwarePatchBoreholeEndToEndMapBeliefTransformer(nn.Module):
    """End-to-end map belief transformer with a variable-aware patch borehole encoder.

    Borehole encoding: each (variable, depth patch) pair becomes a separate
    transformer token, allowing explicit cross-variable and cross-depth attention
    via a learned variable embedding plus sinusoidal depth positional encoding.

    Everything downstream (scatter, SpatialTokenEmbedding, MapBeliefEncoder,
    OreReconstructionHead) is identical to the CLS-patch variant.

    Training objective: full-map MSE reconstruction loss.

    For downstream tasks use encode() to obtain the map belief latent (CLS
    token), or forward_with_latent() to get both outputs in one pass.
    """

    def __init__(self, cfg: VariableAwarePatchBoreholeEndToEndConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.bh_encoder = VariableAwarePatchBoreholeTransformerEncoder(
            cfg.to_encoder_config()
        )

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
