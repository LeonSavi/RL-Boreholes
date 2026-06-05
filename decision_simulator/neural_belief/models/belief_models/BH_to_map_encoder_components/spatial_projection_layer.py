"""SpatialProjectionLayer: bridges borehole encoder output to map encoder input.

Encapsulates the two-step projection from sparse borehole observations to a
transformer-ready spatial token sequence:
  1. scatter_to_map     — scatters per-borehole embeddings, ore values, and the
                          drilled mask onto a dense (B, 2+latent_dim, n_x, n_y) grid.
  2. SpatialTokenEmbedding — projects each grid cell into a d_model token with
                          2-D sinusoidal positional encoding and layer normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..end_to_end.utils.end_to_end_helpers import scatter_to_map
from ..map_encoder_components.components import SpatialTokenEmbedding
from ..model_configs import MapBeliefConfig


class SpatialProjectionLayer(nn.Module):
    """Projects sparse borehole observations onto a spatial token grid.

    Bridges the borehole encoder and the map encoder:

    Borehole Embeddings
        ↓
    scatter_to_map  — sparse (ore, mask, latent) → dense (B, 2+latent_dim, n_x, n_y)
        ↓
    SpatialTokenEmbedding  — linear proj + 2-D sinusoidal PE + LayerNorm
        ↓
    Spatial Tokens  (B, n_x * n_y, d_model)

    Input
    -----
    ore_vals     : (B, K) — observed ore values (0 at padded positions)
    positions    : (B, K, 2) — normalised [0,1] (x, y) grid coordinates
    latents      : (B, K, latent_dim) — per-borehole encoder embeddings
    padding_mask : (B, K) bool — True at padded positions; None if no padding

    Output
    ------
    tokens : (B, n_x * n_y, d_model)
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = SpatialTokenEmbedding(cfg)

    def forward(
        self,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = scatter_to_map(
            ore_vals,
            positions,
            latents,
            padding_mask,
            self.cfg.n_x,
            self.cfg.n_y,
            self.cfg.latent_dim,
        )
        return self.token_embed(x)
