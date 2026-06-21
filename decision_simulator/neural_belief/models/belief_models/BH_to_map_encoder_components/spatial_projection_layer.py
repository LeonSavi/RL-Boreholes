"""SpatialProjectionLayer: bridges borehole encoder output to map encoder input.

Encapsulates the two-step projection from sparse borehole observations to a
transformer-ready spatial token sequence:
  1. scatter_to_map     — scatters per-borehole embeddings, ore values, and the
                          drilled mask onto a dense (B, 2+latent_dim, n_x, n_y) grid.
  2. SpatialTokenEmbedding — projects each grid cell into a d_model token with
                          2-D sinusoidal positional encoding and layer normalisation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..model_configs import MapBeliefConfig


# ---------------------------------------------------------------------------
# 2-D Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def _make_2d_sinusoidal_pe(
    n_x: int,
    n_y: int,
    d_model: int,
    device: torch.device,
    pe_max_freq: float = 10000.0,
) -> torch.Tensor:
    """Build a continuous 2-D sinusoidal positional encoding.

    Extends the 1-D sinusoidal PE to two spatial dimensions by splitting
    ``d_model`` evenly: the first ``d_model//2`` dimensions encode the x-axis
    (row), the last ``d_model//2`` encode the y-axis (column).

    Using continuous coordinates in [0, 1] rather than integer grid indices
    means the encoding is resolution-independent.

    Returns
    -------
    pe : (n_x * n_y, d_model) float32 tensor
    """
    d_half = d_model // 2

    x_coords = torch.linspace(0.0, 1.0, n_x, device=device)
    y_coords = torch.linspace(0.0, 1.0, n_y, device=device)

    x_flat = x_coords[:, None].expand(n_x, n_y).reshape(n_x * n_y, 1)
    y_flat = y_coords[None, :].expand(n_x, n_y).reshape(n_x * n_y, 1)

    div_term = torch.exp(
        torch.arange(0, d_half, 2, device=device, dtype=torch.float32)
        * -(math.log(pe_max_freq) / d_half)
    )

    N = n_x * n_y
    pe_x = torch.zeros(N, d_half, device=device)
    pe_x[:, 0::2] = torch.sin(x_flat * div_term)
    pe_x[:, 1::2] = torch.cos(x_flat * div_term)

    pe_y = torch.zeros(N, d_half, device=device)
    pe_y[:, 0::2] = torch.sin(y_flat * div_term)
    pe_y[:, 1::2] = torch.cos(y_flat * div_term)

    return torch.cat([pe_x, pe_y], dim=1)  # (N, d_model)


# ---------------------------------------------------------------------------
# SpatialTokenEmbedding
# ---------------------------------------------------------------------------

class SpatialTokenEmbedding(nn.Module):
    """Convert a dense grid input into a sequence of spatial tokens.

    Each of the n_x × n_y grid cells becomes one token.  The raw token
    features are the cell's coordinates, observation mask, ore value, and
    borehole latent embedding.  A 2-D sinusoidal positional encoding is
    added before a linear projection to d_model.

    The grid dimensions are read from the input tensor at runtime (not from
    the config) so the module can handle variable-size grids.

    Input
    -----
    x : (B, 2 + latent_dim, n_x, n_y) — standard GeologicalBeliefDataset format

    Output
    ------
    tokens : (B, n_x * n_y, d_model) — one token per grid cell
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.raw_token_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, n_x, n_y = x.shape
        device = x.device

        x_grid = np.broadcast_to(np.linspace(0.0, 1.0, n_x, dtype=np.float32)[:, None], (n_x, n_y)).copy()
        y_grid = np.broadcast_to(np.linspace(0.0, 1.0, n_y, dtype=np.float32)[None, :], (n_x, n_y)).copy()
        coords_np = np.stack([x_grid, y_grid], axis=0)  # (2, n_x, n_y)
        coords = torch.from_numpy(coords_np).to(device=device, dtype=x.dtype)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, n_x, n_y)

        ore  = x[:, 0:1, :, :]
        mask = x[:, 1:2, :, :]
        lat  = x[:, 2:,  :, :]

        raw = torch.cat([coords, mask, ore, lat], dim=1)  # (B, raw_token_dim, n_x, n_y)
        raw = raw.permute(0, 2, 3, 1)                     # (B, n_x, n_y, raw_token_dim)
        raw = raw.reshape(B, n_x * n_y, -1)               # (B, N, raw_token_dim)

        tokens = self.proj(raw)                            # (B, N, d_model)

        pe = _make_2d_sinusoidal_pe(
            n_x, n_y, self.cfg.d_model, device, self.cfg.pe_max_freq
        )
        tokens = tokens + pe.unsqueeze(0)

        return self.norm(tokens)                           # (B, N, d_model)


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
        from ..end_to_end.utils.end_to_end_helpers import scatter_to_map  # lazy — breaks circular import
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
