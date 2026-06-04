"""Reusable map encoder components.

These three classes are the building blocks used by both
PreCompBHMapBeliefTransformer and all end-to-end map belief models.

SpatialTokenEmbedding
    Converts a dense (B, 2+latent_dim, n_x, n_y) borehole map into one token
    per grid cell, with 2-D sinusoidal positional encoding.

MapBeliefEncoder
    Transformer encoder with a learnable CLS token that produces a global
    map belief latent (CLS output) and per-cell spatial representations.

OreReconstructionHead
    Per-cell MLP used during pretraining to reconstruct ore values from
    spatial transformer outputs.
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
# Component 1: SpatialTokenEmbedding
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
        B, C, n_x, n_y = x.shape
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


# ---------------------------------------------------------------------------
# Component 2: MapBeliefEncoder
# ---------------------------------------------------------------------------

class MapBeliefEncoder(nn.Module):
    """Transformer encoder producing a global map belief latent.

    A learnable CLS token is prepended to the sequence of spatial tokens.
    After running the full transformer, the CLS output is taken as the
    map-level belief latent.

    Pre-LayerNorm (norm_first=True) is used for training stability across
    long sequences (1025 tokens).

    Input
    -----
    tokens : (B, n_x * n_y, d_model) — from SpatialTokenEmbedding

    Output
    ------
    cls_out      : (B, d_model)              — map belief latent
    spatial_out  : (B, n_x * n_y, d_model)  — per-cell representations
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.n_encoder_layers)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)      # (B, 1 + N, d_model)

        out = self.transformer(seq)                # (B, 1 + N, d_model)

        cls_out     = out[:, 0, :]                 # (B, d_model)
        spatial_out = out[:, 1:, :]                # (B, N, d_model)

        return cls_out, spatial_out


# ---------------------------------------------------------------------------
# Component 3: OreReconstructionHead
# ---------------------------------------------------------------------------

class OreReconstructionHead(nn.Module):
    """Per-cell MLP that maps transformer outputs to ore value predictions.

    Each spatial token is processed independently through a shared 2-layer
    MLP, producing one scalar ore prediction per grid cell, reshaped back to
    (B, 1, n_x, n_y).

    Used only for reconstruction pretraining.

    Input
    -----
    spatial_tokens : (B, n_x * n_y, d_model)
    n_x, n_y       : int — passed at runtime to support variable grid sizes

    Output
    ------
    ore_map : (B, 1, n_x, n_y)
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.trunc_normal_(self.mlp[-1].weight, std=0.02)

    def forward(
        self, spatial_tokens: torch.Tensor, n_x: int, n_y: int
    ) -> torch.Tensor:
        B = spatial_tokens.shape[0]
        ore_flat = self.mlp(spatial_tokens).squeeze(-1)  # (B, N)
        return ore_flat.reshape(B, 1, n_x, n_y)
