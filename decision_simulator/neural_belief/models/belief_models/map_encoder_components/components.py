"""Reusable map encoder components.

These classes are the building blocks used by both
PreCompBHMapBeliefTransformer and all end-to-end map belief models.

SpatialTokenEmbedding
    Defined in BH_to_map_encoder_components.spatial_projection_layer;
    re-exported here for backward compatibility.

MapBeliefEncoder
    Transformer encoder with a learnable CLS token that produces a global
    map belief latent (CLS output) and per-cell spatial representations.

OreReconstructionHead
    Per-cell MLP used during pretraining to reconstruct ore values from
    spatial transformer outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..BH_to_map_encoder_components.spatial_projection_layer import SpatialTokenEmbedding as SpatialTokenEmbedding
from ..model_configs import MapBeliefConfig


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


# ---------------------------------------------------------------------------
# Component 4: UncertaintyHead
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """Per-cell MLP head predicting spatial prediction uncertainty.

    Structurally identical to OreReconstructionHead, with softplus applied to
    the final output to enforce strictly non-negative uncertainty estimates.

    Input
    -----
    spatial_tokens  : (B, n_x * n_y, d_model)
    n_x, n_y        : int — passed at runtime to support variable grid sizes

    Output
    ------
    uncertainty_map : (B, 1, n_x, n_y)  — strictly non-negative (softplus)
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
        unc_flat = self.mlp(spatial_tokens)           # (B, N, 1)
        unc_flat = F.softplus(unc_flat.squeeze(-1))   # (B, N) — strictly positive
        return unc_flat.reshape(B, 1, n_x, n_y)      # (B, 1, n_x, n_y)
