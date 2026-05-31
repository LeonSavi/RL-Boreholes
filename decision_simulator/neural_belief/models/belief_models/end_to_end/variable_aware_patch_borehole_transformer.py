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

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..map_encoders.map_belief_transformer import (
    MapBeliefConfig,
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)
from .end_to_end_helpers import sinusoidal_pe_1d

# ---------------------------------------------------------------------------
# Borehole encoder config
# ---------------------------------------------------------------------------


@dataclass
class VariableAwarePatchBoreholeConfig:
    """Hyperparameters for VariableAwarePatchBoreholeTransformerEncoder."""

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # Depth patch size — depth is split into non-overlapping patches of this size
    bh_patch_size: int = 20

    # Transformer over patch tokens
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Final borehole embedding dimension
    latent_dim: int = 128

    # Shared dropout rate
    dropout: float = 0.1

    # Fields forwarded from VariableAwarePatchBoreholeEndToEndConfig — used for validation only
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    head_hidden_dim: int = 128
    pe_max_freq: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.bh_n_layers > 0 and self.bh_d_model % self.bh_n_heads != 0:
            raise ValueError(
                f"bh_d_model={self.bh_d_model} must be divisible by "
                f"bh_n_heads={self.bh_n_heads}"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for 2D sinusoidal positional encoding"
            )

    @property
    def n_patches(self) -> int:
        """Number of depth patches (depth is zero-padded to a multiple of bh_patch_size)."""
        return math.ceil(self.n_depth / self.bh_patch_size)


# ---------------------------------------------------------------------------
# Borehole encoder
# ---------------------------------------------------------------------------


class VariableAwarePatchBoreholeTransformerEncoder(nn.Module):
    """Variable-aware patch borehole encoder with CLS-token pooling.

    Architecture:
      1. Divide the depth axis into non-overlapping patches of size bh_patch_size.
         Depth is zero-padded to the nearest multiple of bh_patch_size if needed.
      2. Each (variable, patch) pair becomes one token of size bh_patch_size —
         giving n_variables * n_patches tokens per borehole.
      3. A linear layer projects each token from bh_patch_size to bh_d_model.
      4. A learned variable embedding is added (distinguishes variable identity).
      5. 1D sinusoidal positional encoding is added (encodes depth position).
      6. A learned CLS token is prepended before the transformer.
      7. A small transformer captures cross-variable and cross-depth dependencies.
      8. CLS token output (index 0) + linear projection → latent_dim embedding.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: VariableAwarePatchBoreholeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        self.padded_depth = self.n_patches * cfg.bh_patch_size

        # Each token contains one variable × one depth patch
        self.patch_proj = nn.Linear(cfg.bh_patch_size, cfg.bh_d_model)

        # Learned variable embedding — gives each variable a unique identity
        self.var_embed = nn.Embedding(cfg.n_variables, cfg.bh_d_model)
        nn.init.trunc_normal_(self.var_embed.weight, std=0.02)

        # Learned CLS token — initialised with trunc_normal (std=0.02, same as ViT/JEPA)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.bh_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if cfg.bh_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.bh_d_model,
                nhead=cfg.bh_n_heads,
                dim_feedforward=cfg.bh_d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, cfg.bh_n_layers
            )
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

        n_bh_tokens = cfg.n_variables * self.n_patches
        n_params = sum(p.numel() for p in self.parameters())
        print(
            "VariableAwarePatchBoreholeTransformerEncoder: variable-aware patching enabled"
        )
        print(
            f"  n_variables={cfg.n_variables}, patch_size={cfg.bh_patch_size}, "
            f"n_patches={self.n_patches}"
        )
        print(
            f"  total borehole tokens = {cfg.n_variables} × {self.n_patches} + 1 CLS"
            f" = {n_bh_tokens + 1}"
        )
        print(
            f"  bh_d_model={cfg.bh_d_model}, n_layers={cfg.bh_n_layers}, "
            f"n_heads={cfg.bh_n_heads}, latent_dim={cfg.latent_dim}, "
            f"params={n_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        B, V, D = x.shape
        pad_len = self.padded_depth - D
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # (B, V, padded_depth)

        # Create one token per (variable, depth patch)
        x = x.view(B, V, self.n_patches, self.cfg.bh_patch_size)
        x = x.reshape(B, V * self.n_patches, self.cfg.bh_patch_size)

        feat = self.patch_proj(x)  # (B, V * n_patches, bh_d_model)

        # Variable embedding: token v*n_patches+p belongs to variable v
        var_idx = torch.arange(V, device=x.device).repeat_interleave(self.n_patches)
        var_emb = self.var_embed(var_idx)  # (V * n_patches, bh_d_model)
        feat = feat + var_emb.unsqueeze(0)

        # Depth positional encoding: token v*n_patches+p belongs to patch p
        pe_full = sinusoidal_pe_1d(self.n_patches, self.cfg.bh_d_model, feat.device)
        patch_idx = torch.arange(self.n_patches, device=feat.device).repeat(V)
        depth_pe = pe_full[patch_idx]  # (V * n_patches, bh_d_model)
        feat = feat + depth_pe.unsqueeze(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, bh_d_model)
        feat = torch.cat([cls_tokens, feat], dim=1)  # (B, 1 + V*n_patches, bh_d_model)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)

        cls_out = feat[:, 0]  # (B, bh_d_model)
        return self.out_proj(cls_out)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# End-to-end configuration
# ---------------------------------------------------------------------------


@dataclass
class VariableAwarePatchBoreholeEndToEndConfig:
    """Hyperparameters for VariableAwarePatchBoreholeEndToEndMapBeliefTransformer."""

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder: depth patch size
    bh_patch_size: int = 20

    # Borehole encoder: transformer over (variable, patch) tokens
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Borehole embedding output dimension
    latent_dim: int = 128

    # Spatial grid
    n_x: int = 32
    n_y: int = 32

    # Map belief transformer
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    head_hidden_dim: int = 128
    pe_max_freq: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.bh_n_layers > 0 and self.bh_d_model % self.bh_n_heads != 0:
            raise ValueError(
                f"bh_d_model={self.bh_d_model} must be divisible by "
                f"bh_n_heads={self.bh_n_heads}"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for 2D sinusoidal positional encoding "
                "(d_model/2 dims for x-axis, d_model/2 dims for y-axis)"
            )

    def to_encoder_config(self) -> VariableAwarePatchBoreholeConfig:
        """Build a VariableAwarePatchBoreholeConfig for the borehole encoder."""
        return VariableAwarePatchBoreholeConfig(
            n_variables=self.n_variables,
            n_depth=self.n_depth,
            bh_patch_size=self.bh_patch_size,
            bh_d_model=self.bh_d_model,
            bh_n_heads=self.bh_n_heads,
            bh_n_layers=self.bh_n_layers,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_encoder_layers,
            d_ff=self.d_ff,
            head_hidden_dim=self.head_hidden_dim,
            pe_max_freq=self.pe_max_freq,
        )

    def to_map_belief_config(self) -> MapBeliefConfig:
        """Build a MapBeliefConfig for the map encoder components."""
        return MapBeliefConfig(
            latent_dim=self.latent_dim,
            n_x=self.n_x,
            n_y=self.n_y,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_hidden_dim=self.head_hidden_dim,
            pe_max_freq=self.pe_max_freq,
        )


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

    def _encode_boreholes(
        self,
        boreholes: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode raw boreholes to latent vectors, skipping padded rows.

        Parameters
        ----------
        boreholes    : (B, K, V, D)
        padding_mask : (B, K) bool — True at padded positions; None if no padding

        Returns
        -------
        latents : (B, K, latent_dim)
        """
        B, K = boreholes.shape[:2]

        if padding_mask is not None:
            not_padded = ~padding_mask  # (B, K)
            bh_valid = boreholes[not_padded]  # (N_valid, V, D)
            lat_valid = self.bh_encoder(bh_valid)  # (N_valid, latent_dim)
            lat = torch.zeros(
                B,
                K,
                self.cfg.latent_dim,
                device=boreholes.device,
                dtype=lat_valid.dtype,
            )
            lat[not_padded] = lat_valid
        else:
            bh_flat = boreholes.reshape(B * K, *boreholes.shape[2:])
            lat = self.bh_encoder(bh_flat).reshape(B, K, self.cfg.latent_dim)

        return lat

    def _scatter_to_map(
        self,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a dense (B, 2 + latent_dim, n_x, n_y) map from sparse observations.

        Parameters
        ----------
        ore_vals     : (B, K) — observed ore at drilled cells
        positions    : (B, K, 2) — normalised [0,1] (x, y)
        latents      : (B, K, latent_dim) — borehole embeddings
        padding_mask : (B, K) bool — True at padded positions

        Returns
        -------
        x : (B, 2 + latent_dim, n_x, n_y)
        """
        B, K = ore_vals.shape
        n_x, n_y = self.cfg.n_x, self.cfg.n_y
        N = n_x * n_y
        C = 2 + self.cfg.latent_dim
        device = ore_vals.device
        dtype = ore_vals.dtype

        i_idx = (positions[..., 0] * (n_x - 1)).round().long().clamp(0, n_x - 1)
        j_idx = (positions[..., 1] * (n_y - 1)).round().long().clamp(0, n_y - 1)
        flat_idx = i_idx * n_y + j_idx  # (B, K)

        if padding_mask is not None:
            flat_idx = flat_idx.masked_fill(padding_mask, N)
            valid = (~padding_mask).to(dtype)  # (B, K)
        else:
            valid = torch.ones(B, K, device=device, dtype=dtype)

        ore_ch = (ore_vals * valid).unsqueeze(1)  # (B, 1, K)
        mask_ch = valid.unsqueeze(1)  # (B, 1, K)
        lat_ch = (latents * valid.unsqueeze(-1)).transpose(1, 2)  # (B, latent_dim, K)
        values = torch.cat([ore_ch, mask_ch, lat_ch], dim=1)  # (B, C, K)

        flat_idx_exp = flat_idx.unsqueeze(1).expand(B, C, K)  # (B, C, K)
        out_flat = torch.zeros(B, C, N + 1, device=device, dtype=dtype)
        out_flat.scatter_(2, flat_idx_exp, values)

        return out_flat[:, :, :N].reshape(B, C, n_x, n_y)

    def _forward_all(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, map_belief_latent)."""
        latents = self._encode_boreholes(boreholes, padding_mask)
        x = self._scatter_to_map(ore_vals, positions, latents, padding_mask)
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
