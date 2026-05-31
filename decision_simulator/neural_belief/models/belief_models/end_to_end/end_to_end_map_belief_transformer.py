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

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..map_encoders.map_belief_transformer import (
    MapBeliefConfig,
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)

# ---------------------------------------------------------------------------
# Positional encoding helper (used by BoreholeTransformerEncoder)
# ---------------------------------------------------------------------------


def sinusoidal_pe_1d(n_tokens: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard 1D sinusoidal positional encoding.

    Returns
    -------
    pe : (n_tokens, d_model)
    """
    pos = torch.arange(n_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_tokens, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Borehole encoder config
# ---------------------------------------------------------------------------


@dataclass
class E2EConfig:
    """Hyperparameters for BoreholeTransformerEncoder."""

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # 1D CNN backbone channel sizes
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    # Small transformer on top of CNN
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Final borehole embedding dimension (0 = no encoder)
    latent_dim: int = 128

    # Shared dropout rate
    dropout: float = 0.1

    # Fields forwarded from EndToEndMapBeliefConfig — used for validation only
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
    def n_depth_tokens(self) -> int:
        return max(1, self.n_depth // (2 ** len(self.bh_channels)))


# ---------------------------------------------------------------------------
# Borehole encoder
# ---------------------------------------------------------------------------


class BoreholeTransformerEncoder(nn.Module):
    """1D CNN + transformer borehole encoder.

    Architecture:
      1. 1D CNN backbone reduces depth from D to D/16 tokens.
      2. Small transformer over depth tokens captures long-range relationships.
      3. Mean-pool + linear projection → latent_dim embedding.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: E2EConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_ch = cfg.n_variables
        for out_ch in cfg.bh_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.seq_proj = nn.Linear(cfg.bh_channels[-1], cfg.bh_d_model)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        feat = self.conv(x)  # (B, C, T)
        feat = feat.transpose(1, 2)  # (B, T, C)
        feat = self.seq_proj(feat)  # (B, T, bh_d_model)

        T = feat.shape[1]
        pe = sinusoidal_pe_1d(T, self.cfg.bh_d_model, feat.device)
        feat = feat + pe.unsqueeze(0)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)

        pooled = feat.mean(dim=1)  # (B, bh_d_model)
        return self.out_proj(pooled)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EndToEndMapBeliefConfig:
    """Hyperparameters for EndToEndMapBeliefTransformer.

    Combines borehole encoder settings (mirroring E2EConfig) with map
    transformer settings (mirroring MapBeliefConfig) so the full model can be
    reconstructed from a checkpoint without an external training config.
    """

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder: 1D CNN backbone channel sizes
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    # Borehole encoder: small transformer on top of CNN
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

    def to_e2e_config(self) -> E2EConfig:
        """Build an E2EConfig to construct BoreholeTransformerEncoder."""
        return E2EConfig(
            n_variables=self.n_variables,
            n_depth=self.n_depth,
            bh_channels=self.bh_channels,
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

    def _encode_boreholes(
        self,
        boreholes: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode raw boreholes to latent vectors, skipping padded rows.

        Only non-padded boreholes are passed to the CNN encoder, which prevents
        zero-filled padding from polluting BatchNorm running statistics.

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

        Drilled-cell positions (normalised [0,1]) are converted to flat grid
        indices.  Padded positions are routed to a scratch slot at index N so
        they never overwrite real observations.  The scratch slot is discarded
        before the final reshape.

        Parameters
        ----------
        ore_vals     : (B, K) — observed ore at drilled cells
        positions    : (B, K, 2) — normalised [0,1] (x, y)
        latents      : (B, K, latent_dim) — borehole embeddings
        padding_mask : (B, K) bool — True at padded positions

        Returns
        -------
        x : (B, 2 + latent_dim, n_x, n_y)
            ch 0  : sparse observed ore value
            ch 1  : binary drilled mask (1 = observed)
            ch 2+ : borehole latent embedding
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
