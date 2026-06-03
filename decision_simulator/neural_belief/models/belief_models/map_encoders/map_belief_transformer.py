"""Transformer-based geological belief encoder for sparse borehole maps.

MapBeliefTransformer is a reconstruction-pretraining model that ingests all
grid cells of a partially-observed borehole map as tokens and produces:

  * A single map-level latent (CLS token) summarising the geological belief
    state of the whole map — the primary output for downstream tasks such as
    total ore prediction, next-drill value estimation, or RL policies.

  * Per-cell ore value predictions via a lightweight reconstruction head —
    used for supervised pretraining to ensure the latent contains meaningful
    geological information.

Note on naming
--------------
This model uses reconstruction pretraining rather than the latent-prediction
objective of the original JEPA paper, so it is called MapBeliefTransformer
rather than "Map-JEPA". The JEPA-style training objective may be introduced
in a future iteration.

Architecture
------------

  SpatialTokenEmbedding
      Converts the flat (B, C, H, W) input into one token per grid cell.
      Each token carries the cell's raw features (coordinates, observation
      mask, ore value, borehole latent) plus a continuous 2-D sinusoidal
      positional encoding.

  MapBeliefEncoder
      Transformer encoder that processes the full set of spatial tokens plus
      one learnable CLS token.  The CLS output is the map belief latent.
      All 1024 spatial outputs are kept for the reconstruction head.

  OreReconstructionHead
      A per-cell MLP that maps each spatial transformer output independently
      to a scalar predicted ore value and reshapes back to the 2-D grid.

  MapBeliefTransformer
      Top-level module that chains the three components above.  Provides the
      same forward(x) -> (B, 1, n_x, n_y) interface as UNetBelief, plus
      encode(x) -> (B, d_model) for downstream use.

Input format (same as UNetBelief / GeologicalBeliefDataset)
-----------------------------------------------------------
  (B, 2 + latent_dim, n_x, n_y)  float32
    channel 0 : sparse observed ore value (0 at unobserved cells)
    channel 1 : binary observation mask   (1 = drilled, 0 = not drilled)
    channels 2+: borehole encoder latent  (zero vector at unobserved cells)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ....training_utils import make_coordinate_grid


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MapBeliefConfig:
    """Hyperparameters for the MapBeliefTransformer.

    All architectural dimensions live here so that the model can be
    reconstructed from a saved checkpoint without relying on external
    training configuration.
    """

    # Input
    latent_dim: int = 128       # borehole encoder output dimension
    n_x: int = 32               # spatial grid rows
    n_y: int = 32               # spatial grid columns

    # Transformer
    d_model: int = 256          # token embedding dim; also the map latent dim
    n_heads: int = 8            # attention heads (must divide d_model)
    n_encoder_layers: int = 4   # number of transformer encoder layers
    d_ff: int = 1024            # feedforward dim inside each layer (4 × d_model)
    dropout: float = 0.1

    # Reconstruction head
    head_hidden_dim: int = 128  # hidden dim of the per-cell ore prediction MLP

    # Positional encoding
    pe_max_freq: float = 10000.0  # denominator base for sinusoidal frequencies

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                f"d_model={self.d_model} must be even for 2-D sinusoidal PE "
                f"(d_model/2 dims for x-axis, d_model/2 dims for y-axis)"
            )

    @property
    def in_channels(self) -> int:
        """Total input channels: ore + mask + latent."""
        return 2 + self.latent_dim

    @property
    def raw_token_dim(self) -> int:
        """Per-cell feature dimension before projection: x, y, mask, ore, latent."""
        return 4 + self.latent_dim  # = 132 at default latent_dim=128

    @property
    def n_tokens(self) -> int:
        """Number of spatial tokens (= grid cells)."""
        return self.n_x * self.n_y


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

    Extends the 1-D sinusoidal PE from ``encoder/jepa_encoder.py`` to two
    spatial dimensions by splitting ``d_model`` evenly: the first ``d_model//2``
    dimensions encode the x-axis (row), the last ``d_model//2`` encode the
    y-axis (column).

    Using continuous coordinates in [0, 1] rather than integer grid indices
    means the encoding is resolution-independent — the same frequency ladder
    applies to any (n_x, n_y) without retraining.

    Returns
    -------
    pe : (n_x * n_y, d_model) float32 tensor
    """
    d_half = d_model // 2  # dims allocated to each spatial axis

    # Continuous coordinate grids in [0, 1] — matches make_coordinate_grid()
    x_coords = torch.linspace(0.0, 1.0, n_x, device=device)  # (n_x,)
    y_coords = torch.linspace(0.0, 1.0, n_y, device=device)  # (n_y,)

    # Expand to full grid and flatten to (N, 1)
    x_flat = x_coords[:, None].expand(n_x, n_y).reshape(n_x * n_y, 1)  # (N, 1)
    y_flat = y_coords[None, :].expand(n_x, n_y).reshape(n_x * n_y, 1)  # (N, 1)

    # Frequency terms — identical formula to jepa_encoder.py:138-141
    div_term = torch.exp(
        torch.arange(0, d_half, 2, device=device, dtype=torch.float32)
        * -(math.log(pe_max_freq) / d_half)
    )  # (d_half // 2,)

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
        # Linear projection from raw token features to transformer dimension
        self.proj = nn.Linear(cfg.raw_token_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, n_x, n_y = x.shape
        device = x.device

        # --- Build coordinate channels (same grid used by make_coordinate_grid) ---
        # coords_np: (2, n_x, n_y) float32 with x in [0,1] and y in [0,1]
        coords_np = make_coordinate_grid(n_x, n_y)  # numpy (2, n_x, n_y)
        coords = torch.from_numpy(coords_np).to(device=device, dtype=x.dtype)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, n_x, n_y)

        # --- Assemble raw per-cell feature tensor ---
        # Layout: [x_norm, y_norm, mask, ore_value, latent...]
        ore  = x[:, 0:1, :, :]   # (B, 1, n_x, n_y) — sparse observed ore
        mask = x[:, 1:2, :, :]   # (B, 1, n_x, n_y) — binary drilled flag
        lat  = x[:, 2:,  :, :]   # (B, latent_dim, n_x, n_y) — borehole latent

        raw = torch.cat([coords, mask, ore, lat], dim=1)  # (B, raw_token_dim, n_x, n_y)

        # --- Reshape to token sequence ---
        # permute so features are last, then flatten spatial dims
        raw = raw.permute(0, 2, 3, 1)               # (B, n_x, n_y, raw_token_dim)
        raw = raw.reshape(B, n_x * n_y, -1)         # (B, N, raw_token_dim)

        # --- Linear projection ---
        tokens = self.proj(raw)                      # (B, N, d_model)

        # --- Add 2-D sinusoidal positional encoding (fixed, no parameters) ---
        pe = _make_2d_sinusoidal_pe(
            n_x, n_y, self.cfg.d_model, device, self.cfg.pe_max_freq
        )  # (N, d_model)
        tokens = tokens + pe.unsqueeze(0)            # (B, N, d_model)

        return self.norm(tokens)                     # (B, N, d_model)


# ---------------------------------------------------------------------------
# Component 2: MapBeliefEncoder
# ---------------------------------------------------------------------------

class MapBeliefEncoder(nn.Module):
    """Transformer encoder producing a global map belief latent.

    A learnable CLS token is prepended to the sequence of spatial tokens.
    After running the full transformer, the CLS output is taken as the
    map-level belief latent — a single vector summarising the geological
    state of the partially observed map.

    The 1024 spatial token outputs are also returned so the reconstruction
    head can predict per-cell ore values without a separate decoder.

    Pre-LayerNorm (norm_first=True) is used for training stability, which
    is important here because we process long sequences (1025 tokens) across
    several layers.

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

        # Learnable CLS token — initialised following ViT convention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,   # input shape: (B, T, d_model)
            norm_first=True,    # Pre-LN: more stable for deep / long-sequence transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.n_encoder_layers)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]

        # Prepend CLS token — it will attend to all spatial tokens
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)         # (B, 1 + N, d_model)

        # Full self-attention over CLS + all spatial tokens.
        # No padding mask is used: unobserved cells have mask=0 and zero latent
        # embedded in their feature vector, which the transformer uses to
        # distinguish observed from unobserved cells naturally.
        out = self.transformer(seq)                   # (B, 1 + N, d_model)

        cls_out     = out[:, 0, :]                    # (B, d_model)
        spatial_out = out[:, 1:, :]                   # (B, N, d_model)

        return cls_out, spatial_out


# ---------------------------------------------------------------------------
# Component 3: OreReconstructionHead
# ---------------------------------------------------------------------------

class OreReconstructionHead(nn.Module):
    """Per-cell MLP that maps transformer outputs to ore value predictions.

    Each spatial token is processed independently through a shared 2-layer
    MLP, producing one scalar ore prediction per grid cell.  The flat output
    is reshaped back to the (B, 1, n_x, n_y) format that the rest of the
    pipeline expects.

    This head is used only for reconstruction pretraining.  Once training is
    complete, downstream tasks consume the map belief latent (CLS token) from
    MapBeliefEncoder instead.

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
        # Initialise the final layer with near-zero outputs to avoid large MSE
        # spikes in the first training steps
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.trunc_normal_(self.mlp[-1].weight, std=0.02)

    def forward(
        self, spatial_tokens: torch.Tensor, n_x: int, n_y: int
    ) -> torch.Tensor:
        B = spatial_tokens.shape[0]

        # Apply MLP independently to each token — (B, N, d_model) -> (B, N, 1)
        ore_flat = self.mlp(spatial_tokens)           # (B, N, 1)
        ore_flat = ore_flat.squeeze(-1)               # (B, N)

        return ore_flat.reshape(B, 1, n_x, n_y)      # (B, 1, n_x, n_y)


# ---------------------------------------------------------------------------
# Top-level model: MapBeliefTransformer
# ---------------------------------------------------------------------------

class MapBeliefTransformer(nn.Module):
    """Transformer-based geological belief encoder with reconstruction pretraining.

    Converts sparse borehole observations into:
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

    # ------------------------------------------------------------------
    # Internal helper — single forward pass returning all outputs
    # ------------------------------------------------------------------

    def _forward_all(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full pipeline and return (ore_map, map_belief_latent).

        x : (B, 2 + latent_dim, n_x, n_y)
        returns:
            ore_map           : (B, 1, n_x, n_y)
            map_belief_latent : (B, d_model)
        """
        n_x, n_y = x.shape[2], x.shape[3]

        tokens               = self.token_embed(x)                    # (B, N, d_model)
        cls_out, spatial_out = self.encoder(tokens)                   # (B, d), (B, N, d)
        ore_map              = self.ore_head(spatial_out, n_x, n_y)   # (B, 1, n_x, n_y)

        return ore_map, cls_out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the ore map from sparse observations.

        Drop-in replacement for UNetBelief.forward().

        Parameters
        ----------
        x : (B, 2 + latent_dim, n_x, n_y)

        Returns
        -------
        ore_map : (B, 1, n_x, n_y)  — predicted ore distribution (normalised space)
        """
        ore_map, _ = self._forward_all(x)
        return ore_map

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the partially observed map into a belief latent vector.

        This is the primary output for downstream tasks: total ore prediction,
        next-drill value, mine/abandon decisions, RL policy conditioning, etc.

        Parameters
        ----------
        x : (B, 2 + latent_dim, n_x, n_y)

        Returns
        -------
        latent : (B, d_model)  — map belief latent embedding
        """
        _, latent = self._forward_all(x)
        return latent

    def forward_with_latent(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both the ore reconstruction and the map belief latent.

        Use this during training when both outputs are needed, to avoid
        running the forward pass twice.

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
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Older checkpoints and notebooks may reference the class by its previous name.
MapBeliefModel = MapBeliefTransformer
