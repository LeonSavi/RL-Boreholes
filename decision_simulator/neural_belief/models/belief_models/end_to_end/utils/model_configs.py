"""Borehole encoder configuration dataclasses.

One config per encoder architecture:
  E2EConfig                        -> BoreholeTransformerEncoder  (CNN + transformer, mean-pool)
  PatchBoreholeConfig              -> PatchBoreholeTransformerEncoder  (patch tokeniser, mean-pool)
  PatchBoreholeCLSConfig           -> PatchBoreholeCLSTransformerEncoder  (patch tokeniser, CLS-pool)
  VariableAwarePatchBoreholeConfig -> VariableAwarePatchBoreholeTransformerEncoder  (per-(var,patch) token, CLS-pool)

All four share a common base class (BaseBHEncoderConfig) that holds every field
that is identical across encoders.  Subclasses add only the architecture-specific
fields (bh_channels for CNN; bh_patch_size for patch-based variants).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BaseBHEncoderConfig:
    """Shared hyperparameters for all borehole encoder variants.

    Also carries the map-belief transformer fields (d_model, n_heads, …) that
    are forwarded from the end-to-end model config solely for validation.
    """

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder transformer (shared across all encoder styles)
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Borehole embedding output dimension
    latent_dim: int = 128

    # Shared dropout rate
    dropout: float = 0.1

    # Map belief transformer fields forwarded from the end-to-end config — validation only
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


@dataclass
class E2EConfig(BaseBHEncoderConfig):
    """Hyperparameters for BoreholeTransformerEncoder (1D CNN + transformer)."""

    # 1D CNN backbone channel sizes — each entry doubles the receptive field (MaxPool1d)
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    @property
    def n_depth_tokens(self) -> int:
        """Depth sequence length after CNN downsampling."""
        return max(1, self.n_depth // (2 ** len(self.bh_channels)))


@dataclass
class PatchBoreholeConfig(BaseBHEncoderConfig):
    """Hyperparameters for patch-based borehole encoders (mean-pool variant)."""

    # Depth is split into non-overlapping patches of this size
    bh_patch_size: int = 20

    @property
    def n_patches(self) -> int:
        """Number of depth patches (depth zero-padded to multiple of bh_patch_size)."""
        return math.ceil(self.n_depth / self.bh_patch_size)


@dataclass
class PatchBoreholeCLSConfig(PatchBoreholeConfig):
    """Same fields as PatchBoreholeConfig; marks the CLS-token pooling variant."""


@dataclass
class VariableAwarePatchBoreholeConfig(PatchBoreholeConfig):
    """Same fields as PatchBoreholeConfig; marks the variable-aware encoder variant."""
