"""Centralised model configuration dataclasses for all belief-model architectures.

All *EndToEndConfig and MapBeliefConfig dataclasses live here so that there is a
single source of truth for architectural field definitions.  Training configs in
``training/belief_models/training_configs.py`` inherit the shared base classes
defined here, eliminating the duplication of architectural fields across both
hierarchies.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Map-encoder architecture base
# ---------------------------------------------------------------------------


@dataclass
class BaseMapArchConfig:
    """Shared architectural fields for map-transformer-based configs."""

    latent_dim: int = 128
    n_x: int = 32
    n_y: int = 32
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
        if self.d_model % 2 != 0:
            raise ValueError(
                f"d_model={self.d_model} must be even for 2-D sinusoidal PE "
                f"(d_model/2 dims for x-axis, d_model/2 dims for y-axis)"
            )


@dataclass
class MapBeliefConfig(BaseMapArchConfig):
    """Hyperparameters for the MapBeliefTransformer.

    All architectural dimensions live here so that the model can be reconstructed
    from a saved checkpoint without relying on external training configuration.
    """

    @property
    def in_channels(self) -> int:
        """Total input channels: ore + mask + latent."""
        return 2 + self.latent_dim

    @property
    def raw_token_dim(self) -> int:
        """Per-cell feature dimension before projection: x, y, mask, ore, latent."""
        return 4 + self.latent_dim

    @property
    def n_tokens(self) -> int:
        """Number of spatial tokens (= grid cells)."""
        return self.n_x * self.n_y


# ---------------------------------------------------------------------------
# End-to-end architecture base
# ---------------------------------------------------------------------------


@dataclass
class BaseE2EArchConfig:
    """Shared architectural fields for all end-to-end map-belief configs.

    Both the model-side ``*EndToEndConfig`` classes and the training-side
    ``BaseE2ETrainingConfig`` inherit from this class so that each field is
    declared exactly once.
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

    # Spatial grid (resolved from dataset at training time)
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

    def to_map_belief_config(self) -> MapBeliefConfig:
        """Build a MapBeliefConfig from the shared map-transformer fields."""
        return MapBeliefConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(MapBeliefConfig)}
        )


# ---------------------------------------------------------------------------
# End-to-end model configs
# ---------------------------------------------------------------------------


@dataclass
class EndToEndMapBeliefConfig(BaseE2EArchConfig):
    """Hyperparameters for EndToEndMapBeliefTransformer (CNN borehole encoder).

    Combines borehole encoder settings with map transformer settings so the full
    model can be reconstructed from a checkpoint without an external training config.
    """

    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    def to_e2e_config(self):
        """Build an E2EConfig to construct BoreholeTransformerEncoder."""
        from .end_to_end.utils.model_configs import E2EConfig
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


@dataclass
class PatchBoreholeEndToEndConfig(BaseE2EArchConfig):
    """Hyperparameters for PatchBoreholeEndToEndMapBeliefTransformer.

    Combines borehole patch encoder settings with map transformer settings so the
    full model can be reconstructed from a checkpoint without an external training config.
    """

    bh_patch_size: int = 20

    def to_patch_config(self):
        """Build a PatchBoreholeConfig to construct PatchBoreholeTransformerEncoder."""
        from .end_to_end.utils.model_configs import PatchBoreholeConfig
        return PatchBoreholeConfig(
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


@dataclass
class PatchBoreholeCLSEndToEndConfig(BaseE2EArchConfig):
    """Hyperparameters for PatchBoreholeCLSEndToEndMapBeliefTransformer.

    Identical to PatchBoreholeEndToEndConfig except the borehole encoder uses
    CLS-token pooling instead of mean pooling.
    """

    bh_patch_size: int = 20

    def to_cls_config(self):
        """Build a PatchBoreholeCLSConfig to construct PatchBoreholeCLSTransformerEncoder."""
        from .end_to_end.utils.model_configs import PatchBoreholeCLSConfig
        return PatchBoreholeCLSConfig(
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


@dataclass
class VariableAwarePatchBoreholeEndToEndConfig(BaseE2EArchConfig):
    """Hyperparameters for VariableAwarePatchBoreholeEndToEndMapBeliefTransformer."""

    bh_patch_size: int = 20

    def to_encoder_config(self):
        """Build a VariableAwarePatchBoreholeConfig for the borehole encoder."""
        from .end_to_end.utils.model_configs import VariableAwarePatchBoreholeConfig
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


@dataclass
class CatVarEndToEndConfig(VariableAwarePatchBoreholeEndToEndConfig):
    """Hyperparameters for CatVarEncoder (variable-aware patch + rock-type labels).

    Extends VariableAwarePatchBoreholeEndToEndConfig with n_rock_types, which is
    inferred automatically from labels_vocab.pkl at training time.

    All map-transformer fields (d_model, n_heads, etc.) are inherited unchanged.
    """

    n_rock_types: int = 1   # auto-set from labels_vocab.pkl at training time

    def to_encoder_config(self):
        """Build a CatVarBoreholeConfig for the categorical borehole encoder."""
        from .end_to_end.utils.model_configs import CatVarBoreholeConfig
        return CatVarBoreholeConfig(
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
            n_rock_types=self.n_rock_types,
        )
