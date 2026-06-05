"""Training configuration dataclasses for all belief-model training pipelines."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ...models.belief_models.model_configs import (
    BaseE2EArchConfig,
    BaseMapArchConfig,
    CatVarEndToEndConfig,
    EndToEndMapBeliefConfig,
    MapBeliefConfig,
    PatchBoreholeCLSEndToEndConfig,
    PatchBoreholeEndToEndConfig,
    VariableAwarePatchBoreholeEndToEndConfig,
)


@dataclass
class NeuralBeliefTrainingConfig:
    """Hyper-parameters for training the neural belief updater."""

    # --- dataset ---
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # --- model ---
    # in_channels must equal 2 + latent_dim:
    #   ch 0   : sparse ore map
    #   ch 1   : observation mask
    #   ch 2.. : JEPA latent vector (latent_dim channels)
    in_channels: int = 130
    base_channels: int = 64

    # --- target normalization ---
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # --- optimisation ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- early stopping ---
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0

    # --- coordinate channels ---
    use_coordinate_channels: bool = False

    # --- sequential dataset ---
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])
    sequential_seed: int = 42

    # --- false-positive penalty ---
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    false_positive_threshold: float = 0.05

    # --- misc ---
    seed: int = 42
    latent_dim: int = 128
    borehole_encoder: str = "unknown"
    n_val_plots: int = 20

    def __post_init__(self) -> None:
        expected = 2 + self.latent_dim
        if self.in_channels != expected:
            raise ValueError(
                f"NeuralBeliefTrainingConfig: in_channels={self.in_channels} does not "
                f"match 2 + latent_dim = {expected}. "
                f"UNetBelief input layout is [ore_map, mask, JEPA_latent], so "
                f"in_channels must always equal 2 + latent_dim. "
                f"Either set in_channels={expected} or latent_dim={self.in_channels - 2}."
            )


@dataclass
class MapBeliefTrainingConfig(BaseMapArchConfig):
    """Hyperparameters for training the MapBeliefTransformer.

    Architectural fields (latent_dim, n_x, n_y, d_model, n_heads,
    n_encoder_layers, d_ff, dropout, head_hidden_dim, pe_max_freq) are
    inherited from BaseMapArchConfig and shared with MapBeliefConfig so that
    each field is declared exactly once.
    """

    # --- Dataset ---
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # --- Target normalisation ---
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # --- Optimisation ---
    batch_size: int = 16  # smaller than UNet due to transformer attention memory
    lr: float = 1e-4  # lower than UNet; transformers train more stably at low lr
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- Gradient clipping (important for transformer stability) ---
    grad_clip_norm: float = 1.0  # 0.0 = disabled

    # --- Early stopping ---
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0

    # --- Sequential dataset ---
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])
    sequential_seed: int = 42

    # --- False-positive penalty ---
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    false_positive_threshold: float = 0.05

    # --- Misc ---
    seed: int = 42
    borehole_encoder: str = "unknown"
    n_val_plots: int = 20

    @property
    def in_channels(self) -> int:
        """Total input channels: ore + mask + latent."""
        return 2 + self.latent_dim

    def to_model_config(self) -> MapBeliefConfig:
        """Construct a MapBeliefConfig from the architectural fields of this dataclass."""
        return MapBeliefConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(MapBeliefConfig)}
        )


@dataclass
class BaseE2ETrainingConfig(BaseE2EArchConfig):
    """Common hyperparameters shared by all end-to-end map-belief training configs.

    Architectural fields (n_variables, n_depth, bh_d_model, bh_n_heads,
    bh_n_layers, latent_dim, n_x, n_y, d_model, n_heads, n_encoder_layers,
    d_ff, dropout, head_hidden_dim, pe_max_freq) are inherited from
    BaseE2EArchConfig and shared with the model-side ``*EndToEndConfig``
    classes so that each field is declared exactly once.
    """

    # Dataset
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # Sequential dataset mode: ordered drill sequences at fixed prefix lengths
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])

    # Target normalisation
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # Optimisation
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50
    grad_clip_norm: float = 1.0  # 0.0 = disabled

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0

    # False-positive penalty
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    false_positive_threshold: float = 0.05

    # Misc
    seed: int = 42
    borehole_encoder: str = "end_to_end"
    n_val_plots: int = 20


@dataclass
class E2EMapBeliefConfig(BaseE2ETrainingConfig):
    """Hyperparameters for training EndToEndMapBeliefTransformer (CNN borehole encoder)."""

    # CNN borehole encoder channel progression
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    borehole_encoder: str = "end_to_end"

    def to_model_config(self) -> EndToEndMapBeliefConfig:
        """Build an EndToEndMapBeliefConfig from the architectural fields."""
        return EndToEndMapBeliefConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(EndToEndMapBeliefConfig)}
        )


@dataclass
class PatchBoreholeConfig(BaseE2ETrainingConfig):
    """Training hyperparameters for PatchBoreholeEndToEndMapBeliefTransformer."""

    # Depth patch size for the patch-based borehole encoder
    bh_patch_size: int = 20

    borehole_encoder: str = "patch_borehole"

    def to_model_config(self) -> PatchBoreholeEndToEndConfig:
        """Build a PatchBoreholeEndToEndConfig for model construction."""
        return PatchBoreholeEndToEndConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(PatchBoreholeEndToEndConfig)}
        )


@dataclass
class PatchBoreholeCLSConfig(BaseE2ETrainingConfig):
    """Training hyperparameters for PatchBoreholeCLSEndToEndMapBeliefTransformer."""

    # Depth patch size for the CLS-token patch borehole encoder
    bh_patch_size: int = 20

    borehole_encoder: str = "patch_borehole_cls"

    def to_model_config(self) -> PatchBoreholeCLSEndToEndConfig:
        """Build a PatchBoreholeCLSEndToEndConfig for model construction."""
        return PatchBoreholeCLSEndToEndConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(PatchBoreholeCLSEndToEndConfig)}
        )


@dataclass
class VariableAwarePatchBoreholeConfig(BaseE2ETrainingConfig):
    """Training hyperparameters for VariableAwarePatchBoreholeEndToEndMapBeliefTransformer."""

    # Depth patch size; encoder creates one token per (variable, patch) pair
    bh_patch_size: int = 20

    borehole_encoder: str = "variable_aware_patch"

    def to_model_config(self) -> VariableAwarePatchBoreholeEndToEndConfig:
        """Build a VariableAwarePatchBoreholeEndToEndConfig for model construction."""
        return VariableAwarePatchBoreholeEndToEndConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(VariableAwarePatchBoreholeEndToEndConfig)}
        )


@dataclass
class VariableAwarePatchBoreholeUncertaintyConfig(BaseE2ETrainingConfig):
    """Training hyperparameters for the uncertainty-head variant of VariableAwarePatchBorehole."""

    # Depth patch size; encoder creates one token per (variable, patch) pair
    bh_patch_size: int = 20

    # Uncertainty head
    use_uncertainty_head: bool = True
    uncertainty_weight: float = 0.1

    borehole_encoder: str = "variable_aware_patch_uncertainty"

    def to_model_config(self) -> VariableAwarePatchBoreholeEndToEndConfig:
        """Build a VariableAwarePatchBoreholeEndToEndConfig for model construction."""
        return VariableAwarePatchBoreholeEndToEndConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(VariableAwarePatchBoreholeEndToEndConfig)}
        )


@dataclass
class CatVarConfig(BaseE2ETrainingConfig):
    """Training hyperparameters for CatVarEncoder.

    Extends BaseE2ETrainingConfig with:
    - bh_patch_size  : depth patch size (encoder creates one token per (variable, patch) pair)
    - n_rock_types   : rock-type vocab size — must match labels_vocab.pkl from 4_pull_maps.py
    - n_formations   : formation vocab size — same source
    - uncertainty_weight : weight for the uncertainty MSE loss term

    The training objective mirrors the uncertainty variant:
      ore_loss         = MSE(pred_ore, target)
      uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
      total_loss       = ore_loss + uncertainty_weight * uncertainty_loss
    """

    bh_patch_size: int = 20

    # Vocabulary sizes — must agree with the dataset's labels_vocab.pkl.
    # Index 0 is reserved for 'other'/unknown in both vocabularies.
    n_rock_types: int = 20
    n_formations: int = 40

    # Uncertainty head
    use_uncertainty_head: bool = True
    uncertainty_weight: float = 0.1

    borehole_encoder: str = "cat_var"

    def to_model_config(self) -> CatVarEndToEndConfig:
        """Build a CatVarEndToEndConfig for model construction."""
        return CatVarEndToEndConfig(
            **{f.name: getattr(self, f.name) for f in dataclasses.fields(CatVarEndToEndConfig)}
        )


@dataclass
class GuidedExplorationConfig(E2EMapBeliefConfig):
    """Extends E2EMapBeliefConfig with guided-curriculum knobs."""

    # --- mixing ---
    p_guided: float = 0.25          # fraction of sequences generated with guided drilling

    # --- within-guided mode split (must sum to 1.0) ---
    p_mode_a: float = 0.50          # Mode A: uncertainty-only
    p_mode_ab: float = 0.25         # Mode AB: uncertainty + ore-assisted
    p_mode_ac: float = 0.25         # Mode AC: uncertainty + boundary-assisted

    # --- phase transition ---
    phase1_drills: int = 1          # random drills before guided phase begins

    # --- oracle feature thresholds ---
    ore_top_pct: float = 0.10       # top-10% ore cells used for ore-assisted selection
    boundary_ore_pct: float = 0.50  # percentile of nonzero ore used as boundary threshold

    # --- guide model for Mode A ---
    # Path to a checkpoint produced by train_variable_aware_patch_borehole_uncertainty_transformer.py
    # or any model whose forward() returns (pred_ore, pred_uncertainty) or just pred_ore.
    # Leave None to use the distance-from-drills heuristic instead.
    guide_ckpt_path: str | None = None

    # 0 = static guide (never refreshed); N = reload guide from lagged training ckpt every N epochs.
    # Lagged update only works when guide_ckpt_path is None and the trained model has the
    # same architecture as a model that can produce uncertainty (e.g. same EndToEndMapBeliefTransformer).
    guide_update_interval: int = 0

    # --- visualisation ---
    n_viz_maps: int = 5             # training maps for which trajectory PNGs are saved

    def __post_init__(self) -> None:
        super().__post_init__()
        total = self.p_mode_a + self.p_mode_ab + self.p_mode_ac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"p_mode_a + p_mode_ab + p_mode_ac must sum to 1.0, got {total}"
            )
