"""Training configuration dataclasses for all belief-model training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.map_encoders.map_belief_transformer import MapBeliefConfig
from ..models.end_to_end.candidate_scoring_transformer import E2EConfig


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
class MapBeliefTrainingConfig:
    """Hyperparameters for training the MapBeliefTransformer.

    Dataset and normalisation fields are identical to NeuralBeliefTrainingConfig
    so the two training functions can be called with the same preparation code.
    Architectural fields replace the UNet-specific ``in_channels`` / ``base_channels``.
    """

    # --- Dataset (same as NeuralBeliefTrainingConfig) ---
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # --- Input / grid ---
    latent_dim: int = 128  # borehole encoder latent dim (sets in_channels)
    n_x: int = 32
    n_y: int = 32

    # --- Model architecture ---
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 1024  # feedforward dim (4 × d_model)
    dropout: float = 0.1
    head_hidden_dim: int = 128

    # --- Target normalisation (same as NeuralBeliefTrainingConfig) ---
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # --- Optimisation ---
    batch_size: int = 16  # smaller than UNet due to transformer attention memory
    lr: float = 1e-4  # lower than UNet; transformers train more stably at low lr
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- Gradient clipping (important for transformer stability) ---
    grad_clip_norm: float = 1.0  # 0.0 = disabled

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

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    @property
    def in_channels(self) -> int:
        """Total input channels: ore + mask + latent."""
        return 2 + self.latent_dim

    def to_model_config(self) -> MapBeliefConfig:
        """Construct a MapBeliefConfig from the architectural fields of this dataclass."""
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
        )


@dataclass
class E2ETrainingConfig:
    """Hyperparameters for training the end-to-end candidate scoring model."""

    # Dataset
    n_train_maps: int = 50
    samples_per_map: int = 20
    candidates_per_sample: int = (
        1  # candidates sampled per (map, drilling-history) pair
    )
    n_val_maps: int = 30
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # Sequential dataset mode: ordered drill sequences at fixed prefix lengths.
    # Mirrors the real exploration setting where each drill informs the next.
    # When False (default), K drills are sampled randomly from [min_drills, max_drills].
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])

    # Borehole dimensions — resolved from resources at training time
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder architecture
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2
    latent_dim: int = 128

    # Candidate scoring transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    head_hidden_dim: int = 128

    # Positional encoding
    pe_max_freq: float = 10000.0

    # Optimisation
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"
    grad_clip_norm: float = 1.0  # 0.0 = disabled

    # False-positive penalty: penalise high predictions where true ore = 0
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    fp_threshold: float = 1e-3  # ore values below this are treated as "no ore"

    # Experiment controls
    shuffle_boreholes: bool = False  # variant C sanity check
    pretrained_bh_encoder_path: str | None = None  # variant D: JEPA init

    # Grid dimensions — set automatically from cache in train_end_to_end()
    n_x: int = 32
    n_y: int = 32

    # Misc
    seed: int = 42
    borehole_encoder: str = "end_to_end"  # informational; stored in checkpoint
    n_val_plots: int = 20

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    def to_model_config(self) -> E2EConfig:
        """Build an E2EConfig from the architectural fields of this dataclass."""
        return E2EConfig(
            n_variables=self.n_variables,
            n_depth=self.n_depth,
            bh_channels=self.bh_channels,
            bh_d_model=self.bh_d_model,
            bh_n_heads=self.bh_n_heads,
            bh_n_layers=self.bh_n_layers,
            latent_dim=self.latent_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_hidden_dim=self.head_hidden_dim,
            pe_max_freq=self.pe_max_freq,
        )
