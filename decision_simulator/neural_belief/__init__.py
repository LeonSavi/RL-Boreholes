from .models import (
    UNetBelief,
    MapBeliefConfig,
    PreCompBHMapBeliefTransformer,
    MapBeliefTransformer,   # backward-compat alias
    MapBeliefModel,         # backward-compat alias
)
from .datasets import (
    BeliefDatasetConfig,
    GeologicalBeliefDataset,
)
from .map_cache import NpzMap
from .training_utils import TargetNormalizer
from .training import (
    NeuralBeliefTrainingConfig,
    train_neural_belief,
    load_belief_checkpoint,
    MapBeliefTrainingConfig,
    train_map_belief,
    load_map_belief_checkpoint,
)
from .inference import (
    predict_ore_map,
    predict_from_observations,
    build_latent_map_from_observations,
    build_ore_map_from_observations,
)
from .baselines import evaluate_baselines
from decision_simulator.utils.plotting import plot_belief_sample
from .colab import (
    train_belief_from_colab,
    compare_belief_encoders_from_colab,
    train_sequential_belief_from_colab,
    train_end_to_end_from_colab,
)

__all__ = [
    # Models
    "UNetBelief",
    "MapBeliefConfig",
    "PreCompBHMapBeliefTransformer",
    "MapBeliefTransformer",
    "MapBeliefModel",
    # Dataset
    "BeliefDatasetConfig",
    "GeologicalBeliefDataset",
    "NpzMap",
    # Utils
    "TargetNormalizer",
    # UNet training
    "NeuralBeliefTrainingConfig",
    "train_neural_belief",
    "load_belief_checkpoint",
    # Transformer training
    "MapBeliefTrainingConfig",
    "train_map_belief",
    "load_map_belief_checkpoint",
    # Inference
    "predict_ore_map",
    "predict_from_observations",
    "build_latent_map_from_observations",
    "build_ore_map_from_observations",
    # Misc
    "evaluate_baselines",
    "plot_belief_sample",
    "train_belief_from_colab",
    "compare_belief_encoders_from_colab",
    "train_sequential_belief_from_colab",
    "train_end_to_end_from_colab",
]
