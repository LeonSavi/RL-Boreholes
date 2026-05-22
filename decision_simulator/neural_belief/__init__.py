from .models.map_encoders.unet_belief import UNetBelief
from .models.map_encoders.map_belief_transformer import (
    MapBeliefConfig,
    MapBeliefTransformer,
    MapBeliefModel,  # backward-compat alias
)
from .datasets import (
    BeliefDatasetConfig,
    GeologicalBeliefDataset,
    build_dataset_from_cache,
    build_sequential_dataset_from_cache,
)
from .map_cache import NpzMapCache, NpzMapCacheStore
from .utils import TargetNormalizer
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
from .visualize import plot_belief_sample
from .colab import (
    train_belief_from_colab,
    compare_belief_encoders_from_colab,
    train_sequential_belief_from_colab,
)

__all__ = [
    # Models
    "UNetBelief",
    "MapBeliefConfig",
    "MapBeliefTransformer",
    "MapBeliefModel",
    # Dataset
    "BeliefDatasetConfig",
    "GeologicalBeliefDataset",
    "NpzMapCache",
    "NpzMapCacheStore",
    "build_dataset_from_cache",
    "build_sequential_dataset_from_cache",
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
]
