"""Belief model training pipelines — public API.

Training functions
------------------
train_neural_belief(resources, cfg, device, checkpoint_dir, ...)
    Train UNetBelief. Config: NeuralBeliefTrainingConfig.

train_map_belief(resources, cfg, device, checkpoint_dir, ...)
    Train MapBeliefTransformer. Config: MapBeliefTrainingConfig.

train_raw_borehole_belief(...)
    Planned future training pipeline. Raises NotImplementedError.

Checkpoint loading
------------------
load_belief_checkpoint(path, device)         → UNetBelief
load_map_belief_checkpoint(path, device)     → MapBeliefTransformer

All public symbols are re-exported here so that both the new canonical import
paths and the original ``decision_simulator.neural_belief.training.*`` paths
continue to work without modification.
"""

from .training_configs import (
    NeuralBeliefTrainingConfig,
    MapBeliefTrainingConfig,
    E2ETrainingConfig,
)
from .train_unet_belief import (
    train_neural_belief,
    load_belief_checkpoint,
    save_experiment_config,
)
from .train_map_belief import (
    train_map_belief,
    load_map_belief_checkpoint,
)
from .train_raw_borehole_belief import train_raw_borehole_belief
from .train_end_to_end import (
    E2EDataset,
    collate_e2e,
    train_end_to_end,
    load_e2e_checkpoint,
)

# Shared validation utilities re-exported here for callers that previously
# imported validate_by_drill_bins from decision_simulator.neural_belief.training
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    save_val_plots,
    export_history,
    build_training_config,
    load_model_encoder_checkpoint,
)
from ..sequential_eval import validate_by_step, save_sequential_val_plots
from ..datasets import build_sequential_dataset_from_cache

__all__ = [
    # UNet pipeline
    "NeuralBeliefTrainingConfig",
    "train_neural_belief",
    "load_belief_checkpoint",
    "save_experiment_config",
    # Transformer pipeline
    "MapBeliefTrainingConfig",
    "train_map_belief",
    "load_map_belief_checkpoint",
    # Placeholder
    "train_raw_borehole_belief",
    # End-to-end candidate scoring
    "E2ETrainingConfig",
    "E2EDataset",
    "collate_e2e",
    "train_end_to_end",
    "load_e2e_checkpoint",
    # Shared utilities
    "build_training_config",
    "load_model_encoder_checkpoint",
    "export_history",
    # Shared utilities (backward-compat re-exports)
    "DRILL_BINS",
    "validate_by_drill_bins",
    "validate",
    "save_val_plots",
    # Sequential evaluation
    "validate_by_step",
    "save_sequential_val_plots",
    "build_sequential_dataset_from_cache",
]
