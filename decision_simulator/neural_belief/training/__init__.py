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

from .train_unet_belief import (
    NeuralBeliefTrainingConfig,
    train_neural_belief,
    load_belief_checkpoint,
    build_training_config,
    export_history,
    save_experiment_config,
)
from .train_map_belief import (
    MapBeliefTrainingConfig,
    train_map_belief,
    load_map_belief_checkpoint,
    build_map_belief_training_config,
)
from .train_raw_borehole_belief import train_raw_borehole_belief

# Shared validation utilities re-exported here for callers that previously
# imported validate_by_drill_bins from decision_simulator.neural_belief.training
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    save_val_plots,
)

__all__ = [
    # UNet pipeline
    "NeuralBeliefTrainingConfig",
    "train_neural_belief",
    "load_belief_checkpoint",
    "build_training_config",
    "export_history",
    "save_experiment_config",
    # Transformer pipeline
    "MapBeliefTrainingConfig",
    "train_map_belief",
    "load_map_belief_checkpoint",
    "build_map_belief_training_config",
    # Placeholder
    "train_raw_borehole_belief",
    # Shared utilities (backward-compat re-exports)
    "DRILL_BINS",
    "validate_by_drill_bins",
    "validate",
    "save_val_plots",
]
