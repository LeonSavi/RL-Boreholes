"""Belief model training pipelines — public API.

Training functions
------------------
train_neural_belief(resources, cfg, device, checkpoint_dir, ...)
    Train UNetBelief. Config: NeuralBeliefTrainingConfig.

train_map_belief(resources, cfg, device, checkpoint_dir, ...)
    Train MapBeliefTransformer. Config: MapBeliefTrainingConfig.

train_end_to_end_map_belief(resources, cfg, device, checkpoint_dir, ...)
    Train EndToEndMapBeliefTransformer. Config: E2EMapBeliefConfig.

train_patch_borehole_transformer(resources, cfg, device, checkpoint_dir, ...)
    Train PatchBoreholeEndToEndMapBeliefTransformer. Config: PatchBoreholeConfig.

train_patch_borehole_cls_transformer(resources, cfg, device, checkpoint_dir, ...)
    Train PatchBoreholeCLSEndToEndMapBeliefTransformer. Config: PatchBoreholeCLSConfig.

train_variable_aware_patch_borehole_transformer(resources, cfg, device, checkpoint_dir, ...)
    Train VariableAwarePatchBoreholeEndToEndMapBeliefTransformer. Config: VariableAwarePatchBoreholeConfig.

Checkpoint loading
------------------
load_belief_checkpoint(path, device)              → UNetBelief
load_map_belief_checkpoint(path, device)          → MapBeliefTransformer
load_e2e_map_belief_checkpoint(path, device)      → EndToEndMapBeliefTransformer
load_patch_borehole_checkpoint(path, device)      → PatchBoreholeEndToEndMapBeliefTransformer
load_patch_borehole_cls_checkpoint(path, device)  → PatchBoreholeCLSEndToEndMapBeliefTransformer
load_variable_aware_patch_borehole_checkpoint(path, device)  → VariableAwarePatchBoreholeEndToEndMapBeliefTransformer

All public symbols are re-exported here so that callers importing from
``decision_simulator.neural_belief.training`` continue to work.
"""

from .belief_models.training_configs import (
    NeuralBeliefTrainingConfig,
    MapBeliefTrainingConfig,
    BaseE2ETrainingConfig,
    E2EMapBeliefConfig,
    PatchBoreholeConfig,
    PatchBoreholeCLSConfig,
    VariableAwarePatchBoreholeConfig,
    VariableAwarePatchBoreholeUncertaintyConfig,
)
from .belief_models.map_encoders.train_unet_belief import (
    train_neural_belief,
    load_belief_checkpoint,
)
from .belief_models.map_encoders.train_map_belief import (
    train_map_belief,
    load_map_belief_checkpoint,
)
from .belief_models.end_to_end.train_end_to_end_map_belief import (
    E2EMapDataset,
    collate_e2e_map,
    train_end_to_end_map_belief,
    load_e2e_map_belief_checkpoint,
)
from .belief_models.end_to_end.train_patch_borehole_transformer import (
    train_patch_borehole_transformer,
    load_patch_borehole_checkpoint,
)
from .belief_models.end_to_end.train_patch_borehole_cls_transformer import (
    train_patch_borehole_cls_transformer,
    load_patch_borehole_cls_checkpoint,
)
from .belief_models.end_to_end.train_variable_aware_patch_borehole_transformer import (
    train_variable_aware_patch_borehole_transformer,
    load_variable_aware_patch_borehole_checkpoint,
)
from .belief_models.end_to_end.train_variable_aware_patch_borehole_uncertainty_transformer import (
    train_variable_aware_patch_uncertainty_borehole_transformer,
    load_variable_aware_patch_uncertainty_borehole_checkpoint,
)
from .belief_models.end_to_end.train_guided_exploration_belief import (
    GuidedExplorationConfig,
    GuidedE2EMapDataset,
    GuidedTrainingStats,
    train_guided_exploration_belief,
    load_guided_belief_checkpoint,
)

# Shared validation utilities re-exported for backward compatibility
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    save_val_plots,
    export_history,
    save_experiment_config,
    build_training_config,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from .belief_models.end_to_end.helpers import validate_by_step, save_sequential_val_plots

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
    # E2E config base
    "BaseE2ETrainingConfig",
    # End-to-end map belief
    "E2EMapBeliefConfig",
    "E2EMapDataset",
    "collate_e2e_map",
    "train_end_to_end_map_belief",
    "load_e2e_map_belief_checkpoint",
    # Patch borehole experiment (mean pooling)
    "PatchBoreholeConfig",
    "train_patch_borehole_transformer",
    "load_patch_borehole_checkpoint",
    # Patch borehole experiment (CLS token)
    "PatchBoreholeCLSConfig",
    "train_patch_borehole_cls_transformer",
    "load_patch_borehole_cls_checkpoint",
    # Variable-aware patch borehole experiment
    "VariableAwarePatchBoreholeConfig",
    "train_variable_aware_patch_borehole_transformer",
    "load_variable_aware_patch_borehole_checkpoint",
    # Variable-aware patch borehole + uncertainty head
    "VariableAwarePatchBoreholeUncertaintyConfig",
    "train_variable_aware_patch_uncertainty_borehole_transformer",
    "load_variable_aware_patch_uncertainty_borehole_checkpoint",
    # Guided-exploration curriculum
    "GuidedExplorationConfig",
    "GuidedE2EMapDataset",
    "GuidedTrainingStats",
    "train_guided_exploration_belief",
    "load_guided_belief_checkpoint",
    # Shared utilities
    "build_training_config",
    "load_model_encoder_checkpoint",
    "save_checkpoint_model",
    "export_history",
    # Shared utilities (backward-compat re-exports)
    "DRILL_BINS",
    "validate_by_drill_bins",
    "validate",
    "save_val_plots",
    # Sequential evaluation
    "validate_by_step",
    "save_sequential_val_plots",
]
