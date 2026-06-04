"""End-to-end map belief model — public API."""

from ..borehole_encoder_components.components import (
    BoreholeTransformerEncoder,
    PatchBoreholeTransformerEncoder,
    PatchBoreholeCLSTransformerEncoder,
    VariableAwarePatchBoreholeTransformerEncoder,
)
from ..model_configs import (
    BaseE2EArchConfig,
    EndToEndMapBeliefConfig,
    PatchBoreholeEndToEndConfig,
    PatchBoreholeCLSEndToEndConfig,
    VariableAwarePatchBoreholeEndToEndConfig,
)
from .end_to_end_map_belief_transformer import EndToEndMapBeliefTransformer
from .model_configs import (
    BaseBHEncoderConfig,
    E2EConfig,
    PatchBoreholeConfig,
    PatchBoreholeCLSConfig,
    VariableAwarePatchBoreholeConfig,
)
from .patch_borehole_cls_transformer import PatchBoreholeCLSEndToEndMapBeliefTransformer
from .patch_borehole_transformer import PatchBoreholeEndToEndMapBeliefTransformer
from .precomp_bh_map_belief_transformer import (
    MapBeliefModel,          # backward-compat alias
    MapBeliefTransformer,    # backward-compat alias
    PreCompBHMapBeliefTransformer,
)
from .variable_aware_patch_borehole_transformer import (
    VariableAwarePatchBoreholeEndToEndMapBeliefTransformer,
)

__all__ = [
    # End-to-end model configs (top-level)
    "BaseE2EArchConfig",
    "EndToEndMapBeliefConfig",
    "PatchBoreholeEndToEndConfig",
    "PatchBoreholeCLSEndToEndConfig",
    "VariableAwarePatchBoreholeEndToEndConfig",
    # Borehole encoder configs
    "BaseBHEncoderConfig",
    "E2EConfig",
    "PatchBoreholeConfig",
    "PatchBoreholeCLSConfig",
    "VariableAwarePatchBoreholeConfig",
    # Borehole encoder modules
    "BoreholeTransformerEncoder",
    "PatchBoreholeTransformerEncoder",
    "PatchBoreholeCLSTransformerEncoder",
    "VariableAwarePatchBoreholeTransformerEncoder",
    # End-to-end models
    "EndToEndMapBeliefTransformer",
    "PatchBoreholeEndToEndMapBeliefTransformer",
    "PatchBoreholeCLSEndToEndMapBeliefTransformer",
    "VariableAwarePatchBoreholeEndToEndMapBeliefTransformer",
    "PreCompBHMapBeliefTransformer",
    "MapBeliefTransformer",
    "MapBeliefModel",
]
