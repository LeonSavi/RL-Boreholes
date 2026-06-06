"""End-to-end map belief model — public API."""

from ..borehole_encoder_components.components import (
    BoreholeTransformerEncoder,
    CatVarBoreholeTransformerEncoder,
    PatchBoreholeTransformerEncoder,
    PatchBoreholeCLSTransformerEncoder,
    VariableAwarePatchBoreholeTransformerEncoder,
)
from ..model_configs import (
    BaseE2EArchConfig,
    CatVarEndToEndConfig,
    EndToEndMapBeliefConfig,
    PatchBoreholeEndToEndConfig,
    PatchBoreholeCLSEndToEndConfig,
    VariableAwarePatchBoreholeEndToEndConfig,
)
from .end_to_end_map_belief_transformer import EndToEndMapBeliefTransformer
from .utils.model_configs import (
    BaseBHEncoderConfig,
    CatVarBoreholeConfig,
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
from .cat_var_encoder import CatVarEncoder
from .ore_only_null_encoder import OreOnlyNullEncoder

__all__ = [
    # End-to-end model configs (top-level)
    "BaseE2EArchConfig",
    "CatVarEndToEndConfig",
    "EndToEndMapBeliefConfig",
    "PatchBoreholeEndToEndConfig",
    "PatchBoreholeCLSEndToEndConfig",
    "VariableAwarePatchBoreholeEndToEndConfig",
    # Borehole encoder configs
    "BaseBHEncoderConfig",
    "CatVarBoreholeConfig",
    "E2EConfig",
    "PatchBoreholeConfig",
    "PatchBoreholeCLSConfig",
    "VariableAwarePatchBoreholeConfig",
    # Borehole encoder modules
    "BoreholeTransformerEncoder",
    "CatVarBoreholeTransformerEncoder",
    "PatchBoreholeTransformerEncoder",
    "PatchBoreholeCLSTransformerEncoder",
    "VariableAwarePatchBoreholeTransformerEncoder",
    # End-to-end models
    "CatVarEncoder",
    "OreOnlyNullEncoder",
    "EndToEndMapBeliefTransformer",
    "PatchBoreholeEndToEndMapBeliefTransformer",
    "PatchBoreholeCLSEndToEndMapBeliefTransformer",
    "VariableAwarePatchBoreholeEndToEndMapBeliefTransformer",
    "PreCompBHMapBeliefTransformer",
    "MapBeliefTransformer",
    "MapBeliefModel",
]
