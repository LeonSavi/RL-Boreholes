"""Geological belief model architectures — public API.

Available models
----------------
UNetBelief
    Lightweight 2-level convolutional U-Net. Fast to train, no explicit
    map-level latent. Good baseline.

MapBeliefTransformer
    Transformer encoder that treats each grid cell as a token and produces
    a global CLS-token latent representing the full geological belief state.
    Use encode(x) to obtain the belief latent for downstream tasks.

All models share the same input/output contract:
    forward(x)  : (B, 2+latent_dim, n_x, n_y) → (B, 1, n_x, n_y)
"""

from .belief_models.map_encoders.unet_belief import UNetBelief
from .belief_models.model_configs import MapBeliefConfig
from .belief_models.map_encoder_components.components import (
    SpatialTokenEmbedding,
    MapBeliefEncoder,
    OreReconstructionHead,
)
from .belief_models.end_to_end.precomp_bh_map_belief_transformer import (
    PreCompBHMapBeliefTransformer,
    MapBeliefTransformer,    # backward-compat alias
    MapBeliefModel,          # backward-compat alias
)
from .belief_models.borehole_encoders.autoencoder import (
    AEConfig,
    BoreholeEncoder,
    BoreholeDecoder,
    BoreholeAutoencoder,
    standardise,
    unstandardise,
    save_checkpoint,
    load_checkpoint,
)
from .belief_models.borehole_encoders.jepa_encoder import (
    JEPAConfig,
    BoreholeConvBackbone,
    BoreholeTokenEncoder,
    LatentPredictor,
    JEPAModel,
    sample_context_target_positions,
    sample_context_target_masks,
    save_jepa_checkpoint,
    load_jepa_checkpoint,
)

__all__ = [
    # U-Net reconstruction model
    "UNetBelief",
    # Transformer belief encoder
    "MapBeliefConfig",
    "SpatialTokenEmbedding",
    "MapBeliefEncoder",
    "OreReconstructionHead",
    "PreCompBHMapBeliefTransformer",
    "MapBeliefTransformer",  # backward-compat alias
    "MapBeliefModel",        # backward-compat alias
    # Borehole autoencoder
    "AEConfig",
    "BoreholeEncoder",
    "BoreholeDecoder",
    "BoreholeAutoencoder",
    "standardise",
    "unstandardise",
    "save_checkpoint",
    "load_checkpoint",
    # JEPA encoder
    "JEPAConfig",
    "BoreholeConvBackbone",
    "BoreholeTokenEncoder",
    "LatentPredictor",
    "JEPAModel",
    "sample_context_target_positions",
    "sample_context_target_masks",
    "save_jepa_checkpoint",
    "load_jepa_checkpoint",
]
