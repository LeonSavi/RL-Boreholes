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

RawBoreholeBeliefEncoder
    Planned future model. Raises NotImplementedError on construction.

All models share the same input/output contract:
    forward(x)  : (B, 2+latent_dim, n_x, n_y) → (B, 1, n_x, n_y)
"""

from .map_encoders.unet_belief import UNetBelief
from .map_encoders.map_belief_transformer import (
    MapBeliefConfig,
    SpatialTokenEmbedding,
    MapBeliefEncoder,
    OreReconstructionHead,
    MapBeliefTransformer,
    MapBeliefModel,          # backward-compat alias for MapBeliefTransformer
)
from .map_encoders.raw_borehole_belief import RawBoreholeBeliefEncoder
from .borehole_encoders.autoencoder import (
    AEConfig,
    BoreholeEncoder,
    BoreholeDecoder,
    BoreholeAutoencoder,
    standardise,
    unstandardise,
    save_checkpoint,
    load_checkpoint,
)
from .borehole_encoders.jepa_encoder import (
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
    "MapBeliefTransformer",
    "MapBeliefModel",        # backward-compat alias
    # Future model (placeholder)
    "RawBoreholeBeliefEncoder",
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
