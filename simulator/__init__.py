"""
simulator — synthetic subsurface + autoencoder for Task 1 of the thesis.

Pipeline:
  1. fit distributions from cleaned NLOG+LILY data (distributions.py)
  2. assemble 2D maps with stratigraphy, rock physics, orebodies
     (map_generator.py, stratigraphy.py, orebody.py)
  3. train an autoencoder on streamed synthetic boreholes (autoencoder.py,
     train.py)
  4. export the trained encoder for Task 2's POMDP
"""
from .distributions import DistributionBank, CellDistribution
from .stratigraphy import StratigraphicColumn, sample_column
from .orebody import OreBody, sample_orebodies
from .map_generator import generate_map, MapGenerator, SimConfig
from .autoencoder import (
    BoreholeAutoencoder, AEConfig,
    save_checkpoint, load_checkpoint,
)

__all__ = [
    "DistributionBank", "CellDistribution",
    "StratigraphicColumn", "sample_column",
    "OreBody", "sample_orebodies",
    "generate_map", "MapGenerator", "SimConfig",
    "BoreholeAutoencoder", "AEConfig",
    "save_checkpoint", "load_checkpoint",
]
