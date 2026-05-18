"""Raw borehole belief encoder — placeholder for future implementation.

RawBoreholeBeliefEncoder will process raw borehole observation sequences
directly rather than working with a pre-encoded spatial grid. Intended for
settings where the JEPA encoder is not available or where direct sequence
modeling of drill holes is preferred over spatial latent grids.

Conceptual design (not yet implemented)
---------------------------------------
  * Input: a variable-length sequence of (x, y, ore_value) borehole
    observations in observation order.
  * Architecture: likely a lightweight transformer or RNN that attends
    over the sequence of boreholes without requiring a fixed grid.
  * Output: a map belief latent (same shape as MapBeliefTransformer's CLS
    token) for drop-in compatibility with downstream tasks.

This stub raises NotImplementedError immediately so that any accidental
usage fails loudly rather than silently.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RawBoreholeBeliefEncoder(nn.Module):
    """Placeholder: future raw borehole sequence belief encoder.

    Processes raw borehole observations (location + ore value sequences)
    directly, without requiring a pre-trained spatial encoder or a fixed
    spatial grid representation.

    Not yet implemented — raises ``NotImplementedError`` on construction.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "RawBoreholeBeliefEncoder is not yet implemented. "
            "Use UNetBelief or MapBeliefTransformer instead."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
