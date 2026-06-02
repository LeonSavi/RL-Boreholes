from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeliefState:
    """Snapshot of the agent's belief about the subsurface after a set of observations.

    Attributes
    ----------
    predicted_ore_map : (n_x, n_y) float32
        Predicted ore values in ore-value space (after applying the inverse
        of the target normalizer via NeuralBeliefUpdater).
    predicted_uncertainty_map : (n_x, n_y) float32 or None
        Per-cell prediction uncertainty in the model's *normalised* space
        (i.e. the same space the model was trained in — log1p or z-score).
        None when the belief model does not produce uncertainty estimates.
        These values are intentionally NOT denormalised: uncertainty
        has no meaningful inverse transform through log1p/z-score, and
        policies use it only in a relative sense (higher = more uncertain).
    observed_mask : (n_x, n_y) bool
        True at cells that have already been drilled.
    latent : (d_model,) float32 or None
        Optional map-level CLS token from the belief transformer encoder.
        Useful for downstream conditioning of RL or MCTS agents.
    step : int or None
        The drill step index at which this belief was computed.
    """

    predicted_ore_map: np.ndarray
    predicted_uncertainty_map: np.ndarray | None
    observed_mask: np.ndarray
    latent: np.ndarray | None = field(default=None)
    step: int | None = field(default=None)
