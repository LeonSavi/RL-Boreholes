from __future__ import annotations

import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.policies.base_policy import BasePolicy


class HybridPolicy(BasePolicy):
    """Select the unobserved cell maximising a weighted combination of ore and uncertainty.

    Score = alpha * normalised_ore + beta * normalised_uncertainty

    Both maps are independently normalised to [0, 1] over unobserved cells
    before combining, so alpha and beta are directly interpretable as relative
    importance weights regardless of the absolute scale of each map.

    Uncertainty is in normalised model space (see BeliefState docs); only its
    relative ordering across cells matters for the final score.

    Parameters
    ----------
    alpha : float
        Weight for the predicted ore component (default 1.0).
    beta : float
        Weight for the predicted uncertainty component (default 0.5).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta

    def select_next_borehole(self, belief_state: BeliefState) -> tuple[int, int]:
        if belief_state.predicted_uncertainty_map is None:
            raise ValueError(
                "HybridPolicy requires a belief model that outputs an "
                "uncertainty map (predicted_uncertainty_map is None)."
            )

        mask = belief_state.observed_mask
        unobs = ~mask

        if not unobs.any():
            raise RuntimeError("No unobserved cells remain.")

        ore_norm = _minmax_normalize(belief_state.predicted_ore_map, unobs)
        unc_norm = _minmax_normalize(belief_state.predicted_uncertainty_map, unobs)

        score = self.alpha * ore_norm + self.beta * unc_norm
        score[mask] = -np.inf

        i, j = np.unravel_index(np.argmax(score), score.shape)
        return int(i), int(j)


def _minmax_normalize(arr: np.ndarray, unobs_mask: np.ndarray) -> np.ndarray:
    """Normalise arr to [0, 1] using min/max computed over unobserved cells."""
    vals = arr[unobs_mask]
    lo, hi = float(vals.min()), float(vals.max())
    rng = hi - lo
    if rng > 0:
        return (arr - lo) / rng
    return np.zeros_like(arr)
