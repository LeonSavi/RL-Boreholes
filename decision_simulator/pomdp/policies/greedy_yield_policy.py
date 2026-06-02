from __future__ import annotations

import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.policies.base_policy import BasePolicy


class GreedyYieldPolicy(BasePolicy):
    """Select the unobserved cell with the highest predicted ore value."""

    def select_next_borehole(self, belief_state: BeliefState) -> tuple[int, int]:
        ore = belief_state.predicted_ore_map.copy()
        ore[belief_state.observed_mask] = -np.inf
        if np.all(np.isinf(ore)):
            raise RuntimeError("No unobserved cells remain.")
        i, j = np.unravel_index(np.argmax(ore), ore.shape)
        return int(i), int(j)
