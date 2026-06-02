from __future__ import annotations

import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.policies.base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """Select a uniformly random unobserved cell."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_next_borehole(self, belief_state: BeliefState) -> tuple[int, int]:
        unobserved = np.argwhere(~belief_state.observed_mask)
        if len(unobserved) == 0:
            raise RuntimeError("No unobserved cells remain.")
        idx = self._rng.integers(len(unobserved))
        return int(unobserved[idx, 0]), int(unobserved[idx, 1])
