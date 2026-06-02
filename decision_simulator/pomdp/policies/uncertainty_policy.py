from __future__ import annotations

import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.policies.base_policy import BasePolicy


class UncertaintyPolicy(BasePolicy):
    """Select the unobserved cell with the highest predicted uncertainty.

    Requires a belief model that outputs an uncertainty map.
    Uncertainty values are in normalised model space (see BeliefState docs);
    only their relative ordering matters for selection.
    """

    def select_next_borehole(self, belief_state: BeliefState) -> tuple[int, int]:
        if belief_state.predicted_uncertainty_map is None:
            raise ValueError(
                "UncertaintyPolicy requires a belief model that outputs an "
                "uncertainty map (predicted_uncertainty_map is None)."
            )
        unc = belief_state.predicted_uncertainty_map.copy()
        unc[belief_state.observed_mask] = -np.inf
        if np.all(np.isinf(unc)):
            raise RuntimeError("No unobserved cells remain.")
        i, j = np.unravel_index(np.argmax(unc), unc.shape)
        return int(i), int(j)
