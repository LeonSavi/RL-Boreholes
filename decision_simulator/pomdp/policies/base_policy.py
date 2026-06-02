from __future__ import annotations

from abc import ABC, abstractmethod

from decision_simulator.pomdp.beliefs.belief_state import BeliefState


class BasePolicy(ABC):
    """Abstract base class for borehole selection policies.

    Policies receive only the BeliefState — they must not access the true map,
    raw boreholes, or any environment internals.
    """

    @abstractmethod
    def select_next_borehole(self, belief_state: BeliefState) -> tuple[int, int]:
        """Select the next borehole location from the current belief state.

        Parameters
        ----------
        belief_state : BeliefState
            Current belief over the subsurface, including the observed mask
            and predicted ore / uncertainty maps.

        Returns
        -------
        tuple[int, int]
            (i, j) grid coordinates of the selected unobserved cell.
        """
        ...
