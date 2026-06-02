from __future__ import annotations

import numpy as np

from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.pomdp.observations.borehole_observations import (
    compute_ore_value,
    extract_borehole,
    standardise_borehole,
)


class DrillingEnvironment:
    """Hidden-state environment for the geological drilling POMDP.

    Stores the true subsurface map and exposes borehole data on demand.
    The true map is never exposed to the policy — observations flow only
    through BoreholeObservationState to the neural belief model.

    Responsibilities
    ----------------
    - Store the hidden true map.
    - Provide standardised borehole profiles at requested grid cells.
    - Provide true ore values at requested grid cells.
    - NOT track which cells have been drilled (that is BoreholeObservationState).
    - NOT decide where to drill.
    """

    def __init__(
        self,
        true_map: dict,
        resources: DecisionSimulationResources,
    ) -> None:
        self._true_map = true_map
        self._resources = resources
        self.n_x, self.n_y = true_map["yield_field"].shape[:2]

    def get_true_ore_map(self) -> np.ndarray:
        """Return peak ore value per cell as a (n_x, n_y) float32 array.

        Handles both 2-D yield fields ``(n_x, n_y)`` and 3-D depth stacks
        ``(n_x, n_y, depth)``; the latter is reduced by taking the maximum
        across the depth axis.
        """
        yield_field = self._true_map["yield_field"]
        return yield_field.max(axis=2) if yield_field.ndim == 3 else yield_field

    def get_borehole(self, i: int, j: int) -> np.ndarray:
        """Extract and standardise the borehole profile at grid cell (i, j).

        Returns
        -------
        np.ndarray of shape (V, D) — V variables, D depth layers, float32,
        standardised using training z-score statistics.
        """
        raw = extract_borehole(self._true_map, i, j, self._resources.variable_names)
        return standardise_borehole(
            raw, self._resources.norm_stats, self._resources.variable_names
        )

    def get_ore_value(self, i: int, j: int) -> float:
        """Return the peak ore yield at cell (i, j)."""
        return compute_ore_value(self._true_map, i, j)

    def drill(self, i: int, j: int) -> tuple[np.ndarray, float]:
        """Reveal the standardised borehole and ore value at (i, j).

        Returns
        -------
        borehole  : (V, D) float32 — standardised borehole profile
        ore_value : float — peak ore yield at this location
        """
        borehole = self.get_borehole(i, j)
        ore_value = self.get_ore_value(i, j)
        return borehole, ore_value
