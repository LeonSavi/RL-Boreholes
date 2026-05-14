"""
decision_simulator/pomdp/policies/greedy_policy.py

Greedy action selection: choose the candidate location with highest predicted ore.
"""

from __future__ import annotations

import numpy as np


def select_greedy_action(
    preds: np.ndarray,
    unvisited: list[tuple[int, int]],
) -> tuple[tuple[int, int], float]:
    """Select the unvisited location with highest predicted ore value."""
    best_idx = int(np.argmax(preds))
    best_loc = unvisited[best_idx]
    best_pred = float(preds[best_idx])

    return best_loc, best_pred
