"""
decision_simulator/pomdp/policies/particle_belief_policy.py

Particle-belief action selection: choose the candidate with highest expected ore under belief.
"""

from __future__ import annotations


def select_particle_belief_action(
    scores: dict[tuple[int, int], float],
) -> tuple[tuple[int, int], float]:
    """Select the location with highest expected ore under the current belief."""
    best_loc = max(scores, key=scores.__getitem__)
    best_score = scores[best_loc]

    return best_loc, best_score
