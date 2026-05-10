from __future__ import annotations

import numpy as np
from simulator.map_generator import MapGenerator, SimConfig

from decision_simulator.resources import DecisionSimulationResources


def sample_particles(
    resources: DecisionSimulationResources,
    sim_cfg: SimConfig,
    n_particles: int,
    seed: int,
) -> list[dict]:
    """Generate `n_particles` maps using seeds derived from `seed`."""
    rng = np.random.default_rng(seed)
    particle_seeds = rng.integers(0, 2**31, size=n_particles).tolist()
    particles = []
    for ps in particle_seeds:
        gen = MapGenerator(
            resources.distribution_bank,
            resources.formation_geometry,
            sim_cfg,
            seed=int(ps),
            prior=resources.discovery_prior,
        )
        particles.append(next(gen))
    return particles


def compute_particle_weights(
    particle_ore_maps: np.ndarray,
    observations: list[dict],
    temperature: float,
) -> np.ndarray:
    """Weight particles by ore consistency with observations.

    Parameters
    ----------
    particle_ore_maps : (n_particles, n_x, n_y) precomputed peak ore per cell.

    Squared error at each observed location accumulates across observations.
    weight = exp(-total_error / temperature), then normalised to sum to 1.
    Falls back to uniform weights if all weights collapse to zero.
    """
    n = particle_ore_maps.shape[0]
    if not observations:
        return np.ones(n, dtype=np.float64) / n

    obs_is = np.array([o["location"][0] for o in observations])
    obs_js = np.array([o["location"][1] for o in observations])
    true_ores = np.array([o["ore_value"] for o in observations])

    particle_ores = particle_ore_maps[:, obs_is, obs_js]  # (n_particles, n_obs)
    errors = ((particle_ores - true_ores[np.newaxis, :]) ** 2).sum(axis=1)

    # Particles with lower error become exponentially more likely.
    weights = np.exp(-errors / temperature)
    total = weights.sum()
    if total < 1e-300:
        return np.ones(n, dtype=np.float64) / n
    return weights / total


def score_candidates_by_expected_ore(
    particle_ore_maps: np.ndarray,
    weights: np.ndarray,
    unvisited: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Return weighted expected ore for every unvisited location.

    Parameters
    ----------
    particle_ore_maps : (n_particles, n_x, n_y) precomputed peak ore per cell.
    """
    cand_is = np.array([i for (i, _) in unvisited])
    cand_js = np.array([j for (_, j) in unvisited])
    candidate_ores = particle_ore_maps[:, cand_is, cand_js]
    expected = weights @ candidate_ores  # (n_unvisited,)
    return {loc: float(expected[m]) for m, loc in enumerate(unvisited)}
