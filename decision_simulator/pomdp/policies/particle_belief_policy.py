from __future__ import annotations

import numpy as np
from simulator.map_generator import MapGenerator, SimConfig

from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.config_decision_experiments import ParticleBeliefConfig
from decision_simulator.pomdp.observations.borehole_observations import (
    drill_at,
    run_initial_random_drills,
)
from decision_simulator.pomdp.beliefs.particle_belief import (
    sample_particles,
    compute_particle_weights,
    score_candidates_by_expected_ore,
)


def run_particle_belief_simulation(
    seed: int,
    cfg: ParticleBeliefConfig,
    resources: DecisionSimulationResources,
    device: str,
    verbose: bool = True,
) -> tuple[list[dict], dict, str]:
    """Generate one true map and run particle-belief drilling.

    Returns
    -------
    observations : list[dict]
    true_map     : dict
    decision     : "MINE" | "ABANDON"
    """
    if verbose:
        print(f"\nGenerating synthetic map (seed={seed})...")
    sim_cfg = SimConfig()
    gen = MapGenerator(
        resources.distribution_bank,
        resources.formation_geometry,
        sim_cfg,
        seed=seed,
        prior=resources.discovery_prior,
    )
    true_map = next(gen)

    n_x, n_y = sim_cfg.n_x, sim_cfg.n_y
    if verbose:
        print(f"  map size  : {n_x}x{n_y}  ({n_x * n_y} candidate locations)")
        print(f"  ore bodies: {len(true_map['bodies'])}")
        print(f"  sampling {cfg.n_particles} particles...")

    # Particle seeds offset to avoid overlapping with experiment seeds (0..999).
    particles = sample_particles(
        resources, sim_cfg, cfg.n_particles, seed=seed + 1_000_000
    )

    # Potential state maps
    particle_ore_maps = np.stack(
        [p["yield_field"].max(axis=2) for p in particles]
    )  # (n_particles, n_x, n_y)

    all_candidate_borehole_coords: list[tuple[int, int]] = [
        (i, j) for i in range(n_x) for j in range(n_y)
    ]
    rng = np.random.default_rng(seed)

    drill_kwargs = dict(
        true_map=true_map,
        resources=resources,
        device=device,
        verbose=verbose,
    )

    # Phase 1: initial random drilling
    observations, unvisited = run_initial_random_drills(
        all_candidate_borehole_coords, cfg.initial_random_drills, rng, drill_kwargs
    )

    # Phase 2: particle belief selection
    n_active = cfg.drilling_budget - cfg.initial_random_drills
    if verbose:
        print(f"\nPhase 2 - particle belief selection ({n_active} drills)")

    for step in range(cfg.initial_random_drills + 1, cfg.drilling_budget + 1):
        weights = compute_particle_weights(
            particle_ore_maps, observations, cfg.temperature
        )

        scores = score_candidates_by_expected_ore(particle_ore_maps, weights, unvisited)

        best_loc = max(scores, key=scores.__getitem__)
        pred_ore = scores[best_loc]

        obs = drill_at(best_loc, step, pred_ore=pred_ore, **drill_kwargs)
        observations.append(obs)
        unvisited.remove(best_loc)

    # Decision
    best_observed_ore = max(o["ore_value"] for o in observations)
    decision = "MINE" if best_observed_ore >= cfg.mine_threshold else "ABANDON"

    if verbose:
        print(f"\nBest observed ore value: {best_observed_ore:.4f}")
        print(f"Decision: {decision}")

    return observations, true_map, decision
