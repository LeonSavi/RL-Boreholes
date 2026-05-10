"""
decision_simulator/pomdp/policies/greedy_policy.py

Per-seed greedy drilling simulation policy.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from simulator.map_generator import MapGenerator, SimConfig

from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.config_decision_experiments import GreedyConfig
from decision_simulator.pomdp.observations.borehole_observations import (
    drill_at,
    run_initial_random_drills,
)

from decision_simulator.typing import DrillObservation


def run_greedy_simulation(
    seed: int,
    cfg: GreedyConfig,
    resources: DecisionSimulationResources,
    device: str,
    verbose: bool = True,
) -> tuple[list[dict], dict, str]:
    """Generate one map and run the full greedy drilling loop.

    Returns
    -------
    observations : list[dict]  - one dict per drilled location
    true_map     : dict        - full generated map (ground truth)
    decision     : str         - "MINE" or "ABANDON"
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

    all_candidates: list[list[int, int]] = [
        [i, j] for i in range(n_x) for j in range(n_y)
    ]
    rng = np.random.default_rng(seed)

    drill_kwargs = dict(
        true_map=true_map,
        resources=resources,
        device=device,
        verbose=verbose,
    )

    # Phase 1: initial random drilling
    observations: list[DrillObservation]
    observations, unvisited = run_initial_random_drills(
        all_candidates, cfg.initial_random_drills, rng, drill_kwargs
    )

    # Phase 2: greedy selection
    n_drills = cfg.drilling_budget - cfg.initial_random_drills
    if verbose:
        print(f"\nPhase 2 - greedy selection ({n_drills} drills)")

    for step in range(cfg.initial_random_drills + 1, cfg.drilling_budget + 1):
        obs_locs = np.array(
            [[o["location"][0], o["location"][1]] for o in observations],
        )
        obs_latents = np.array([o["latent"] for o in observations], dtype=np.float32)
        obs_ores = np.array([o["ore_value"] for o in observations], dtype=np.float32)

        X_train = np.concatenate([obs_locs, obs_latents], axis=1)  # (N_obs, 2+D_lat)

        k = min(cfg.k_neighbors, len(observations))
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_train, obs_ores)

        # Proxy latent: nearest observed borehole in (x, y) space
        cand_locs = np.array(unvisited, dtype=np.float32)  # (N_u, 2)
        spatial_dists = np.linalg.norm(
            cand_locs[:, None, :] - obs_locs[None, :, :], axis=-1
        )  # (N_u, N_obs)
        nearest_idx = spatial_dists.argmin(axis=1)  # (N_u,)
        proxy_latents = obs_latents[nearest_idx]  # (N_u, D_lat)

        X_cand = np.concatenate([cand_locs, proxy_latents], axis=1)  # (N_u, 2+D_lat)
        preds = knn.predict(X_cand)  # (N_u,)

        best_idx = int(np.argmax(preds))
        best_loc = unvisited[best_idx]
        best_pred = float(preds[best_idx])

        obs = drill_at(best_loc, step, pred_ore=best_pred, **drill_kwargs)
        observations.append(obs)
        unvisited.pop(best_idx)

    # Decision ----------------------------------------------------------------
    best_observed_ore = max(o["ore_value"] for o in observations)
    decision = "MINE" if best_observed_ore >= cfg.mine_threshold else "ABANDON"

    if verbose:
        print(f"\nBest observed ore value: {best_observed_ore:.4f}")
        print(f"Decision: {decision}")

    return observations, true_map, decision
