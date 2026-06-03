"""
POMDP orchestration for fixed-budget geological drilling.

Responsibility map
------------------
DrillingEnvironment      → hidden truth; reveals borehole profiles and ore values
BoreholeObservationState → accumulates drilled observations
NeuralBeliefUpdater      → observations → predicted ore map + uncertainty map
BasePolicy               → belief maps → next borehole location
pomdp.py                 → orchestration loop (this file)

The decision policy never sees the true map or raw boreholes.
Only the environment reads from the true map; only the observation state
passes standardised borehole profiles to the belief model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.beliefs.neural_belief import NeuralBeliefUpdater
from decision_simulator.pomdp.environment.drilling_environment import DrillingEnvironment
from decision_simulator.pomdp.observations.borehole_observations import (
    BoreholeObservationState,
)
from decision_simulator.pomdp.policies.base_policy import BasePolicy

# Type alias for the optional per-step callback.
# Called after each policy-driven drill with:
#   step        : current step index (1-based)
#   belief_state: BeliefState computed BEFORE the latest drill (the decision basis)
#   obs_state   : BoreholeObservationState AFTER the latest drill has been added
StepCallback = Callable[[int, BeliefState, BoreholeObservationState], None]


def run_fixed_budget_episode(
    environment: DrillingEnvironment,
    belief_updater: NeuralBeliefUpdater,
    policy: BasePolicy,
    initial_drills: int = 1,
    drilling_budget: int = 10,
    seed: int = 42,
    initial_locations: list[tuple[int, int]] | None = None,
    step_callback: StepCallback | None = None,
) -> dict:
    """Run a fixed-budget geological drilling episode.

    Loop
    ----
    1. Drill the initial borehole(s): either the explicit ``initial_locations``
       list, or ``initial_drills`` randomly selected cells (seeded).
    2. For each remaining step up to ``drilling_budget``:
       a. Compute BeliefState from current observations.
       b. Policy selects next cell from the BeliefState.
       c. Environment reveals the borehole and ore value.
       d. Update observation state.
       e. Invoke ``step_callback`` if provided.
       f. Log per-step metrics.
    3. Compute final BeliefState and return episode result.

    Parameters
    ----------
    environment : DrillingEnvironment
        Wraps the hidden true map.
    belief_updater : NeuralBeliefUpdater
        Converts observations to a BeliefState via the pretrained model.
    policy : BasePolicy
        Selects the next borehole from the current BeliefState.
    initial_drills : int
        Number of randomly selected boreholes drilled before the policy takes
        over. Ignored when ``initial_locations`` is provided. Counted as part
        of ``drilling_budget``.
    drilling_budget : int
        Total number of boreholes to drill (initial + policy-driven).
    seed : int
        Random seed for the initial drill selection (unused when
        ``initial_locations`` is provided).
    initial_locations : list of (i, j) tuples, optional
        Fixed starting locations drilled before the policy takes over.
        When given, ``initial_drills`` and ``seed`` are ignored for the
        initialisation phase.
    step_callback : callable, optional
        Called after each policy-driven drill with signature
        ``(step, belief_state, obs_state)``. ``belief_state`` is the belief
        computed BEFORE the latest drill; ``obs_state`` already contains the
        latest drill so the newest location appears in sparse-ore / mask plots.

    Returns
    -------
    dict with keys:
        final_observations      : BoreholeObservationState
        step_history            : list[dict] — one entry per policy-driven step
        final_belief_state      : BeliefState
        true_ore_map            : np.ndarray (n_x, n_y)
        final_best_observed_ore : float
    """
    rng = np.random.default_rng(seed)
    n_x, n_y = environment.n_x, environment.n_y
    obs_state = BoreholeObservationState(n_x, n_y)

    # --- Phase 1: initialisation ---
    if initial_locations is not None:
        start_coords = initial_locations
    else:
        all_coords = [(i, j) for i in range(n_x) for j in range(n_y)]
        init_indices = rng.choice(len(all_coords), size=initial_drills, replace=False)
        start_coords = [all_coords[int(idx)] for idx in init_indices]

    for i, j in start_coords:
        borehole, ore_value = environment.drill(i, j)
        obs_state.add_observation(i, j, borehole, ore_value)

    step_history: list[dict] = []
    total_true_ore = float(environment.get_true_ore_map().sum())

    # --- Phase 2: policy-driven drills ---
    n_policy_steps = drilling_budget - len(start_coords)
    for step in range(1, n_policy_steps + 1):
        # Compute belief from observations collected so far
        belief_state = belief_updater.update(obs_state, step=step)

        # Policy selects next location — sees only belief maps, not true map
        i, j = policy.select_next_borehole(belief_state)

        # Environment reveals ground truth at selected location
        borehole, ore_value = environment.drill(i, j)

        # Capture metrics from belief_state computed BEFORE this new drill
        pred_ore_at_loc = float(belief_state.predicted_ore_map[i, j])
        total_predicted_ore = float(belief_state.predicted_ore_map.sum())

        if belief_state.predicted_uncertainty_map is not None:
            unc_map = belief_state.predicted_uncertainty_map
            pred_unc_at_loc = float(unc_map[i, j])
            total_uncertainty = float(unc_map.sum())
            mean_uncertainty = float(unc_map.mean())
        else:
            pred_unc_at_loc = float("nan")
            total_uncertainty = float("nan")
            mean_uncertainty = float("nan")

        # Update observation state with the newly drilled borehole
        obs_state.add_observation(i, j, borehole, ore_value)

        # Callback receives the pre-drill belief but the post-drill obs_state
        # so plots show where the agent decided to drill (newest red X)
        if step_callback is not None:
            step_callback(step, belief_state, obs_state)

        step_history.append(
            {
                "step": step,
                "location": (i, j),
                "ore_value": ore_value,
                "predicted_ore": pred_ore_at_loc,
                "predicted_uncertainty": pred_unc_at_loc,
                "total_predicted_ore": total_predicted_ore,
                "total_uncertainty": total_uncertainty,
                "mean_uncertainty": mean_uncertainty,
            }
        )

    # Final belief state after all drills
    final_belief = belief_updater.update(obs_state, step=drilling_budget)

    return {
        "final_observations": obs_state,
        "step_history": step_history,
        "final_belief_state": final_belief,
        "true_ore_map": environment.get_true_ore_map(),
        "total_true_ore": total_true_ore,
        "final_best_observed_ore": max(obs_state.observed_ore_values),
    }


if __name__ == "__main__":
    import datetime
    from pathlib import Path

    from simulator.map_generator import MapGenerator, SimConfig
    from decision_simulator.resources import load_decision_resources
    from decision_simulator.neural_belief.training.belief_models.train_variable_aware_patch_borehole_uncertainty_transformer import (
        load_variable_aware_patch_uncertainty_borehole_checkpoint,
    )
    from decision_simulator.utils.plotting import (
        plot_belief_sample,
        plot_step_belief_grid,
        plot_policy_evolution,
        plot_policy_map_evolution,
    )
    from decision_simulator.pomdp.policies.random_policy import RandomPolicy
    from decision_simulator.pomdp.policies.greedy_yield_policy import GreedyYieldPolicy
    from decision_simulator.pomdp.policies.uncertainty_policy import UncertaintyPolicy
    from decision_simulator.pomdp.policies.hybrid_policy import HybridPolicy

    CHECKPOINT = Path(__file__).parent / "beliefs" / "variable_aware_patch_uncertainty_guided.pt"
    DEVICE = "cuda"
    DRILLING_BUDGET = 11  # 1 initial + 10 policy steps
    SELECTED_STEPS = [1, 2, 3, 5, 8, 10]  # Fibonacci-like steps + final
    N_MAPS = 10
    PLOT_DIR = Path(__file__).parent.parent / "plots"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading resources...")
    resources, _ = load_decision_resources(borehole_encoder="jepa", device=DEVICE)

    print(f"Loading belief model from {CHECKPOINT}...")
    model, _, normalizer, _ = load_variable_aware_patch_uncertainty_borehole_checkpoint(
        CHECKPOINT, device=DEVICE
    )

    updater = NeuralBeliefUpdater(model, normalizer, DEVICE)

    for map_idx in range(N_MAPS):
        print(f"\n{'#' * 60}")
        print(f"Map {map_idx:02d} / {N_MAPS - 1}")
        print(f"{'#' * 60}")

        n_ore_bodies = int(np.random.default_rng(map_idx).integers(0, 2))
        gen = MapGenerator(
            resources.distribution_bank,
            resources.formation_geometry,
            SimConfig(n_ore_bodies=n_ore_bodies),
            seed=map_idx,
            prior=resources.discovery_prior,
        )
        true_map = next(gen)
        env = DrillingEnvironment(true_map, resources)
        true_ore_map = env.get_true_ore_map()
        center = (env.n_x // 2, env.n_y // 2)
        print(f"  n_ore_bodies: {n_ore_bodies}  Initial drill: center cell {center}")

        map_results: dict[str, dict] = {}
        # step_beliefs[policy_name][n_drills] = BeliefState computed from n_drills observations
        step_beliefs: dict[str, dict[int, BeliefState]] = {}
        CAPTURE_DRILLS = {2, 3}

        policy_list = [
            ("random",       RandomPolicy(seed=42)),
            ("greedy_yield", GreedyYieldPolicy()),
            ("uncertainty",  UncertaintyPolicy()),
            ("hybrid",       HybridPolicy(alpha=1.0, beta=0.5)),
        ]

        for policy_name, policy in policy_list:
            print(f"\n  Policy: {policy_name}")

            captured: dict[int, BeliefState] = {}
            step_beliefs[policy_name] = captured

            # Capture loop variables for the closure
            def make_step_callback(
                name: str,
                m_idx: int,
                t_ore: np.ndarray,
                store: dict,
            ) -> StepCallback:
                def callback(step: int, belief_state: BeliefState, obs_state: BoreholeObservationState) -> None:
                    save_path = (
                        PLOT_DIR / name / timestamp
                        / f"map_{m_idx:02d}_step_{step:02d}.png"
                    )
                    plot_belief_sample(
                        sparse_ore_map=obs_state.get_sparse_ore_map(),
                        observation_mask=obs_state.get_observed_mask(),
                        true_ore_map=t_ore,
                        predicted_ore_map=belief_state.predicted_ore_map,
                        predicted_uncertainty_map=belief_state.predicted_uncertainty_map,
                        save_path=save_path,
                        title=f"{name} | Map {m_idx:02d} | Step {step:02d}",
                        timestamp=timestamp,
                    )
                    # belief_state at step k was computed from k observations
                    n_drills = int(belief_state.observed_mask.sum())
                    if n_drills in CAPTURE_DRILLS:
                        store[n_drills] = belief_state
                return callback

            result = run_fixed_budget_episode(
                environment=env,
                belief_updater=updater,
                policy=policy,
                drilling_budget=DRILLING_BUDGET,
                initial_locations=[center],
                step_callback=make_step_callback(policy_name, map_idx, true_ore_map, captured),
            )
            map_results[policy_name] = result

            total_true = result["total_true_ore"]
            print(f"    {'step':>4}  {'loc':>8}  {'true_ore':>8}  {'pred_ore':>8}  {'pred_unc':>8}  {'total_pred':>10}  {'total_true':>10}")
            for row in result["step_history"]:
                print(
                    f"    {row['step']:>4}  "
                    f"{str(row['location']):>8}  "
                    f"{row['ore_value']:>8.4f}  "
                    f"{row['predicted_ore']:>8.4f}  "
                    f"{row['predicted_uncertainty']:>8.4f}  "
                    f"{row['total_predicted_ore']:>10.4f}  "
                    f"{total_true:>10.4f}"
                )
            print(f"    Total true ore: {total_true:.4f}")

            evo_grid_path = (
                PLOT_DIR / policy_name / timestamp
                / f"map_{map_idx:02d}_evolution.png"
            )
            plot_policy_map_evolution(
                policy_name=policy_name,
                step_plot_dir=PLOT_DIR,
                map_idx=map_idx,
                selected_steps=SELECTED_STEPS,
                timestamp=timestamp,
                save_path=evo_grid_path,
            )
            print(f"    Evolution grid saved -> {evo_grid_path}")

        # 4×4 grid: rows=policies, cols=ore@2drills|unc@2drills|ore@3drills|unc@3drills
        step_grid_path = (
            PLOT_DIR / "step_belief_grid" / timestamp / f"map_{map_idx:02d}_step_grid.png"
        )
        plot_step_belief_grid(
            step_beliefs, map_idx, drill_counts=(2, 3),
            timestamp=timestamp, save_path=step_grid_path,
        )
        print(f"\n  Step belief grid saved -> {step_grid_path}")

        # Line plot: predicted vs true total ore over steps for all policies
        evo_path = PLOT_DIR / "evolution" / timestamp / f"map_{map_idx:02d}_evolution.png"
        plot_policy_evolution(map_results, map_idx, save_path=evo_path)
        print(f"  Policy comparison plot saved -> {evo_path}")
