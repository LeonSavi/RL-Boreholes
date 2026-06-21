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
    import argparse
    import csv
    import datetime
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    import h5py
    import numpy as np
    import torch

    import dataclasses

    from decision_simulator.neural_belief.map_hdf5 import HDF5MapStore, MapPool
    from decision_simulator.neural_belief.training.belief_models.end_to_end.train_cat_var_encoder import (
        load_cat_var_checkpoint,
    )
    from decision_simulator.neural_belief.training.belief_models.end_to_end.train_ore_only_null_encoder import (
        load_ore_only_null_checkpoint,
    )
    from decision_simulator.pomdp.beliefs.belief_state import BeliefState
    from decision_simulator.pomdp.observations.borehole_observations import (
        BoreholeObservationState,
    )
    from decision_simulator.pomdp.policies.greedy_yield_policy import GreedyYieldPolicy
    from decision_simulator.pomdp.policies.random_policy import RandomPolicy
    from decision_simulator.pomdp.policies.uncertainty_policy import UncertaintyPolicy
    from decision_simulator.resources import load_decision_resources
    from decision_simulator.utils.evaluate_policies import run_evaluation, _orebody_dir_name
    from decision_simulator.utils.plotting import plot_evolution_map

    # ── Argument parsing ──────────────────────────────────────────────────────

    BELIEFS_DIR = Path(__file__).parent / "beliefs"
    # Base results directory: decision_simulator/results/
    PLOT_DIR = Path(__file__).parent.parent / "results"
    SELECTED_STEPS = {1, 2, 3, 5, 8, 10}

    p = argparse.ArgumentParser(
        description="Run POMDP episodes using the cat_var or ore_only_null encoder."
    )
    p.add_argument(
        "--model",
        choices=["cat_var", "ore_only_null"],
        required=True,
        help="End-to-end encoder to evaluate.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the model checkpoint. "
            "Defaults to beliefs/cat_var_best_500.pt or beliefs/ore_only_null_best_500.pt "
            "(beliefs/cat_var_best_multiple_1000.pt or beliefs/ore_only_null_multiple_1000.pt "
            "when --multiple is set)."
        ),
    )
    p.add_argument(
        "--map-dir",
        type=Path,
        default=Path("C:/validation_datasets/One_orebody"),
        help="Directory containing pre-generated HDF5 map shards (default: C:/validation_datasets/One_orebody).",
    )
    p.add_argument(
        "--multiple",
        action="store_true",
        default=False,
        help=(
            "Switch to Two-orebody mode: sets the default map directory to "
            "C:/validation_datasets/Two_orebodies and uses the _multiple_1000 checkpoints."
        ),
    )
    p.add_argument(
        "--n-maps",
        type=int,
        default=20,
        help="Number of maps to evaluate (default: 20).",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=11,
        help="Total drilling budget including the initial drill (default: 11).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device string (default: cuda).",
    )
    args = p.parse_args()

    if args.multiple and args.map_dir == Path("C:/validation_datasets/One_orebody"):
        args.map_dir = Path("C:/validation_datasets/Two_orebodies")

    if args.checkpoint is None:
        if args.multiple:
            args.checkpoint = BELIEFS_DIR / (
                "cat_var_best_multiple_1000.pt" if args.model == "cat_var" else "ore_only_null_multiple_1000.pt"
            )
        else:
            args.checkpoint = BELIEFS_DIR / (
                "cat_var_best_500.pt" if args.model == "cat_var" else "ore_only_null_best_500.pt"
            )

    # Short label used in directory names: cat_var or only_ore
    model_label = "cat_var" if args.model == "cat_var" else "only_ore"

    orebody_folder = "multiple_orebodies" if args.multiple else "single_orebody"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # All outputs land under:
    #   decision_simulator/results/{orebody_folder}/{model_label}/{timestamp}/
    run_dir = PLOT_DIR / orebody_folder / model_label / timestamp

    # CSVs land in a prediction_summaries subfolder of the run directory:
    #   decision_simulator/results/{orebody_folder}/{model_label}/{timestamp}/prediction_summaries/
    csv_dir = run_dir / "prediction_summaries"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers (imported from helpers.py) ───────────────────────────

    from decision_simulator.pomdp.helpers import (
        HDF5DrillEnv as _HDF5DrillEnv,
        CatVarUpdater as _CatVarUpdater,
        OreOnlyNullUpdater as _OreOnlyNullUpdater,
    )

    # ── Load resources and model ──────────────────────────────────────────────

    print("Loading resources...")
    resources, _ = load_decision_resources(borehole_encoder="jepa", device=args.device)

    print(f"Loading {args.model} checkpoint: {args.checkpoint}")
    if args.model == "cat_var":
        model, _, normalizer, _ = load_cat_var_checkpoint(
            args.checkpoint, device=args.device
        )
    else:
        model, _, normalizer, _ = load_ore_only_null_checkpoint(
            args.checkpoint, device=args.device
        )
    model.eval()

    # ── Load maps from HDF5 ───────────────────────────────────────────────────

    print(f"Loading {args.n_maps} maps from {args.map_dir}...")
    shard_files = (
        sorted(args.map_dir.glob("maps_0_and_1_orebodies_*.h5"))
        or sorted(args.map_dir.glob("maps_stratified_*.h5"))
        or sorted(args.map_dir.glob("maps_[0-9][0-9][0-9][0-9][0-9]_*.h5"))
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No HDF5 shards found in {args.map_dir}. "
            "Run generate_training_maps.py first."
        )
    _pools: list[MapPool] = []
    _n_bodies_chunks: list[np.ndarray] = []
    _n_remaining = args.n_maps
    for _sf in shard_files:
        if _n_remaining <= 0:
            break
        with h5py.File(_sf, "r") as _hf:
            _n_in_shard = int(_hf.attrs["n_maps"])
            _n_from_this = min(_n_remaining, _n_in_shard)
            _n_bodies_chunks.append(_hf["n_bodies"][:_n_from_this])
        _pools.append(HDF5MapStore(_sf).load_subset(list(range(_n_from_this))))
        _n_remaining -= _n_from_this

    if len(_pools) == 1:
        npz_map = _pools[0]
    else:
        _p0 = _pools[0]
        npz_map = MapPool(
            borehole_arrays=[b for _p in _pools for b in _p.borehole_arrays],
            targets=[t for _p in _pools for t in _p.targets],
            drill_patterns=[d for _p in _pools for d in _p.drill_patterns],
            cfg=dataclasses.replace(_p0.cfg, n_maps=args.n_maps),
            n_x=_p0.n_x,
            n_y=_p0.n_y,
            rocks_arrays=[r for _p in _pools if _p.rocks_arrays for r in _p.rocks_arrays] or None,
        )
    _n_bodies_arr = np.concatenate(_n_bodies_chunks)
    print(
        f"  Loaded {npz_map.pool_size} maps  "
        f"(n_x={npz_map.n_x}, n_y={npz_map.n_y}, "
        f"rocks={'yes' if npz_map.rocks_arrays else 'no'})"
    )

    # ── CSV field names ───────────────────────────────────────────────────────

    _STEP_FIELDS = [
        "policy", "step", "loc_i", "loc_j",
        "true_ore", "predicted_ore", "predicted_uncertainty",
        "total_predicted_ore", "total_true_ore", "top_ore",
    ]
    _SUMMARY_FIELDS = ["map_idx", "policy", "total_true_ore", "best_observed_ore"]
    all_summary_rows: list[dict] = []
    all_step_rows: list[dict] = []
    # Track orebody count per map_idx for summary grouping
    _map_n_bodies: dict[int, int] = {}

    # Deferred plot task list — filled during the POMDP loop, rendered in parallel
    # at the end.  Each entry is a kwargs dict matching the target function's signature.
    evo_map_tasks: list[dict] = []         # → plot_evolution_map(**kw)
    _evo_map_counts: dict[str, int] = {}   # maps saved per policy
    _EVO_MAP_LIMIT = 50

    # ── Main evaluation loop ──────────────────────────────────────────────────

    for map_idx in range(npz_map.pool_size):
        print(f"\n{'#' * 60}")
        print(f"Map {map_idx:02d} / {npz_map.pool_size - 1}")
        print(f"{'#' * 60}")

        n_bodies_val = int(_n_bodies_arr[map_idx])
        _map_n_bodies[map_idx] = n_bodies_val

        rocks_arr = npz_map.rocks_arrays[map_idx] if npz_map.rocks_arrays else None
        env = _HDF5DrillEnv(
            bh_raw=npz_map.borehole_arrays[map_idx],
            target=npz_map.targets[map_idx],
            rocks=rocks_arr,
            n_x=npz_map.n_x,
            n_y=npz_map.n_y,
            norm_stats=resources.norm_stats,
            variable_names=resources.variable_names,
        )
        center = (env.n_x // 2, env.n_y // 2)
        true_ore_map = env.get_true_ore_map()
        top10_threshold = float(np.percentile(true_ore_map, 90))

        map_rows: list[dict] = []
        map_results: dict[str, dict] = {}

        policy_list = [
            ("uncertainty", UncertaintyPolicy()),
            ("greedy",      GreedyYieldPolicy()),
            ("random",      RandomPolicy(seed=42 + map_idx)),
        ]

        for policy_name, policy in policy_list:
            print(f"\n  Policy: {policy_name}")

            policy_plot_dir = run_dir / "plots" / policy_name
            policy_plot_dir.mkdir(parents=True, exist_ok=True)
            collected_steps: list[dict] = []

            if args.model == "cat_var":
                updater = _CatVarUpdater(model, normalizer, args.device, env)
            else:
                updater = _OreOnlyNullUpdater(model, normalizer, args.device, env)

            def make_step_callback(store: list, t_ore: np.ndarray) -> StepCallback:
                def callback(
                    step: int,
                    belief_state: BeliefState,
                    obs_state: BoreholeObservationState,
                ) -> None:
                    if step not in SELECTED_STEPS:
                        return
                    store.append(
                        dict(
                            step=step,
                            sparse_ore_map=obs_state.get_sparse_ore_map(),
                            observation_mask=obs_state.get_observed_mask(),
                            true_ore_map=t_ore.copy(),
                            predicted_ore_map=belief_state.predicted_ore_map.copy(),
                            predicted_uncertainty_map=(
                                belief_state.predicted_uncertainty_map.copy()
                                if belief_state.predicted_uncertainty_map is not None
                                else None
                            ),
                        )
                    )
                return callback

            result = run_fixed_budget_episode(
                environment=env,
                belief_updater=updater,
                policy=policy,
                drilling_budget=args.budget,
                initial_locations=[center],
                step_callback=make_step_callback(collected_steps, true_ore_map),
            )
            map_results[policy_name] = result

            total_true = result["total_true_ore"]
            print(
                f"    {'step':>4}  {'loc':>8}  {'true_ore':>8}  "
                f"{'pred_ore':>8}  {'pred_unc':>8}  "
                f"{'total_pred':>10}  {'total_true':>10}"
            )
            for row in result["step_history"]:
                i_loc, j_loc = row["location"]
                print(
                    f"    {row['step']:>4}  ({i_loc:2d},{j_loc:2d})  "
                    f"{row['ore_value']:>8.4f}  "
                    f"{row['predicted_ore']:>8.4f}  "
                    f"{row['predicted_uncertainty']:>8.4f}  "
                    f"{row['total_predicted_ore']:>10.4f}  "
                    f"{total_true:>10.4f}"
                )
                map_rows.append(
                    {
                        "policy":                policy_name,
                        "step":                  row["step"],
                        "loc_i":                 i_loc,
                        "loc_j":                 j_loc,
                        "true_ore":              round(row["ore_value"], 3),
                        "predicted_ore":         round(row["predicted_ore"], 3),
                        "predicted_uncertainty": round(row["predicted_uncertainty"], 3),
                        "total_predicted_ore":   round(row["total_predicted_ore"], 3),
                        "total_true_ore":        round(total_true, 3),
                        "top_ore":               row["ore_value"] >= top10_threshold,
                    }
                )
            print(f"    Total true ore: {total_true:.4f}")

            all_summary_rows.append(
                {
                    "map_idx":           map_idx,
                    "policy":            policy_name,
                    "total_true_ore":    round(total_true, 3),
                    "best_observed_ore": round(result["final_best_observed_ore"], 3),
                }
            )

            if _evo_map_counts.get(policy_name, 0) < _EVO_MAP_LIMIT:
                evo_map_tasks.append(
                    dict(
                        steps=collected_steps,
                        save_path=policy_plot_dir / f"map_{map_idx:02d}_evolution.png",
                        title=f"{policy_name} | Map {map_idx:02d}",
                    )
                )
                _evo_map_counts[policy_name] = _evo_map_counts.get(policy_name, 0) + 1

        # Per-map CSV → prediction_summaries/{orebody_dir}/maps/
        orebody_maps_dir = csv_dir / _orebody_dir_name(n_bodies_val) / "maps"
        orebody_maps_dir.mkdir(parents=True, exist_ok=True)
        map_csv_path = orebody_maps_dir / f"map_{map_idx:02d}.csv"
        with open(map_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_STEP_FIELDS)
            writer.writeheader()
            writer.writerows(map_rows)
        print(f"\n  Results CSV -> {map_csv_path}")
        all_step_rows.extend({**row, "map_idx": map_idx, "n_bodies": n_bodies_val} for row in map_rows)

    # ── Summary CSV ───────────────────────────────────────────────────────────

    # ── Summary CSVs (one per orebody group) ─────────────────────────────────
    from collections import defaultdict as _dd
    _summary_by_orebody: dict[int, list[dict]] = _dd(list)
    for _row in all_summary_rows:
        _summary_by_orebody[_map_n_bodies[_row["map_idx"]]].append(_row)
    for _nb, _rows in sorted(_summary_by_orebody.items()):
        _rows.sort(key=lambda r: (r["policy"], r["map_idx"]))
        _summary_dir = csv_dir / _orebody_dir_name(_nb)
        _summary_dir.mkdir(parents=True, exist_ok=True)
        _summary_path = _summary_dir / "summary.csv"
        with open(_summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(_rows)
        print(f"\nSummary ({_orebody_dir_name(_nb)}) -> {_summary_path}")

    # ── Policy evaluation metrics ─────────────────────────────────────────────
    print("\nRunning policy evaluation...")
    run_evaluation(all_step_rows, csv_dir)

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    metadata = {
        "timestamp": timestamp,
        "n_maps": npz_map.pool_size,
        "model_file": args.checkpoint.name,
        "model_label": model_label,
        "orebody_folder": orebody_folder,
        "policies": [name for name, _ in policy_list],
        "budget": args.budget,
        "grid_size": [npz_map.n_x, npz_map.n_y],
        "dataset_path": str(shard_files[0]),
        "created_at": datetime.datetime.now().isoformat(),
    }
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Metadata -> {metadata_path}")

    # ── Parallel plot rendering ───────────────────────────────────────────────
    # Uses ProcessPoolExecutor so each worker has its own matplotlib instance
    # (matplotlib is not thread-safe).

    n_workers = min(os.cpu_count() or 4, 8)
    print(f"\nRendering {len(evo_map_tasks)} evolution maps with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        [pool.submit(plot_evolution_map, **kw) for kw in evo_map_tasks]

    print("All evolution maps rendered.")
