"""Robustness evaluation: ten predefined initial borehole locations (Task 2).

Repeats the existing sequential decision simulation for 10 spread-out initial
borehole coordinates. Within each starting-location condition the same initial
observation is used for all policies and both model variants.

For each (model, starting location) combination the output mirrors a full
pomdp.py run: per-map step CSVs, summary.csv, per_map_metrics.csv,
aggregate_metrics.csv, cumulative_ore_by_step.csv, summary_by_step.csv, and
cheap line plots.

Execution is strictly sequential (one start at a time, one model at a time).
This is intentional: the GPU is the bottleneck, and multiple processes or
threads competing for the same CUDA device via separate contexts add overhead
without increasing throughput on a single GPU.

Policies evaluated: random, greedy (ore), uncertainty-guided.
Models evaluated:   CatVarEncoder (full), OreOnlyNullEncoder (map-only baseline).

Output root: decision_simulator/results/robustness/{timestamp}/

Usage
-----
    python -m decision_simulator.utils.evaluate_robustness \
        --cat-var-checkpoint   decision_simulator/pomdp/beliefs/cat_var_best_2000.pt \
        --null-checkpoint      decision_simulator/pomdp/beliefs/ore_only_null_best_2000.pt \
        --map-dir              C:/validation_datasets/One_orebody \
        --n-maps               1000 \
        --budget               11 \
        --device               cuda
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from decision_simulator.neural_belief.map_hdf5 import HDF5MapStore, MapPool
from decision_simulator.neural_belief.training.belief_models.end_to_end.train_cat_var_encoder import (
    load_cat_var_checkpoint,
)
from decision_simulator.neural_belief.training.belief_models.end_to_end.train_ore_only_null_encoder import (
    load_ore_only_null_checkpoint,
)
from decision_simulator.pomdp.helpers import HDF5DrillEnv, make_updater
from decision_simulator.pomdp.policies.greedy_yield_policy import GreedyYieldPolicy
from decision_simulator.pomdp.policies.random_policy import RandomPolicy
from decision_simulator.pomdp.policies.uncertainty_policy import UncertaintyPolicy
from decision_simulator.pomdp.pomdp import run_fixed_budget_episode
from decision_simulator.resources import load_decision_resources
from decision_simulator.utils.evaluate_policies import (
    _build_per_map_metrics,
    _orebody_dir_name,
    run_evaluation,
)


# ── Ten initial locations on a 32x32 reference grid (0-indexed) ──────────────

_INITIAL_LOCATIONS_REF = [
    ( 3,  3), ( 3, 16), ( 3, 28),
    (16,  3), (16, 16), (16, 28),
    (28,  3), (28, 16), (28, 28),
    (10, 22),
]

_STEP_FIELDS = [
    "policy", "step", "loc_i", "loc_j",
    "true_ore", "predicted_ore", "predicted_uncertainty",
    "total_predicted_ore", "total_true_ore", "top_ore",
]
_SUMMARY_FIELDS = ["map_idx", "policy", "total_true_ore", "best_observed_ore"]

_ALL_METRIC_COLS = [
    "top_ore_hit_rate", "first_top_ore_step", "n_top_ore_found",
    "avg_true_ore_drilled", "cumulative_true_ore",
    "mae", "pearson_r", "spearman_r",
    "unc_abs_error_corr", "unc_true_ore_corr",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_locations(
    ref_locs: list[tuple[int, int]], n_x: int, n_y: int
) -> list[tuple[int, int]]:
    return [
        (round(r / 31 * (n_x - 1)), round(c / 31 * (n_y - 1)))
        for r, c in ref_locs
    ]


def _start_dir_name(start_idx: int, i: int, j: int) -> str:
    return f"start_{start_idx:02d}_r{i:02d}_c{j:02d}"


def _load_maps(map_dir: Path, n_maps: int) -> tuple[MapPool, np.ndarray, list[Path]]:
    shard_files = (
        sorted(map_dir.glob("maps_0_and_1_orebodies_*.h5"))
        or sorted(map_dir.glob("maps_stratified_*.h5"))
        or sorted(map_dir.glob("maps_[0-9][0-9][0-9][0-9][0-9]_*.h5"))
    )
    if not shard_files:
        raise FileNotFoundError(f"No HDF5 shards found in {map_dir}.")

    pools: list[MapPool] = []
    n_bodies_chunks: list[np.ndarray] = []
    n_remaining = n_maps
    for sf in shard_files:
        if n_remaining <= 0:
            break
        with h5py.File(sf, "r") as hf:
            n_in   = int(hf.attrs["n_maps"])
            n_from = min(n_remaining, n_in)
            n_bodies_chunks.append(hf["n_bodies"][:n_from])
        pools.append(HDF5MapStore(sf).load_subset(list(range(n_from))))
        n_remaining -= n_from

    if len(pools) == 1:
        npz_map = pools[0]
    else:
        p0 = pools[0]
        npz_map = MapPool(
            borehole_arrays=[b for p in pools for b in p.borehole_arrays],
            targets=[t for p in pools for t in p.targets],
            drill_patterns=[d for p in pools for d in p.drill_patterns],
            cfg=dataclasses.replace(p0.cfg, n_maps=n_maps),
            n_x=p0.n_x,
            n_y=p0.n_y,
            rocks_arrays=[r for p in pools if p.rocks_arrays for r in p.rocks_arrays] or None,
        )

    return npz_map, np.concatenate(n_bodies_chunks), shard_files


def _aggregate_by(per_map_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in per_map_df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_cols, keys))
        row["n_maps"] = len(grp)
        for col in _ALL_METRIC_COLS:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)
    df.to_csv(path, index=False)


# ── Per-start simulation ──────────────────────────────────────────────────────

def _run_one_start(
    start_idx: int,
    init_loc: tuple[int, int],
    model_name: str,
    model_label: str,
    model,
    normalizer,
    npz_map: MapPool,
    n_bodies_arr: np.ndarray,
    resources,
    budget: int,
    device: str,
    run_root: Path,
    timestamp: str,
    dataset_path: str,
) -> pd.DataFrame:
    init_i, init_j = init_loc
    start_run_dir = run_root / model_label / _start_dir_name(start_idx, init_i, init_j)
    csv_dir = start_run_dir / "prediction_summaries"
    start_run_dir.mkdir(parents=True, exist_ok=True)

    all_step_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    _map_n_bodies: dict[int, int] = {}

    for map_idx in range(npz_map.pool_size):
        n_bodies_val = int(n_bodies_arr[map_idx])
        _map_n_bodies[map_idx] = n_bodies_val

        rocks_arr = npz_map.rocks_arrays[map_idx] if npz_map.rocks_arrays else None
        env = HDF5DrillEnv(
            bh_raw=npz_map.borehole_arrays[map_idx],
            target=npz_map.targets[map_idx],
            rocks=rocks_arr,
            n_x=npz_map.n_x,
            n_y=npz_map.n_y,
            norm_stats=resources.norm_stats,
            variable_names=resources.variable_names,
        )
        true_ore_map    = env.get_true_ore_map()
        total_true      = float(true_ore_map.sum())
        top10_threshold = float(np.percentile(true_ore_map, 90))

        updater = make_updater(model_name, model, normalizer, device, env)

        policies = [
            ("uncertainty", UncertaintyPolicy()),
            ("greedy",      GreedyYieldPolicy()),
            ("random",      RandomPolicy(seed=42 + map_idx * 100 + start_idx)),
        ]

        map_rows: list[dict] = []

        for policy_name, policy in policies:
            result = run_fixed_budget_episode(
                environment=env,
                belief_updater=updater,
                policy=policy,
                drilling_budget=budget,
                initial_locations=[init_loc],
            )
            for row in result["step_history"]:
                i_loc, j_loc = row["location"]
                map_rows.append({
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
                })
            all_summary_rows.append({
                "map_idx":           map_idx,
                "policy":            policy_name,
                "total_true_ore":    round(total_true, 3),
                "best_observed_ore": round(result["final_best_observed_ore"], 3),
            })

        orebody_maps_dir = csv_dir / _orebody_dir_name(n_bodies_val) / "maps"
        orebody_maps_dir.mkdir(parents=True, exist_ok=True)
        with open(orebody_maps_dir / f"map_{map_idx:03d}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_STEP_FIELDS)
            writer.writeheader()
            writer.writerows(map_rows)

        all_step_rows.extend(
            {**row, "map_idx": map_idx, "n_bodies": n_bodies_val}
            for row in map_rows
        )

        if (map_idx + 1) % 100 == 0 or map_idx == npz_map.pool_size - 1:
            print(f"  [{model_label}] start {start_idx:02d}  map {map_idx + 1}/{npz_map.pool_size}")

    _summary_by_orebody: dict[int, list[dict]] = defaultdict(list)
    for _row in all_summary_rows:
        _summary_by_orebody[_map_n_bodies[_row["map_idx"]]].append(_row)
    for _nb, _rows in sorted(_summary_by_orebody.items()):
        _rows.sort(key=lambda r: (r["policy"], r["map_idx"]))
        _od = csv_dir / _orebody_dir_name(_nb)
        _od.mkdir(parents=True, exist_ok=True)
        with open(_od / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(_rows)

    run_evaluation(all_step_rows, csv_dir)

    with open(start_run_dir / "metadata.json", "w") as fh:
        json.dump({
            "timestamp":    timestamp,
            "model_name":   model_name,
            "model_label":  model_label,
            "start_idx":    start_idx,
            "init_i":       init_i,
            "init_j":       init_j,
            "n_maps":       npz_map.pool_size,
            "budget":       budget,
            "grid_size":    [npz_map.n_x, npz_map.n_y],
            "dataset_path": dataset_path,
            "created_at":   datetime.datetime.now().isoformat(),
        }, fh, indent=2)

    print(f"  [{model_label}] start {start_idx:02d} done -> {start_run_dir}")

    step_df = pd.DataFrame(all_step_rows)
    per_map_df = _build_per_map_metrics(step_df)
    per_map_df["model"]     = model_name
    per_map_df["start_idx"] = start_idx
    per_map_df["init_i"]    = init_i
    per_map_df["init_j"]    = init_j
    return per_map_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    BELIEFS_DIR = Path(__file__).parent.parent / "pomdp" / "beliefs"
    RESULTS_DIR = Path(__file__).parent.parent / "results"

    p = argparse.ArgumentParser(
        description="Robustness evaluation across 10 initial borehole locations."
    )
    p.add_argument(
        "--cat-var-checkpoint",
        type=Path,
        default=BELIEFS_DIR / "cat_var_best_2000.pt",
        help="CatVarEncoder checkpoint. Single: cat_var_best_2000.pt; multi: cat_var_best_multi_2000.pt.",
    )
    p.add_argument(
        "--null-checkpoint",
        type=Path,
        default=BELIEFS_DIR / "ore_only_null_best_2000.pt",
        help="OreOnlyNullEncoder checkpoint. Single: ore_only_null_best_2000.pt; multi: ore_only_best_multi_2000.pt.",
    )
    p.add_argument("--map-dir",  type=Path, default=Path("C:/validation_datasets/One_orebody"))
    p.add_argument("--n-maps",   type=int,  default=1000)
    p.add_argument("--budget",   type=int,  default=11)
    p.add_argument("--device",   default="cuda")
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root  = RESULTS_DIR / "robustness" / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_maps} maps from {args.map_dir}...")
    npz_map, n_bodies_arr, shard_files = _load_maps(args.map_dir, args.n_maps)
    initial_locs = _scale_locations(_INITIAL_LOCATIONS_REF, npz_map.n_x, npz_map.n_y)
    dataset_path = str(shard_files[0])
    print(f"  Grid: {npz_map.n_x} x {npz_map.n_y}  |  {npz_map.pool_size} maps")
    print(f"  Initial locations: {initial_locs}\n")

    print("Loading resources...")
    resources, _ = load_decision_resources(borehole_encoder="jepa", device=args.device)

    model_variants = [
        ("cat_var",       "cat_var",   args.cat_var_checkpoint, "cat_var"),
        ("ore_only_null", "only_ore",  args.null_checkpoint,    "null"),
    ]

    cross_start_dfs: list[pd.DataFrame] = []

    for model_name, model_label, ckpt_path, ckpt_tag in model_variants:
        print(f"{'=' * 70}")
        print(f"Model: {model_name}  ({len(initial_locs)} starts x {npz_map.pool_size} maps)")
        print(f"{'=' * 70}")

        print(f"Loading {ckpt_tag} checkpoint: {ckpt_path}")
        if model_name == "cat_var":
            model, _, normalizer, _ = load_cat_var_checkpoint(ckpt_path, device=args.device)
        else:
            model, _, normalizer, _ = load_ore_only_null_checkpoint(ckpt_path, device=args.device)
        model.eval()

        for start_idx, init_loc in enumerate(initial_locs):
            per_map_df = _run_one_start(
                start_idx, init_loc,
                model_name, model_label, model, normalizer,
                npz_map, n_bodies_arr, resources,
                args.budget, args.device,
                run_root, timestamp, dataset_path,
            )
            cross_start_dfs.append(per_map_df)

        print(f"  Model {model_name} complete.\n")

    # ── Cross-start aggregates ────────────────────────────────────────────────

    cross_df = pd.concat(cross_start_dfs, ignore_index=True)

    agg_by_start = _aggregate_by(
        cross_df, ["start_idx", "init_i", "init_j", "model", "policy"]
    )
    _save_df(agg_by_start, run_root / "aggregate_by_start.csv")
    print(f"Aggregate by start -> {run_root / 'aggregate_by_start.csv'}  ({len(agg_by_start)} rows)")

    agg_overall = _aggregate_by(cross_df, ["model", "policy"])
    agg_overall.insert(2, "n_starts", len(initial_locs))
    _save_df(agg_overall, run_root / "aggregate_overall.csv")
    print(f"Aggregate overall  -> {run_root / 'aggregate_overall.csv'}  ({len(agg_overall)} rows)")

    with open(run_root / "metadata.json", "w") as fh:
        json.dump({
            "timestamp":    timestamp,
            "n_maps":       args.n_maps,
            "n_starts":     len(initial_locs),
            "initial_locs": initial_locs,
            "budget":       args.budget,
            "grid_size":    [npz_map.n_x, npz_map.n_y],
            "dataset_path": dataset_path,
            "cat_var_ckpt": str(args.cat_var_checkpoint),
            "null_ckpt":    str(args.null_checkpoint),
            "created_at":   datetime.datetime.now().isoformat(),
        }, fh, indent=2)
    print(f"Top-level metadata -> {run_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
