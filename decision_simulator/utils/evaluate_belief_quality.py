"""Controlled belief-map quality evaluation (Task 1).

Policy-independent evaluation: the same fixed random drill sequence is fed to
both the full CatVarEncoder and the map-only OreOnlyNullEncoder, and
reconstruction quality is measured on *unobserved* cells after every added
borehole.

Metrics per (model, map, n_observed_boreholes):
  - MAE between predicted and true ore on unobserved cells
  - Pearson r between predicted and true ore on unobserved cells
  - Spearman r between predicted and true ore on unobserved cells
  - Spearman r between predicted uncertainty and |predicted - true| on
    unobserved cells (NaN when the model has no uncertainty output)

Per-map metrics are computed independently, then aggregated (mean / std / n_maps)
across maps.

Usage
-----
    python -m decision_simulator.utils.evaluate_belief_quality \
        --cat-var-checkpoint   decision_simulator/pomdp/beliefs/cat_var_best_2000.pt \
        --null-checkpoint      decision_simulator/pomdp/beliefs/ore_only_null_best_2000.pt \
        --map-dir              C:/validation_datasets/One_orebody \
        --n-maps               100 \
        --max-boreholes        10 \
        --device               cuda
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from decision_simulator.neural_belief.map_hdf5 import HDF5MapStore, MapPool
from decision_simulator.neural_belief.training.belief_models.end_to_end.train_cat_var_encoder import (
    load_cat_var_checkpoint,
)
from decision_simulator.neural_belief.training.belief_models.end_to_end.train_ore_only_null_encoder import (
    load_ore_only_null_checkpoint,
)
from decision_simulator.pomdp.helpers import HDF5DrillEnv, CatVarUpdater, OreOnlyNullUpdater
from decision_simulator.pomdp.observations.borehole_observations import (
    BoreholeObservationState,
)
from decision_simulator.resources import load_decision_resources
from decision_simulator.utils.evaluate_policies import _safe_pearson, _safe_spearman


# ── Metric helpers ────────────────────────────────────────────────────────────

def _compute_unobserved_metrics(
    belief_state,
    true_ore_map: np.ndarray,
) -> dict:
    """Compute reconstruction metrics restricted to unobserved cells."""
    unobs = ~belief_state.observed_mask  # (n_x, n_y) bool
    true_unobs = true_ore_map[unobs]
    pred_unobs = belief_state.predicted_ore_map[unobs]
    abs_err    = np.abs(pred_unobs - true_unobs)

    mae        = float(np.mean(abs_err))
    pearson_r  = _safe_pearson(pred_unobs, true_unobs)
    spearman_r = _safe_spearman(pred_unobs, true_unobs)

    if belief_state.predicted_uncertainty_map is not None:
        unc_unobs    = belief_state.predicted_uncertainty_map[unobs]
        unc_spearman = _safe_spearman(unc_unobs, abs_err)
    else:
        unc_spearman = float("nan")

    return {
        "n_unobserved_cells": int(unobs.sum()),
        "mae":                 mae,
        "pearson_r":           pearson_r,
        "spearman_r":          spearman_r,
        "unc_abs_error_spearman": unc_spearman,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

_METRIC_COLS = ["mae", "pearson_r", "spearman_r", "unc_abs_error_spearman"]


def _aggregate(per_map_rows: list[dict]) -> list[dict]:
    """Aggregate per-map rows into mean / std / n_maps per (model, n_bodies, k)."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_map_rows:
        groups[(row["model"], row["n_bodies"], row["n_observed_boreholes"])].append(row)

    agg_rows = []
    for (model, n_bodies, k), grp in sorted(groups.items()):
        agg: dict = {"model": model, "n_bodies": n_bodies, "n_observed_boreholes": k, "n_maps": len(grp)}
        for col in _METRIC_COLS:
            vals = np.array([r[col] for r in grp], dtype=float)
            valid = vals[~np.isnan(vals)]
            agg[f"{col}_mean"] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
            agg[f"{col}_std"]  = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
            agg[f"n_maps_{col}"] = int(len(valid))
        agg_rows.append(agg)
    return agg_rows


# ── CSV writers ───────────────────────────────────────────────────────────────

_PER_MAP_FIELDS = [
    "model", "map_idx", "n_bodies", "n_observed_boreholes", "n_unobserved_cells",
    "mae", "pearson_r", "spearman_r", "unc_abs_error_spearman",
]

_AGG_FIELDS = [
    "model", "n_bodies", "n_observed_boreholes", "n_maps",
    "mae_mean", "mae_std", "n_maps_mae",
    "pearson_r_mean", "pearson_r_std", "n_maps_pearson_r",
    "spearman_r_mean", "spearman_r_std", "n_maps_spearman_r",
    "unc_abs_error_spearman_mean", "unc_abs_error_spearman_std", "n_maps_unc_abs_error_spearman",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rounded = {}
            for k, v in row.items():
                if isinstance(v, float):
                    rounded[k] = round(v, 4)
                else:
                    rounded[k] = v
            writer.writerow(rounded)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    BELIEFS_DIR = Path(__file__).parent.parent / "pomdp" / "beliefs"
    RESULTS_DIR = Path(__file__).parent.parent / "results"

    p = argparse.ArgumentParser(
        description="Controlled belief-map quality evaluation (policy-independent)."
    )
    p.add_argument(
        "--cat-var-checkpoint",
        type=Path,
        default=BELIEFS_DIR / "cat_var_best_2000.pt",
        help="CatVarEncoder checkpoint. Single-orebody: cat_var_best_2000.pt; multi: cat_var_best_multi_2000.pt.",
    )
    p.add_argument(
        "--null-checkpoint",
        type=Path,
        default=BELIEFS_DIR / "ore_only_null_best_2000.pt",
        help="OreOnlyNullEncoder checkpoint. Single-orebody: ore_only_null_best_2000.pt; multi: ore_only_best_multi_2000.pt.",
    )
    p.add_argument(
        "--map-dir",
        type=Path,
        default=Path("C:/validation_datasets/One_orebody"),
        help="Directory containing pre-generated HDF5 map shards.",
    )
    p.add_argument(
        "--n-maps",
        type=int,
        default=1000,
        help="Number of maps to evaluate (default: 1000).",
    )
    p.add_argument(
        "--max-boreholes",
        type=int,
        default=10,
        help="Maximum number of boreholes to observe per map (default: 10).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device string (default: cuda).",
    )
    args = p.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / "belief_quality" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load resources ────────────────────────────────────────────────────────

    print("Loading resources...")
    resources, _ = load_decision_resources(borehole_encoder="jepa", device=args.device)

    print(f"Loading cat_var checkpoint: {args.cat_var_checkpoint}")
    cat_var_model, _, cat_var_norm, _ = load_cat_var_checkpoint(
        args.cat_var_checkpoint, device=args.device
    )
    cat_var_model.eval()

    print(f"Loading null checkpoint: {args.null_checkpoint}")
    null_model, _, null_norm, _ = load_ore_only_null_checkpoint(
        args.null_checkpoint, device=args.device
    )
    null_model.eval()

    # ── Load maps ─────────────────────────────────────────────────────────────

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
    _n_bodies_arr = np.concatenate(_n_bodies_chunks)

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

    print(
        f"  Loaded {npz_map.pool_size} maps  "
        f"(n_x={npz_map.n_x}, n_y={npz_map.n_y})"
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────

    per_map_rows: list[dict] = []

    for map_idx in range(npz_map.pool_size):
        n_bodies_val = int(_n_bodies_arr[map_idx])
        print(f"\nMap {map_idx:03d} / {npz_map.pool_size - 1}  (n_bodies={n_bodies_val})")

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
        true_ore_map = env.get_true_ore_map()

        cat_var_updater = CatVarUpdater(cat_var_model, cat_var_norm, args.device, env)
        null_updater    = OreOnlyNullUpdater(null_model, null_norm, args.device, env)

        # Fixed random drill sequence for this map
        rng = np.random.default_rng(42 + map_idx)
        drill_order = rng.permutation(npz_map.n_x * npz_map.n_y)  # flat indices

        obs_state = BoreholeObservationState(npz_map.n_x, npz_map.n_y)

        for k in range(1, args.max_boreholes + 1):
            flat = int(drill_order[k - 1])
            i, j = divmod(flat, npz_map.n_y)
            borehole, ore_value = env.drill(i, j)
            obs_state.add_observation(i, j, borehole, ore_value)

            for model_name, updater in [
                ("cat_var",       cat_var_updater),
                ("ore_only_null", null_updater),
            ]:
                belief = updater.update(obs_state, step=k)
                metrics = _compute_unobserved_metrics(belief, true_ore_map)
                per_map_rows.append({
                    "model":                model_name,
                    "map_idx":              map_idx,
                    "n_bodies":             n_bodies_val,
                    "n_observed_boreholes": k,
                    **metrics,
                })

            print(
                f"  k={k:2d}  cat_var mae={per_map_rows[-2]['mae']:.4f}"
                f"  null mae={per_map_rows[-1]['mae']:.4f}"
            )

    # ── Write outputs ─────────────────────────────────────────────────────────

    per_map_path = out_dir / "per_map_results.csv"
    _write_csv(per_map_path, _PER_MAP_FIELDS, per_map_rows)
    print(f"\nPer-map results -> {per_map_path}")

    agg_rows = _aggregate(per_map_rows)
    agg_path = out_dir / "aggregate_results.csv"
    _write_csv(agg_path, _AGG_FIELDS, agg_rows)
    print(f"Aggregate results -> {agg_path}")


if __name__ == "__main__":
    main()
