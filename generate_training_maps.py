"""Generate belief map pool for neural-belief training.

Maps are generated in parallel and stored as per-map npz files.
Re-running with a larger ``--n-maps`` appends only the missing files (resume-safe).
Convert the resulting pool to HDF5 shards with ``colab_npz_to_hdf5_full.py``.

Output layout
-------------
    <out-dir>/
        config.pkl              pool metadata
        labels_vocab.pkl        {rocks: {name: idx}, formations: {name: idx}}
        map_00000.npz           per-map arrays
        map_00001.npz
        ...

Per-map npz keys
----------------
    boreholes     : (n_x*n_y, V, D) float32 — raw (unstandardised) variables
    yield_target  : (n_x, n_y)      float32 — max-pooled yield field
    rocks         : (n_x*n_y, D)    int8    — rock-type vocab indices
    formations    : (n_x*n_y, D)    int8    — formation vocab indices
    drill_locs    : (S, max_drills, 2) int16
    drill_ore_vals: (S, max_drills)    float32
    drill_counts  : (S,)               int16

Usage
-----
    python generate_training_maps.py
    python generate_training_maps.py --n-maps 500 --workers 8 --out-dir data/belief_maps
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import MapGenerator, SimConfig

# ---------------------------------------------------------------------------
# Worker globals (populated in each worker process by _worker_init)
# ---------------------------------------------------------------------------

_WORKER_GEN = None
_WORKER_VARS = None
_WORKER_OUT = None
_WORKER_ROCK_VOCAB: dict[str, int] | None = None
_WORKER_FM_VOCAB: dict[str, int] | None = None
_WORKER_N_X: int = 0
_WORKER_N_Y: int = 0
_WORKER_SAMPLES_PER_MAP: int = 0
_WORKER_MIN_DRILLS: int = 0
_WORKER_MAX_DRILLS: int = 0
_WORKER_SEED: int = 0


def _worker_init(
    bank_path,
    geom_path,
    prior_path,
    out_dir,
    variables,
    worker_seed_base,
    rock_vocab,
    fm_vocab,
    n_x,
    n_y,
    samples_per_map,
    min_drills,
    max_drills,
    main_seed,
    n_ore_bodies,
):
    global _WORKER_GEN, _WORKER_VARS, _WORKER_OUT
    global _WORKER_ROCK_VOCAB, _WORKER_FM_VOCAB
    global _WORKER_N_X, _WORKER_N_Y
    global _WORKER_SAMPLES_PER_MAP, _WORKER_MIN_DRILLS, _WORKER_MAX_DRILLS
    global _WORKER_SEED

    bank = DistributionBank.load(bank_path)
    geom = FormationGeometry.load(geom_path)
    prior = DiscoveryPrior.load(prior_path)
    seed = worker_seed_base + os.getpid()
    _WORKER_GEN = MapGenerator(bank, geom, SimConfig(n_ore_bodies=n_ore_bodies), seed=seed, prior=prior)
    _WORKER_VARS = list(variables)
    _WORKER_OUT = Path(out_dir)
    _WORKER_ROCK_VOCAB = rock_vocab
    _WORKER_FM_VOCAB = fm_vocab
    _WORKER_N_X = n_x
    _WORKER_N_Y = n_y
    _WORKER_SAMPLES_PER_MAP = samples_per_map
    _WORKER_MIN_DRILLS = min_drills
    _WORKER_MAX_DRILLS = max_drills
    _WORKER_SEED = main_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vocabs(
    bank: DistributionBank,
    geom: FormationGeometry,
) -> tuple[dict[str, int], dict[str, int]]:
    rocks_seen: set[str] = {"other"}
    rocks_seen.update(getattr(bank, "rock_types", []) or [])
    rocks_seen.update(k[0] for k in getattr(bank, "cells", {}).keys())
    rocks = ["other"] + sorted(r for r in rocks_seen if r != "other")
    rock_vocab = {r: i for i, r in enumerate(rocks)}

    fms_seen: set[str] = set(getattr(geom, "formations", {}) or {})
    fms_seen.add("other")
    formations = ["other"] + sorted(f for f in fms_seen if f != "other")
    fm_vocab = {f: i for i, f in enumerate(formations)}
    return rock_vocab, fm_vocab


def _encode_labels(arr: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    """Map a (nx, ny, nz) object array of strings to (nx*ny, nz) int8."""
    import pandas as pd

    nx, ny, nz = arr.shape
    flat = arr.reshape(nx * ny, nz).astype(object)
    cat = pd.Categorical(flat.ravel(), categories=list(vocab.keys()))
    codes = cat.codes.copy()
    codes[codes < 0] = vocab.get("other", 0)
    return codes.reshape(nx * ny, nz).astype(np.int8)


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------


def _generate_one(map_idx: int) -> tuple[int, float]:
    """Generate and save one map.  Skipped if the output file already exists."""
    out_path = _WORKER_OUT / f"map_{map_idx:05d}.npz"
    if out_path.exists():
        return map_idx, 0.0

    t0 = time.perf_counter()
    m = next(_WORKER_GEN)
    n_x, n_y = _WORKER_N_X, _WORKER_N_Y
    n_depth = len(m["depth_axis"])

    # Raw borehole variables — float32, NOT standardised
    bh = np.empty((n_x * n_y, len(_WORKER_VARS), n_depth), dtype=np.float32)
    for vi, v in enumerate(_WORKER_VARS):
        bh[:, vi, :] = m["variables"][v].reshape(n_x * n_y, n_depth)
    bh = np.nan_to_num(bh, nan=0.0)

    # Yield target: max over depth → (n_x, n_y)
    yield_target = m["yield_field"].max(axis=2).astype(np.float32)

    # Geology labels
    rocks = _encode_labels(np.asarray(m["rock_types"]), _WORKER_ROCK_VOCAB)
    formations = _encode_labels(np.asarray(m["formations"]), _WORKER_FM_VOCAB)

    # Drill patterns — seeded deterministically per map so they are reproducible
    all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]
    rng = np.random.default_rng(_WORKER_SEED * 100003 + map_idx)

    drill_locs_arr = np.zeros(
        (_WORKER_SAMPLES_PER_MAP, _WORKER_MAX_DRILLS, 2), dtype=np.int16
    )
    drill_ore_arr = np.zeros(
        (_WORKER_SAMPLES_PER_MAP, _WORKER_MAX_DRILLS), dtype=np.float32
    )
    drill_counts_arr = np.zeros(_WORKER_SAMPLES_PER_MAP, dtype=np.int16)

    for s in range(_WORKER_SAMPLES_PER_MAP):
        n_drills = int(rng.integers(_WORKER_MIN_DRILLS, _WORKER_MAX_DRILLS + 1))
        chosen = rng.choice(len(all_locations), size=n_drills, replace=False)
        drill_counts_arr[s] = n_drills
        for k, loc_idx in enumerate(chosen):
            r, c = all_locations[loc_idx]
            drill_locs_arr[s, k] = [r, c]
            drill_ore_arr[s, k] = float(yield_target[r, c])

    np.savez_compressed(
        out_path,
        boreholes=bh,
        yield_target=yield_target,
        rocks=rocks,
        formations=formations,
        drill_locs=drill_locs_arr,
        drill_ore_vals=drill_ore_arr,
        drill_counts=drill_counts_arr,
        n_bodies=np.int8(len(m["bodies"])),
    )

    return map_idx, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate belief map pool for neural-belief training."
    )
    p.add_argument("--out-dir", type=Path, default=Path("C://dataset_complete"))
    p.add_argument("--n-maps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument(
        "--samples-per-map", type=int, default=20, help="drill patterns stored per map"
    )
    p.add_argument("--min-drills", type=int, default=1)
    p.add_argument("--max-drills", type=int, default=15)
    p.add_argument("--bank", type=Path, default=Path("data/clean/distributions.pkl"))
    p.add_argument(
        "--geom", type=Path, default=Path("data/clean/formation_geometry.pkl")
    )
    p.add_argument("--prior", type=Path, default=Path("data/clean/discovery_prior.pkl"))
    p.add_argument(
        "--n-bodies", type=int, default=None,
        help="fix ore body count per map (0-3); omit for random 0-3",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimConfig(n_ore_bodies=args.n_bodies)
    n_x, n_y = sim_cfg.n_x, sim_cfg.n_y
    variables = list(sim_cfg.variables)

    vocab_path = args.out_dir / "labels_vocab.pkl"
    config_path = args.out_dir / "config.pkl"

    # Build or load label vocabs
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        rock_vocab = vocabs["rocks"]
        fm_vocab = vocabs["formations"]
        print(f"  loaded label vocab from {vocab_path}")
    else:
        bank = DistributionBank.load(args.bank)
        geom = FormationGeometry.load(args.geom)
        rock_vocab, fm_vocab = _build_vocabs(bank, geom)
        with open(vocab_path, "wb") as f:
            pickle.dump({"rocks": rock_vocab, "formations": fm_vocab}, f)
        print(
            f"  label vocab: {len(rock_vocab)} rocks, {len(fm_vocab)} formations"
            f" -> {vocab_path}"
        )
        del bank, geom

    # Write config (read by colab_npz_to_hdf5_full.py during HDF5 conversion)
    with open(config_path, "wb") as f:
        pickle.dump(
            {
                "cache_version": "2",
                "variables": variables,
                "n_maps": args.n_maps,
                "seed": args.seed,
                "samples_per_map": args.samples_per_map,
                "min_drills": args.min_drills,
                "max_drills": args.max_drills,
                "n_x": n_x,
                "n_y": n_y,
            },
            f,
        )

    # Determine which map indices still need to be generated
    pending = [
        i
        for i in range(args.n_maps)
        if not (args.out_dir / f"map_{i:05d}.npz").exists()
    ]
    if not pending:
        print(f"  all {args.n_maps} maps already exist in {args.out_dir}")
        return
    skipped = args.n_maps - len(pending)
    print(
        f"  generating {len(pending)} maps with {args.workers} workers"
        + (f" (skipping {skipped} already saved)" if skipped else "")
    )

    # Parallel generation
    t0 = time.perf_counter()
    completed = 0
    last_report = t0
    init_args = (
        str(args.bank),
        str(args.geom),
        str(args.prior),
        str(args.out_dir),
        variables,
        args.seed * 7919,  # worker_seed_base
        rock_vocab,
        fm_vocab,
        n_x,
        n_y,
        args.samples_per_map,
        args.min_drills,
        args.max_drills,
        args.seed,  # main_seed for drill-pattern RNG
        args.n_bodies,
    )
    with mp.Pool(args.workers, initializer=_worker_init, initargs=init_args) as pool:
        for idx, dt in pool.imap_unordered(_generate_one, pending, chunksize=4):
            completed += 1
            now = time.perf_counter()
            if now - last_report >= 10 or completed == len(pending):
                elapsed = now - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(pending) - completed
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"    {completed:>6d}/{len(pending)}"
                    f"  ({rate:.1f} maps/s, ETA {eta / 60:.1f} min)"
                )
                last_report = now

    print(f"\n  done in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"  pool -> {args.out_dir}  ({args.n_maps} maps)")

    # Write a flat index so colab_npz_to_hdf5_full.py can stratify by body count
    # without opening every map file.
    index_path = args.out_dir / "n_bodies_index.npy"
    index = np.array(
        [int(np.load(args.out_dir / f"map_{i:05d}.npz")["n_bodies"])
         for i in range(args.n_maps)],
        dtype=np.int8,
    )
    np.save(index_path, index)
    unique, counts = np.unique(index, return_counts=True)
    print("  ore-body distribution:")
    for u, c in zip(unique, counts):
        print(f"    {u} bodies: {c} maps ({c/args.n_maps:.1%})")
    print(f"  index -> {index_path}")


if __name__ == "__main__":
    main()
