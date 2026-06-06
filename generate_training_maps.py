"""Generate formation-based belief map pool for neural-belief training.

Uses FormationMapGenerator (variables drawn from a formation-indexed bank)
instead of the rock-indexed MapGenerator.  Maps are generated in parallel and
written directly to HDF5 shards, matching the exact structure expected by
HDF5MapStore / HDF5MapDirectory (decision_simulator/neural_belief/map_hdf5.py).

Re-running with the same --out-dir resumes from the last completed shard
(completed map indices are recovered from the map_index dataset in each
existing shard).

Output layout
-------------
    <out-dir>/
        config.pkl
        labels_vocab.pkl
        maps_00000_00499.h5     one shard = --shard-size maps (default 500)
        maps_00500_00999.h5
        ...

HDF5 shard structure (per shard, N maps)
-----------------------------------------
    Attributes:
        n_maps, pool_n_x, pool_n_y, pool_samples_per_map, pool_min_drills,
        pool_max_drills, pool_seed, created_utc, file_type, class_counts,
        variables (utf-8), rock_vocab (utf-8), formation_vocab (utf-8)
    Datasets:
        boreholes      (N, n_x*n_y, V, D) float32
        yield_target   (N, n_x, n_y)       float32
        rocks          (N, n_x*n_y, D)     int8
        formations     (N, n_x*n_y, D)     int8
        drill_locs     (N, S, max_drills, 2) int16
        drill_ore_vals (N, S, max_drills)  float32
        drill_counts   (N, S)              int16
        n_bodies       (N,)                int8
        map_index      (N,)                int32
        filenames      (N,)                str

Usage
-----
    python generate_training_maps.py
    python generate_training_maps.py --n-maps 5000 --workers 8 --out-dir data/formation_maps
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulator.formation_distributions import FormationDistributionBank
from simulator.distributions import DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import FormationMapGenerator, SimConfig

# ---------------------------------------------------------------------------
# Worker globals (populated in each worker process by _worker_init)
# ---------------------------------------------------------------------------

_WORKER_GEN = None
_WORKER_VARS = None
_WORKER_ROCK_VOCAB: dict[str, int] | None = None
_WORKER_FM_VOCAB: dict[str, int] | None = None
_WORKER_N_X: int = 0
_WORKER_N_Y: int = 0
_WORKER_SAMPLES_PER_MAP: int = 0
_WORKER_MIN_DRILLS: int = 0
_WORKER_MAX_DRILLS: int = 0
_WORKER_SEED: int = 0
_WORKER_MIN_N_BODIES: int = 0


def _worker_init(
    bank_path,
    geom_path,
    prior_path,
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
    min_n_bodies,
):
    global _WORKER_GEN, _WORKER_VARS
    global _WORKER_ROCK_VOCAB, _WORKER_FM_VOCAB
    global _WORKER_N_X, _WORKER_N_Y
    global _WORKER_SAMPLES_PER_MAP, _WORKER_MIN_DRILLS, _WORKER_MAX_DRILLS
    global _WORKER_SEED, _WORKER_MIN_N_BODIES

    bank = FormationDistributionBank.load(bank_path)
    geom = FormationGeometry.load(geom_path)
    prior = DiscoveryPrior.load(prior_path)
    seed = worker_seed_base + os.getpid()
    cfg = SimConfig() if n_ore_bodies is None else SimConfig(max_ore_bodies=n_ore_bodies)
    _WORKER_GEN = FormationMapGenerator(bank, geom, cfg, seed=seed, prior=prior)
    _WORKER_VARS = list(variables)
    _WORKER_ROCK_VOCAB = rock_vocab
    _WORKER_FM_VOCAB = fm_vocab
    _WORKER_N_X = n_x
    _WORKER_N_Y = n_y
    _WORKER_SAMPLES_PER_MAP = samples_per_map
    _WORKER_MIN_DRILLS = min_drills
    _WORKER_MAX_DRILLS = max_drills
    _WORKER_SEED = main_seed
    _WORKER_MIN_N_BODIES = min_n_bodies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vocabs(
    bank: FormationDistributionBank,
    geom: FormationGeometry,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build {rocks, formations} string→index maps.

    Rocks are sourced from FormationGeometry (the rock Markov-chain pool is
    what actually appears in generated columns; FormationDistributionBank does
    not index by rock type).  Formations are sourced from both the bank and
    the geometry to cover the full set.
    """
    rocks_seen: set[str] = {"other"}
    for stats in geom.formations.values():
        rocks_seen.update(getattr(stats, "facies", {}).keys() or [])
    rocks = ["other"] + sorted(r for r in rocks_seen if r != "other")
    rock_vocab = {r: i for i, r in enumerate(rocks)}

    fms_seen: set[str] = {"other"}
    fms_seen.update(getattr(bank, "formations", []) or [])
    fms_seen.update(getattr(geom, "formations", {}) or {})
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


def _generate_one(map_idx: int) -> tuple[int, dict, float]:
    """Generate one map and return its arrays (no disk I/O in the worker)."""
    t0 = time.perf_counter()
    while True:
        m = next(_WORKER_GEN)
        if len(m["bodies"]) >= _WORKER_MIN_N_BODIES:
            break
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

    # Drill patterns — seeded deterministically per map
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

    return map_idx, {
        "boreholes": bh,
        "yield_target": yield_target,
        "rocks": rocks,
        "formations": formations,
        "drill_locs": drill_locs_arr,
        "drill_ore_vals": drill_ore_arr,
        "drill_counts": drill_counts_arr,
        "n_bodies": np.int8(len(m["bodies"])),
    }, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# HDF5 shard writer
# ---------------------------------------------------------------------------

_COMPRESS_KWARGS: dict = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def _write_hdf5_shard(
    out_path: Path,
    batch: list[tuple[int, dict]],
    pool_config: dict,
    rock_vocab: dict[str, int],
    fm_vocab: dict[str, int],
) -> None:
    """Write a sorted batch of (map_idx, arrays) to a single HDF5 shard.

    The shard structure exactly matches the output of colab_npz_to_hdf5_full.py
    and is readable by HDF5MapStore / HDF5MapDirectory.
    """
    batch.sort(key=lambda x: x[0])
    map_indices = [x[0] for x in batch]
    maps = [x[1] for x in batch]
    N = len(batch)

    n_bodies_arr = np.array([int(m["n_bodies"]) for m in maps], dtype=np.int8)
    class_counts = np.array(
        [(n_bodies_arr == c).sum() for c in range(4)], dtype=np.int32
    )

    def _chunk1(shape: tuple) -> tuple:
        return (1, *shape[1:])

    with h5py.File(out_path, "w") as hf:
        # ── Attributes ────────────────────────────────────────────────────
        hf.attrs["n_maps"]               = N
        hf.attrs["pool_n_x"]             = pool_config["n_x"]
        hf.attrs["pool_n_y"]             = pool_config["n_y"]
        hf.attrs["pool_samples_per_map"] = pool_config["samples_per_map"]
        hf.attrs["pool_min_drills"]      = pool_config["min_drills"]
        hf.attrs["pool_max_drills"]      = pool_config["max_drills"]
        hf.attrs["pool_seed"]            = pool_config["seed"]
        hf.attrs["created_utc"]          = datetime.now(timezone.utc).isoformat()
        hf.attrs["file_type"]            = "formation"
        hf.attrs["class_counts"]         = class_counts
        hf.attrs["variables"] = np.array(
            [s.encode("utf-8") for s in pool_config["variables"]]
        )
        hf.attrs["rock_vocab"] = np.array(
            [s.encode("utf-8") for s in rock_vocab]
        )
        hf.attrs["formation_vocab"] = np.array(
            [s.encode("utf-8") for s in fm_vocab]
        )

        # ── Index datasets ─────────────────────────────────────────────────
        hf.create_dataset(
            "n_bodies",  data=n_bodies_arr,                          **_COMPRESS_KWARGS
        )
        hf.create_dataset(
            "map_index", data=np.array(map_indices, dtype=np.int32), **_COMPRESS_KWARGS
        )
        fn_ds = hf.create_dataset(
            "filenames", (N,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        for i, idx in enumerate(map_indices):
            fn_ds[i] = f"map_{idx:05d}"

        # ── Data datasets (chunked, one map per chunk) ─────────────────────
        bh_shape  = (N, *maps[0]["boreholes"].shape)
        yt_shape  = (N, *maps[0]["yield_target"].shape)
        ro_shape  = (N, *maps[0]["rocks"].shape)
        fm_shape  = (N, *maps[0]["formations"].shape)
        dl_shape  = (N, *maps[0]["drill_locs"].shape)
        dov_shape = (N, *maps[0]["drill_ore_vals"].shape)
        dc_shape  = (N, *maps[0]["drill_counts"].shape)

        def _ds(name: str, shape: tuple, dtype) -> h5py.Dataset:
            return hf.create_dataset(
                name, shape=shape, dtype=dtype,
                chunks=_chunk1(shape), **_COMPRESS_KWARGS
            )

        bh_ds  = _ds("boreholes",      bh_shape,  np.float32)
        yt_ds  = _ds("yield_target",   yt_shape,  np.float32)
        ro_ds  = _ds("rocks",          ro_shape,  np.int8)
        fm_ds  = _ds("formations",     fm_shape,  np.int8)
        dl_ds  = _ds("drill_locs",     dl_shape,  np.int16)
        dov_ds = _ds("drill_ore_vals", dov_shape, np.float32)
        dc_ds  = _ds("drill_counts",   dc_shape,  np.int16)

        for i, m in enumerate(maps):
            bh_ds[i]  = m["boreholes"]
            yt_ds[i]  = m["yield_target"]
            ro_ds[i]  = m["rocks"]
            fm_ds[i]  = m["formations"]
            dl_ds[i]  = m["drill_locs"]
            dov_ds[i] = m["drill_ore_vals"]
            dc_ds[i]  = m["drill_counts"]

    size_mb = out_path.stat().st_size / 1024 ** 2
    print(
        f"    wrote {out_path.name}"
        f"  ({N} maps, dist={class_counts.tolist()}, {size_mb:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------


def _find_completed_maps(out_dir: Path) -> set[int]:
    """Return all map indices already stored in existing HDF5 shards."""
    done: set[int] = set()
    for h5 in sorted(out_dir.glob("maps_*.h5")):
        try:
            with h5py.File(h5, "r") as hf:
                if "map_index" in hf:
                    done.update(int(i) for i in hf["map_index"][:])
        except Exception:
            pass
    return done


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def _generate_phase(
    pending: list[int],
    phase_start: int,
    shard_prefix: str,
    args,
    pool_config: dict,
    rock_vocab: dict[str, int],
    fm_vocab: dict[str, int],
    max_ore_bodies: int | None,
    min_ore_bodies: int,
    n_x: int,
    n_y: int,
    variables: list,
) -> None:
    """Run one generation phase and write shards with the given prefix."""
    init_args = (
        str(args.bank),
        str(args.geom),
        str(args.prior),
        variables,
        args.seed * 7919,
        rock_vocab,
        fm_vocab,
        n_x,
        n_y,
        args.samples_per_map,
        args.min_drills,
        args.max_drills,
        args.seed,
        max_ore_bodies,
        min_ore_bodies,
    )

    existing_shards = sorted(args.out_dir.glob(f"{shard_prefix}_*.h5"))
    shard_seq = len(existing_shards)

    t0 = time.perf_counter()
    completed_count = 0
    last_report = t0
    current_batch: list[tuple[int, dict]] = []

    with mp.Pool(args.workers, initializer=_worker_init, initargs=init_args) as pool:
        for map_idx, data, _dt in pool.imap_unordered(
            _generate_one, pending, chunksize=2
        ):
            current_batch.append((map_idx, data))
            completed_count += 1

            now = time.perf_counter()
            if now - last_report >= 10 or completed_count == len(pending):
                elapsed   = now - t0
                rate      = completed_count / elapsed if elapsed > 0 else 0
                remaining = len(pending) - completed_count
                eta       = remaining / rate if rate > 0 else 0
                print(
                    f"    {completed_count:>6d}/{len(pending)}"
                    f"  ({rate:.1f} maps/s, ETA {eta/60:.1f} min)"
                )
                last_report = now

            if len(current_batch) >= args.shard_size:
                start = phase_start + shard_seq * args.shard_size
                end   = start + args.shard_size - 1
                shard_path = args.out_dir / f"{shard_prefix}_{start:05d}_{end:05d}.h5"
                _write_hdf5_shard(
                    shard_path, current_batch, pool_config,
                    list(rock_vocab.keys()), list(fm_vocab.keys()),
                )
                current_batch = []
                shard_seq += 1

    if current_batch:
        first_idx = min(x[0] for x in current_batch)
        last_idx  = max(x[0] for x in current_batch)
        shard_path = args.out_dir / f"{shard_prefix}_{first_idx:05d}_{last_idx:05d}.h5"
        _write_hdf5_shard(
            shard_path, current_batch, pool_config,
            list(rock_vocab.keys()), list(fm_vocab.keys()),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate formation-based belief map pool for neural-belief training."
    )
    p.add_argument("--out-dir", type=Path, default=Path("C://dataset_formation"))
    p.add_argument("--n-maps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--shard-size", type=int, default=500,
                   help="number of maps per HDF5 shard (default 500)")
    p.add_argument(
        "--samples-per-map", type=int, default=20,
        help="drill patterns stored per map",
    )
    p.add_argument("--min-drills", type=int, default=1)
    p.add_argument("--max-drills", type=int, default=15)
    p.add_argument(
        "--bank", type=Path,
        default=Path("data/clean/formation_distributions.pkl"),
    )
    p.add_argument(
        "--geom", type=Path,
        default=Path("data/clean/formation_geometry.pkl"),
    )
    p.add_argument(
        "--prior", type=Path,
        default=Path("data/clean/discovery_prior.pkl"),
    )
    p.add_argument(
        "--n-bodies", type=int, default=None,
        help="cap ore body count per map (0-3); omit to use SimConfig default (2)",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimConfig()
    n_x, n_y = sim_cfg.n_x, sim_cfg.n_y
    variables = list(sim_cfg.variables)

    vocab_path  = args.out_dir / "labels_vocab.pkl"
    config_path = args.out_dir / "config.pkl"

    # ── Build or load label vocabs ─────────────────────────────────────────
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        rock_vocab = vocabs["rocks"]
        fm_vocab   = vocabs["formations"]
        print(f"  loaded label vocab from {vocab_path}")
    else:
        bank = FormationDistributionBank.load(args.bank)
        geom = FormationGeometry.load(args.geom)
        rock_vocab, fm_vocab = _build_vocabs(bank, geom)
        with open(vocab_path, "wb") as f:
            pickle.dump({"rocks": rock_vocab, "formations": fm_vocab}, f)
        print(
            f"  label vocab: {len(rock_vocab)} rocks, {len(fm_vocab)} formations"
            f" -> {vocab_path}"
        )
        del bank, geom

    # ── Write / update pool config ─────────────────────────────────────────
    pool_config = {
        "cache_version": "2",
        "variables": variables,
        "n_maps": args.n_maps,
        "seed": args.seed,
        "samples_per_map": args.samples_per_map,
        "min_drills": args.min_drills,
        "max_drills": args.max_drills,
        "n_x": n_x,
        "n_y": n_y,
    }
    with open(config_path, "wb") as f:
        pickle.dump(pool_config, f)

    # ── Phase definitions ──────────────────────────────────────────────────
    # Phase 1: first 4 batches — only 0 or 1 ore body maps
    # Phase 2: next 4 batches  — only 2 or 3 ore body maps (min_ore_bodies=2)
    # Phase 3: remaining maps  — stratified (default ore body distribution)
    _BATCHES_PER_PHASE = 4
    phase1_n = min(_BATCHES_PER_PHASE * args.shard_size, args.n_maps)
    phase2_n = min(_BATCHES_PER_PHASE * args.shard_size, max(0, args.n_maps - phase1_n))
    phase3_n = max(0, args.n_maps - phase1_n - phase2_n)

    phase_defs = [
        # (global_start, count, max_ore_bodies, min_ore_bodies, shard_prefix)
        (0,                       phase1_n, 1,            0, "maps_0_and_1_orebodies"),
        (phase1_n,                phase2_n, 3,            2, "maps_2_and_3_orebodies"),
        (phase1_n + phase2_n,     phase3_n, args.n_bodies, 0, "maps_stratified"),
    ]

    # ── Resume: find which maps are already in existing shards ─────────────
    completed = _find_completed_maps(args.out_dir)
    total_pending = sum(
        1
        for ph_start, ph_n, *_ in phase_defs
        for i in range(ph_start, ph_start + ph_n)
        if i not in completed
    )
    if total_pending == 0:
        print(f"  all {args.n_maps} maps already written to {args.out_dir}")
        return
    skipped = args.n_maps - total_pending
    print(
        f"  generating {total_pending} maps with {args.workers} workers"
        + (f" (skipping {skipped} already in shards)" if skipped else "")
    )

    # ── Phase-based parallel generation ───────────────────────────────────
    t0_total = time.perf_counter()
    for ph_start, ph_n, max_ore_bodies, min_ore_bodies, shard_prefix in phase_defs:
        if ph_n == 0:
            continue
        pending = [i for i in range(ph_start, ph_start + ph_n) if i not in completed]
        if not pending:
            print(f"  [{shard_prefix}] all {ph_n} maps already written")
            continue
        print(
            f"  [{shard_prefix}] generating {len(pending)} maps"
            f"  (ore bodies: min={min_ore_bodies}, max={max_ore_bodies})"
        )
        _generate_phase(
            pending, ph_start, shard_prefix, args, pool_config,
            rock_vocab, fm_vocab, max_ore_bodies, min_ore_bodies, n_x, n_y, variables,
        )

    total_shards = len(list(args.out_dir.glob("maps_*.h5")))
    print(f"\n  done in {(time.perf_counter() - t0_total) / 60:.1f} min")
    print(f"  pool -> {args.out_dir}  ({args.n_maps} maps, {total_shards} shards)")


if __name__ == "__main__":
    main()
