# ============================================================
# colab_npz_to_hdf5_full.py
# ============================================================
# Full migration of an .npz map pool to a suite of HDF5 shards.
# Each shard holds 500 maps. Three shard types are produced:
#
#   maps_0_and_1_orebodies_NNN.h5  250 zero-body + 250 one-body
#   maps_N_orebodies_NNN.h5        500 maps, all with N ore bodies
#   maps_stratified_NNN.h5         <500-per-class tail, balanced mix
#
# After each shard is written, the used .npz files are deleted
# from Drive to free space (controlled by DELETE_FROM_DRIVE).
#
# Resume-safe: already-converted map indices are recovered from
# existing HDF5 files so a crash mid-run can be safely restarted.
# ============================================================

from __future__ import annotations

import pickle
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import h5py
import numpy as np
from google.colab import drive

# ============================================================
# SECTION 1 — mount Drive
# ============================================================

drive.mount("/content/drive")

# ============================================================
# SECTION 2 — configuration
# ============================================================

# Source .npz pool (must contain config.pkl, labels_vocab.pkl, n_bodies_index.npy)
DRIVE_POOL_DIR = Path("/content/drive/MyDrive/Thesis/data/dataset_complete")

# Local staging area — much faster I/O than Drive
LOCAL_STAGE_DIR = Path("/content/stage_npz")

# Output directory for HDF5 shards
OUT_HDF5_DIR = Path("/content/drive/MyDrive/Thesis/data/dataset_HDF5")

# RNG seed for pool shuffling
SEED = 42

# Set to False for a dry run that skips Drive deletion
DELETE_FROM_DRIVE = True

COMPRESS       = "gzip"
COMPRESS_LEVEL = 1

LOCAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_HDF5_DIR.mkdir(parents=True, exist_ok=True)

compress_kwargs = {
    "compression": COMPRESS,
    "compression_opts": COMPRESS_LEVEL,
    "shuffle": True,
}

# ============================================================
# SECTION 3 — load pool metadata
# ============================================================

print("=" * 60)
print("STEP 1 — loading pool metadata")
print("=" * 60)

with open(DRIVE_POOL_DIR / "config.pkl", "rb") as _f:
    pool_config: dict = pickle.load(_f)
with open(DRIVE_POOL_DIR / "labels_vocab.pkl", "rb") as _f:
    _vocabs = pickle.load(_f)
rock_vocab: list[str] = _vocabs.get("rocks", [])
fm_vocab:   list[str] = _vocabs.get("formations", [])
body_index: np.ndarray = np.load(DRIVE_POOL_DIR / "n_bodies_index.npy")  # (N,) int8

print(f"  Pool size : {len(body_index)} maps")
for _cls in range(4):
    print(f"  Class {_cls}   : {int((body_index == _cls).sum())} maps")

# ============================================================
# SECTION 4 — probe reference shapes from first .npz
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 — probing reference shapes")
print("=" * 60)

_probe_idx  = int(np.where(body_index >= 0)[0][0])
_probe_path = DRIVE_POOL_DIR / f"map_{_probe_idx:05d}.npz"
_probe      = np.load(_probe_path)
ref_shapes: dict[str, tuple] = {k: _probe[k].shape for k in _probe.files}
_probe.close()

print("  Key shapes:")
for _k, _s in ref_shapes.items():
    print(f"    {_k:20s} {_s}")

# ============================================================
# SECTION 5 — build pools (resume-safe)
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 — building pools")
print("=" * 60)

# Maps whose .npz still exists on Drive
existing_on_drive: set[int] = set(
    int(p.stem.split("_")[1])
    for p in DRIVE_POOL_DIR.glob("map_*.npz")
)
print(f"  .npz files on Drive : {len(existing_on_drive)}")

# Maps already converted to HDF5 (recovered from any existing shard)
already_processed: set[int] = set()
for _h5 in OUT_HDF5_DIR.glob("*.h5"):
    try:
        with h5py.File(_h5, "r") as _hf:
            if "map_index" in _hf:
                already_processed.update(int(i) for i in _hf["map_index"][:])
    except Exception:
        pass
print(f"  Already in HDF5     : {len(already_processed)}")

# Build shuffled pools per class
rng = np.random.default_rng(SEED)
pools: dict[int, list[int]] = {}
for _cls in range(4):
    _all = [
        int(i) for i in np.where(body_index == _cls)[0]
        if i in existing_on_drive and i not in already_processed
    ]
    rng.shuffle(_all)
    pools[_cls] = _all
    print(f"  Class {_cls} pending   : {len(pools[_cls])}")

# File counters — continue from existing shard count
n_mixed = len(list(OUT_HDF5_DIR.glob("maps_0_and_1_orebodies_*.h5")))
n_pure  = {cls: len(list(OUT_HDF5_DIR.glob(f"maps_{cls}_orebodies_*.h5"))) for cls in range(4)}
n_strat = len(list(OUT_HDF5_DIR.glob("maps_stratified_*.h5")))

# ============================================================
# SECTION 6 — write helper
# ============================================================


def _write_attrs(hf: h5py.File, n: int) -> None:
    hf.attrs["n_maps"]               = n
    hf.attrs["pool_n_x"]             = pool_config["n_x"]
    hf.attrs["pool_n_y"]             = pool_config["n_y"]
    hf.attrs["pool_samples_per_map"] = pool_config["samples_per_map"]
    hf.attrs["pool_min_drills"]      = pool_config["min_drills"]
    hf.attrs["pool_max_drills"]      = pool_config["max_drills"]
    hf.attrs["pool_seed"]            = pool_config["seed"]
    hf.attrs["created_utc"]          = datetime.now(timezone.utc).isoformat()
    if pool_config.get("variables"):
        hf.attrs["variables"] = np.array(
            [s.encode("utf-8") for s in pool_config["variables"]]
        )
    if rock_vocab:
        hf.attrs["rock_vocab"] = np.array([s.encode("utf-8") for s in rock_vocab])
    if fm_vocab:
        hf.attrs["formation_vocab"] = np.array([s.encode("utf-8") for s in fm_vocab])


def write_shard(map_indices: list[int], out_path: Path, file_type: str) -> None:
    """Copy .npz → write HDF5 → delete local → optionally delete from Drive."""
    N = len(map_indices)

    # ---- copy to local staging ----
    for idx in map_indices:
        src = DRIVE_POOL_DIR / f"map_{int(idx):05d}.npz"
        dst = LOCAL_STAGE_DIR / f"map_{int(idx):05d}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)

    # ---- write HDF5 ----
    n_bodies_all = np.array([int(body_index[i]) for i in map_indices], dtype=np.int8)
    class_counts = np.array([(n_bodies_all == c).sum() for c in range(4)], dtype=np.int32)

    def _chunk1(full_shape: tuple) -> tuple:
        return (1, *full_shape[1:])

    with h5py.File(out_path, "w") as hf:
        _write_attrs(hf, N)
        hf.attrs["file_type"]    = file_type
        hf.attrs["class_counts"] = class_counts  # [n0, n1, n2, n3]

        hf.create_dataset("n_bodies",  data=n_bodies_all,                              **compress_kwargs)
        hf.create_dataset("map_index", data=np.array(map_indices, dtype=np.int32),     **compress_kwargs)
        fn_ds = hf.create_dataset("filenames", (N,), dtype=h5py.string_dtype(encoding="utf-8"))
        for i, idx in enumerate(map_indices):
            fn_ds[i] = f"map_{int(idx):05d}.npz"

        bh_s  = (N, *ref_shapes["boreholes"])
        yt_s  = (N, *ref_shapes["yield_target"])
        ro_s  = (N, *ref_shapes["rocks"])
        fm_s  = (N, *ref_shapes["formations"])
        dl_s  = (N, *ref_shapes["drill_locs"])
        dov_s = (N, *ref_shapes["drill_ore_vals"])
        dc_s  = (N, *ref_shapes["drill_counts"])

        def _ds(name, shape, dtype):
            return hf.create_dataset(
                name, shape=shape, dtype=dtype,
                chunks=_chunk1(shape), **compress_kwargs
            )

        bh_ds  = _ds("boreholes",      bh_s,  np.float32)
        yt_ds  = _ds("yield_target",   yt_s,  np.float32)
        ro_ds  = _ds("rocks",          ro_s,  np.int8)
        fm_ds  = _ds("formations",     fm_s,  np.int8)
        dl_ds  = _ds("drill_locs",     dl_s,  np.int16)
        dov_ds = _ds("drill_ore_vals", dov_s, np.float32)
        dc_ds  = _ds("drill_counts",   dc_s,  np.int16)

        for i, idx in enumerate(map_indices):
            d = np.load(LOCAL_STAGE_DIR / f"map_{int(idx):05d}.npz")
            bh_ds[i]  = d["boreholes"]
            yt_ds[i]  = d["yield_target"]
            ro_ds[i]  = d["rocks"]
            fm_ds[i]  = d["formations"]
            dl_ds[i]  = d["drill_locs"]
            dov_ds[i] = d["drill_ore_vals"]
            dc_ds[i]  = d["drill_counts"]
            d.close()

    size_mb = out_path.stat().st_size / 1024 ** 2
    print(
        f"  wrote  {out_path.name}"
        f"  ({N} maps, dist={class_counts.tolist()}, {size_mb:.1f} MB)"
    )

    # ---- delete local copies ----
    for idx in map_indices:
        local = LOCAL_STAGE_DIR / f"map_{int(idx):05d}.npz"
        if local.exists():
            local.unlink()

    # ---- optionally delete from Drive ----
    if DELETE_FROM_DRIVE:
        n_del = 0
        for idx in map_indices:
            p = DRIVE_POOL_DIR / f"map_{int(idx):05d}.npz"
            if p.exists():
                p.unlink()
                n_del += 1
        print(f"         deleted {n_del} .npz files from Drive")

# ============================================================
# SECTION 7 — Phase 1: mixed 0+1 files
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 — mixed 0+1 orebody files")
print("=" * 60)

while len(pools[0]) >= 250 and len(pools[1]) >= 250:
    batch_0 = pools[0][:250];  pools[0] = pools[0][250:]
    batch_1 = pools[1][:250];  pools[1] = pools[1][250:]
    batch = batch_0 + batch_1
    rng.shuffle(batch)
    fname = OUT_HDF5_DIR / f"maps_0_and_1_orebodies_{n_mixed:03d}.h5"
    write_shard(batch, fname, "mixed_0_1")
    n_mixed += 1

print(f"\n  Mixed files total: {n_mixed}")
print(f"  Remaining  class-0={len(pools[0])}, class-1={len(pools[1])}")

# ============================================================
# SECTION 8 — Phase 2: pure single-class files
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 — pure single-class files")
print("=" * 60)

for cls in range(4):
    while len(pools[cls]) >= 500:
        batch = pools[cls][:500];  pools[cls] = pools[cls][500:]
        fname = OUT_HDF5_DIR / f"maps_{cls}_orebodies_{n_pure[cls]:03d}.h5"
        write_shard(batch, fname, f"pure_{cls}")
        n_pure[cls] += 1
    print(f"  Class {cls}: {n_pure[cls]} pure files written, {len(pools[cls])} remaining")

# ============================================================
# SECTION 9 — Phase 3: stratified tail files
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3 — stratified tail files")
print("=" * 60)

for cls in range(4):
    print(f"  Class {cls} tail: {len(pools[cls])} maps")

# Interleave classes for maximum per-file balance
active = [pools[cls] for cls in range(4) if pools[cls]]
interleaved: list[int] = []
for row in zip_longest(*active):
    for item in row:
        if item is not None:
            interleaved.append(item)

for i in range(0, len(interleaved), 500):
    batch = interleaved[i : i + 500]
    fname = OUT_HDF5_DIR / f"maps_stratified_{n_strat:03d}.h5"
    write_shard(batch, fname, "stratified")
    n_strat += 1

# ============================================================
# SECTION 10 — summary
# ============================================================

print("\n" + "=" * 60)
print("MIGRATION COMPLETE")
print("=" * 60)
print(f"  Mixed 0+1 files : {n_mixed}")
for cls in range(4):
    print(f"  Pure class-{cls}    : {n_pure[cls]}")
print(f"  Stratified tail : {n_strat}")
print(f"  Total shards    : {n_mixed + sum(n_pure.values()) + n_strat}")
print(f"  Output dir      : {OUT_HDF5_DIR}")
print(
    "\nNote: if maps_00000_00499.h5 exists from a prior test run,"
    " rename it to maps_0_and_1_orebodies_NNN.h5 or delete it;"
    " HDF5MapDirectory will not discover it under the old name."
)
