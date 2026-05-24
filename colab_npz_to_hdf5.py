# ============================================================
# colab_npz_to_hdf5.py
# ============================================================
# Converts a balanced subset of the .npz map pool into a single
# HDF5 shard (250 zero-body maps + 250 one-body maps = 500 total).
#
# Run cell-by-cell in Google Colab, or as a standalone script.
# ============================================================

# ============================================================
# SECTION 0 — imports
# ============================================================
import os
import pickle
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# ============================================================
# SECTION 1 — mount Google Drive
# ============================================================
from google.colab import drive

drive.mount("/content/drive")

# ============================================================
# SECTION 2 — configuration  (edit these paths as needed)
# ============================================================

# Directory on Drive that holds map_00000.npz … map_NNNNN.npz
# plus config.pkl, labels_vocab.pkl, n_bodies_index.npy
DRIVE_POOL_DIR = Path("/content/drive/MyDrive/Thesis/data/dataset_complete")

# Local Colab staging area — read/write is much faster here than on Drive
LOCAL_STAGE_DIR = Path("/content/selected_npz")

# Where to save the final HDF5 file
OUT_HDF5_DIR = Path("/content/drive/MyDrive/Thesis/data/dataset_HDF5")
OUT_HDF5_FILE = OUT_HDF5_DIR / "maps_00000_00499.h5"

N_PER_CLASS = 250   # 250 × class-0  +  250 × class-1  =  500 total
SEED = 42

# HDF5 compression: "gzip" level 1 gives fast reads and good size reduction.
# Set COMPRESS = None to write uncompressed (fastest possible reads).
COMPRESS = "gzip"
COMPRESS_LEVEL = 1

# ============================================================
# SECTION 3 — load ore-body index and select balanced subset
# ============================================================

print("=" * 60)
print("STEP 1/6 — selecting balanced map subset")
print("=" * 60)

index_path = DRIVE_POOL_DIR / "n_bodies_index.npy"

if index_path.exists():
    body_index = np.load(index_path)   # shape (n_maps,)  dtype int8
    print(f"  Loaded n_bodies_index.npy  — pool size: {len(body_index)} maps")
else:
    # Fallback: build the index by scanning every npz (slow but robust)
    print("  n_bodies_index.npy not found — scanning all .npz files (slow) …")
    map_files = sorted(DRIVE_POOL_DIR.glob("map_*.npz"))
    body_index = np.array(
        [int(np.load(p)["n_bodies"]) for p in map_files], dtype=np.int8
    )
    print(f"  Scanned {len(body_index)} maps.")

zero_candidates = np.where(body_index == 0)[0]
one_candidates  = np.where(body_index == 1)[0]

print(f"  Maps with 0 ore bodies : {len(zero_candidates)}")
print(f"  Maps with 1 ore body   : {len(one_candidates)}")

assert len(zero_candidates) >= N_PER_CLASS, (
    f"Need {N_PER_CLASS} zero-body maps, only {len(zero_candidates)} available."
)
assert len(one_candidates) >= N_PER_CLASS, (
    f"Need {N_PER_CLASS} one-body maps, only {len(one_candidates)} available."
)

rng = np.random.default_rng(SEED)
zero_chosen = rng.choice(zero_candidates, size=N_PER_CLASS, replace=False)
one_chosen  = rng.choice(one_candidates,  size=N_PER_CLASS, replace=False)

selected_indices = np.concatenate([zero_chosen, one_chosen]).astype(np.int32)
class_labels     = np.array([0] * N_PER_CLASS + [1] * N_PER_CLASS, dtype=np.int8)

# Shuffle so HDF5 order is random rather than "all zeros first"
perm             = rng.permutation(len(selected_indices))
selected_indices = selected_indices[perm]
class_labels     = class_labels[perm]

# ---- validate balance ----
n_zero = int((class_labels == 0).sum())
n_one  = int((class_labels == 1).sum())
assert n_zero == N_PER_CLASS, f"Balance check failed: {n_zero} zero-body maps (expected {N_PER_CLASS})"
assert n_one  == N_PER_CLASS, f"Balance check failed: {n_one} one-body maps (expected {N_PER_CLASS})"
print(f"\n  Selected {len(selected_indices)} maps  "
      f"({n_zero} zero-body, {n_one} one-body)  — balance check OK")

# ============================================================
# SECTION 4 — copy selected .npz files to local Colab storage
# ============================================================

print("\n" + "=" * 60)
print("STEP 2/6 — copying .npz files to local storage")
print("=" * 60)

LOCAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)

n_copied = 0
for map_idx in selected_indices:
    src = DRIVE_POOL_DIR / f"map_{map_idx:05d}.npz"
    dst = LOCAL_STAGE_DIR / f"map_{map_idx:05d}.npz"
    if not dst.exists():
        shutil.copy2(src, dst)
        n_copied += 1

print(f"  Done — {n_copied} files copied, "
      f"{len(selected_indices) - n_copied} already present.")

# ============================================================
# SECTION 5 — load config / vocab metadata from Drive
# ============================================================

print("\n" + "=" * 60)
print("STEP 3/6 — loading pool metadata")
print("=" * 60)

config_path = DRIVE_POOL_DIR / "config.pkl"
vocab_path  = DRIVE_POOL_DIR / "labels_vocab.pkl"

pool_config: dict = {}
if config_path.exists():
    with open(config_path, "rb") as f:
        pool_config = pickle.load(f)
    print("  Loaded config.pkl")
    for k, v in pool_config.items():
        print(f"    {k}: {v}")
else:
    print("  config.pkl not found — pool-level metadata will be omitted.")

rock_vocab:      list[str] = []
formation_vocab: list[str] = []
if vocab_path.exists():
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    rock_vocab      = vocabs.get("rocks",      [])
    formation_vocab = vocabs.get("formations", [])
    print(f"  Loaded labels_vocab.pkl  "
          f"({len(rock_vocab)} rocks, {len(formation_vocab)} formations)")
else:
    print("  labels_vocab.pkl not found — vocabulary metadata will be omitted.")

# ============================================================
# SECTION 6 — probe first file and validate shapes across all
# ============================================================

print("\n" + "=" * 60)
print("STEP 4/6 — validating array shapes")
print("=" * 60)

probe_idx  = int(selected_indices[0])
probe_path = LOCAL_STAGE_DIR / f"map_{probe_idx:05d}.npz"
probe      = np.load(probe_path)

print(f"  Reference file: map_{probe_idx:05d}.npz")
ref_shapes = {}
for key in probe.files:
    arr = probe[key]
    ref_shapes[key] = arr.shape
    print(f"    {key:20s}  dtype={arr.dtype}  shape={arr.shape}")
probe.close()

# Validate all selected files match the reference shapes
print(f"\n  Checking {len(selected_indices)} files …", end="", flush=True)
shape_errors: list[str] = []
for map_idx in selected_indices:
    path = LOCAL_STAGE_DIR / f"map_{map_idx:05d}.npz"
    d = np.load(path)
    for key, ref_shape in ref_shapes.items():
        if d[key].shape != ref_shape:
            shape_errors.append(
                f"map_{map_idx:05d}  key={key}  got={d[key].shape}  expected={ref_shape}"
            )
    d.close()

if shape_errors:
    for err in shape_errors[:10]:
        print(f"\n  SHAPE MISMATCH: {err}")
    raise RuntimeError(
        f"{len(shape_errors)} shape mismatches — cannot stack arrays into HDF5."
    )

print("  all consistent — OK")

N = len(selected_indices)

# ============================================================
# SECTION 7 — write HDF5
# ============================================================

print("\n" + "=" * 60)
print("STEP 5/6 — writing HDF5")
print("=" * 60)

OUT_HDF5_DIR.mkdir(parents=True, exist_ok=True)

compress_kwargs: dict = {}
if COMPRESS:
    compress_kwargs = {"compression": COMPRESS, "compression_opts": COMPRESS_LEVEL,
                       "shuffle": True}


def _make_ds(hf: h5py.File, name: str, shape: tuple, dtype, chunks: tuple | None = None):
    """Create a pre-allocated, optionally chunked+compressed dataset."""
    kw = dict(compress_kwargs)
    if chunks is not None:
        kw["chunks"] = chunks
    return hf.create_dataset(name, shape=shape, dtype=dtype, **kw)


t_write = datetime.now()
print(f"  Writing to {OUT_HDF5_FILE} …")

with h5py.File(OUT_HDF5_FILE, "w") as hf:

    # ---- file-level attributes ----
    hf.attrs["n_maps"]           = N
    hf.attrs["n_zero_body"]      = n_zero
    hf.attrs["n_one_body"]       = n_one
    hf.attrs["selection_seed"]   = SEED
    hf.attrs["created_utc"]      = datetime.now(timezone.utc).isoformat()
    # Pool-level metadata from config.pkl
    for key in ("n_x", "n_y", "samples_per_map", "min_drills", "max_drills", "seed"):
        if key in pool_config:
            hf.attrs[f"pool_{key}"] = pool_config[key]
    if pool_config.get("variables"):
        hf.attrs["variables"] = np.array([s.encode("utf-8") for s in pool_config["variables"]])
    if rock_vocab:
        hf.attrs["rock_vocab"] = np.array([s.encode("utf-8") for s in rock_vocab])
    if formation_vocab:
        hf.attrs["formation_vocab"] = np.array([s.encode("utf-8") for s in formation_vocab])

    # ---- metadata vectors (written all-at-once) ----
    hf.create_dataset("class_label", data=class_labels,              **compress_kwargs)
    hf.create_dataset("map_index",   data=selected_indices,          **compress_kwargs)
    n_bodies_all = np.array(
        [int(body_index[int(i)]) for i in selected_indices], dtype=np.int8
    )
    hf.create_dataset("n_bodies", data=n_bodies_all, **compress_kwargs)

    # Variable-length UTF-8 strings for filenames
    dt_str = h5py.string_dtype(encoding="utf-8")
    filenames_ds = hf.create_dataset("filenames", (N,), dtype=dt_str)
    for i, map_idx in enumerate(selected_indices):
        filenames_ds[i] = f"map_{int(map_idx):05d}.npz"

    # ---- pre-allocate large numeric arrays ----
    # Chunk on sample dimension (one chunk = one sample) for fast row reads.
    def _single_sample_chunk(full_shape: tuple) -> tuple:
        return (1, *full_shape[1:])

    bh_shape       = (N, *ref_shapes["boreholes"])
    yt_shape       = (N, *ref_shapes["yield_target"])
    rocks_shape    = (N, *ref_shapes["rocks"])
    fm_shape       = (N, *ref_shapes["formations"])
    dl_shape       = (N, *ref_shapes["drill_locs"])
    dov_shape      = (N, *ref_shapes["drill_ore_vals"])
    dc_shape       = (N, *ref_shapes["drill_counts"])

    boreholes_ds      = _make_ds(hf, "boreholes",      bh_shape,    np.float32, _single_sample_chunk(bh_shape))
    yield_target_ds   = _make_ds(hf, "yield_target",   yt_shape,    np.float32, _single_sample_chunk(yt_shape))
    rocks_ds          = _make_ds(hf, "rocks",           rocks_shape, np.int8,   _single_sample_chunk(rocks_shape))
    formations_ds     = _make_ds(hf, "formations",      fm_shape,    np.int8,   _single_sample_chunk(fm_shape))
    drill_locs_ds     = _make_ds(hf, "drill_locs",      dl_shape,    np.int16,  _single_sample_chunk(dl_shape))
    drill_ore_vals_ds = _make_ds(hf, "drill_ore_vals",  dov_shape,   np.float32, _single_sample_chunk(dov_shape))
    drill_counts_ds   = _make_ds(hf, "drill_counts",    dc_shape,    np.int16,  _single_sample_chunk(dc_shape))

    print(f"  Planned dataset shapes:")
    for ds_name, ds in [
        ("boreholes",      boreholes_ds),
        ("yield_target",   yield_target_ds),
        ("rocks",          rocks_ds),
        ("formations",     formations_ds),
        ("drill_locs",     drill_locs_ds),
        ("drill_ore_vals", drill_ore_vals_ds),
        ("drill_counts",   drill_counts_ds),
    ]:
        print(f"    {ds_name:20s}  {ds.shape}  dtype={ds.dtype}")

    # ---- fill row by row ----
    print(f"\n  Writing {N} samples …")
    for i, map_idx in enumerate(selected_indices):
        path = LOCAL_STAGE_DIR / f"map_{int(map_idx):05d}.npz"
        d = np.load(path)

        boreholes_ds[i]      = d["boreholes"]
        yield_target_ds[i]   = d["yield_target"]
        rocks_ds[i]          = d["rocks"]
        formations_ds[i]     = d["formations"]
        drill_locs_ds[i]     = d["drill_locs"]
        drill_ore_vals_ds[i] = d["drill_ore_vals"]
        drill_counts_ds[i]   = d["drill_counts"]

        d.close()

        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"    [{i + 1:4d}/{N}] written")

elapsed = (datetime.now() - t_write).total_seconds()
file_size_mb = OUT_HDF5_FILE.stat().st_size / 1024**2
print(f"\n  HDF5 written in {elapsed:.1f}s  —  {file_size_mb:.1f} MB")
print(f"  Path: {OUT_HDF5_FILE}")

# ============================================================
# SECTION 8 — read-back verification
# ============================================================

print("\n" + "=" * 60)
print("STEP 6/6 — read-back verification")
print("=" * 60)

with h5py.File(OUT_HDF5_FILE, "r") as hf:

    print("File-level attributes:")
    for k, v in hf.attrs.items():
        val_str = str(v) if len(str(v)) < 80 else str(v)[:77] + "…"
        print(f"  {k}: {val_str}")

    print("\nDatasets (key → shape  dtype):")
    def _print_datasets(group, prefix=""):
        for k in group.keys():
            item = group[k]
            if isinstance(item, h5py.Dataset):
                print(f"  {prefix}{k:24s}  shape={item.shape}  dtype={item.dtype}")
            else:
                _print_datasets(item, prefix=k + "/")
    _print_datasets(hf)

    # Spot-check 3 random samples
    sample_idxs = sorted(random.sample(range(N), 3))
    print(f"\nSpot-checking samples at positions {sample_idxs}:")
    for pos in sample_idxs:
        fname       = hf["filenames"][pos]
        # h5py may return bytes or str depending on version
        fname_str   = fname.decode() if isinstance(fname, bytes) else str(fname)
        map_idx_val = int(hf["map_index"][pos])
        n_bod       = int(hf["n_bodies"][pos])
        cls         = int(hf["class_label"][pos])
        yt          = hf["yield_target"][pos]
        bh          = hf["boreholes"][pos]
        dc          = hf["drill_counts"][pos]

        print(f"\n  Position {pos}:")
        print(f"    filename     : {fname_str}")
        print(f"    map_index    : {map_idx_val}")
        print(f"    n_bodies     : {n_bod}")
        print(f"    class_label  : {cls}  (expected: {int(cls == 1)})")
        print(f"    yield_target : shape={yt.shape}  "
              f"range=[{yt.min():.4f}, {yt.max():.4f}]")
        print(f"    boreholes    : shape={bh.shape}  "
              f"range=[{bh.min():.4f}, {bh.max():.4f}]")
        print(f"    drill_counts : {dc.tolist()}")

        # Cross-check: class_label must match n_bodies
        expected_cls = 0 if n_bod == 0 else 1
        assert cls == expected_cls, (
            f"class_label={cls} does not match n_bodies={n_bod} at position {pos}"
        )

    # Validate overall class balance
    all_labels = hf["class_label"][:]
    assert (all_labels == 0).sum() == N_PER_CLASS, "Balance mismatch in written file!"
    assert (all_labels == 1).sum() == N_PER_CLASS, "Balance mismatch in written file!"
    print(f"\n  Class balance: {(all_labels==0).sum()} zero-body, "
          f"{(all_labels==1).sum()} one-body  — OK")

print("\nRead-back verification complete — HDF5 file is valid.")
print(f"\nFinal file: {OUT_HDF5_FILE}  ({file_size_mb:.1f} MB)")
