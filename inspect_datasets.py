"""Inspect contents of C:/dataset (borehole .npy files) and C:/dataset_raw_maps (HDF5)."""
import pickle
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# C:/dataset  —  config.pkl
# ---------------------------------------------------------------------------

print("=" * 60)
print("C:/dataset  —  config.pkl")
print("=" * 60)

with open("C:/dataset/config.pkl", "rb") as f:
    config = pickle.load(f)

for k, v in config.items():
    print(f"  {k}: {v}")

variables = config.get("variables", [])


# ---------------------------------------------------------------------------
# C:/dataset  —  stats.pkl
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset  —  stats.pkl")
print("=" * 60)

with open("C:/dataset/stats.pkl", "rb") as f:
    stats = pickle.load(f)

for var, (mu, sd) in stats.items():
    print(f"  {var:20s}  mean={mu:.6f}  std={sd:.6f}")


# ---------------------------------------------------------------------------
# C:/dataset  —  boreholes_00000.npy
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset  —  boreholes_00000.npy")
print("=" * 60)

bh = np.load("C:/dataset/boreholes_00000.npy").astype(np.float32)
var_names = variables if variables else [f"var_{i}" for i in range(bh.shape[1])]

print(f"  Full array shape: {bh.shape}  (n_boreholes, n_variables, n_depth)")
for i, var in enumerate(var_names):
    v = bh[:, i, :]
    print(f"  {var:20s}  min={v.min():.4f}  max={v.max():.4f}  "
          f"mean={v.mean():.4f}  std={v.std():.4f}  NaNs={np.isnan(v).sum()}")


# ---------------------------------------------------------------------------
# C:/dataset  —  labels_00000.npz
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset  —  labels_00000.npz")
print("=" * 60)

labels = np.load("C:/dataset/labels_00000.npz")
for key in labels.files:
    arr = labels[key]
    print(f"  {key:20s}  shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min()}  max={arr.max()}  unique={np.unique(arr).size}")


# ---------------------------------------------------------------------------
# C:/dataset_raw_maps  —  HDF5 file-level metadata
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset_raw_maps/raw_pool.h5  —  file metadata")
print("=" * 60)

with h5py.File("C:/dataset_raw_maps/raw_pool.h5", "r") as f:
    attrs = dict(f.attrs)
    n_maps = attrs.get("n_maps", "?")
    top_keys = list(f.keys())

print(f"  Number of maps : {n_maps}")
print(f"  Top-level keys : {top_keys}")
for k, v in attrs.items():
    print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# C:/dataset_raw_maps  —  first map: true_map datasets
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset_raw_maps/raw_pool.h5  —  first map: true_map")
print("=" * 60)

with h5py.File("C:/dataset_raw_maps/raw_pool.h5", "r") as f:
    first_key = sorted(f["maps"].keys())[0]
    grp = f["maps"][first_key]
    tm = grp["true_map"]

    print(f"  Map key: {first_key}")
    for name in tm.keys():
        ds = tm[name]
        if hasattr(ds, "shape"):
            print(f"  true_map/{name:20s}  shape={ds.shape}  dtype={ds.dtype}")
        else:
            for sub_name, sub_ds in ds.items():
                print(f"  true_map/{name}/{sub_name:15s}  shape={sub_ds.shape}  dtype={sub_ds.dtype}")


# ---------------------------------------------------------------------------
# C:/dataset_raw_maps  —  first map: drill patterns & target
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("C:/dataset_raw_maps/raw_pool.h5  —  first map: drill patterns & target")
print("=" * 60)

with h5py.File("C:/dataset_raw_maps/raw_pool.h5", "r") as f:
    first_key = sorted(f["maps"].keys())[0]
    grp = f["maps"][first_key]

    for name in ("drill_locs", "drill_ore_vals", "drill_counts", "target"):
        arr = grp[name][:]
        print(f"  {name:20s}  shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")
