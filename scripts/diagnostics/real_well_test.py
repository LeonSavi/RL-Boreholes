"""Real-well transfer test for the borehole encoder (Charlie's point 4).

The encoder is trained ONLY on synthetic boreholes. Here we run it on REAL
NLOG wells that carry all five wireline channels, and compute the SAME metrics
used on synthetic data (silhouette / k-NN / linear probe by rock & formation),
plus a UMAP. This measures transfer to real log data (it is not held out from
the simulator fit, so it is a domain-transfer check, not unseen-geology
generalisation).

Uses the 5-channel checkpoints jepa.pt / ae.pt -- the ones that produced the
poster's synthetic +0.70 / +0.63 -- so synthetic-vs-real is apples-to-apples.

Writes plots/encoders/analysis/real_well_umap.png and prints a comparison table.
"""
import os
os.chdir("/github/RL-Boreholes")
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from encoder.autoencoder import load_checkpoint as load_ae_checkpoint
from encoder.jepa_encoder import load_jepa_checkpoint
import importlib.util as _ilu
# analysis_encoders.py was renamed to 8_analysis_encoders.py; a digit-prefixed
# module name cannot be imported with `import`, so load it from its file path.
_ae_path = Path(__file__).resolve().parents[2] / "8_analysis_encoders.py"
_spec = _ilu.spec_from_file_location("analysis_encoders", _ae_path)
AEMOD = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(AEMOD)
from encoder.encoder_validations.latent_validation import (
    compute_all_labels, reduce_to_2d,
)

VARS = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
NZ = 440
CELL_M = 10.0                      # 10 m per cell -> 0..4400 m
MIN_POPULATED_CELLS = 30           # >= ~300 m of logged column
PARQUET = "data/clean/samples.parquet"
OUT = Path("plots/encoders/analysis/real_well_umap.png")

SYN = {  # synthetic reference (silhouette_summary.csv, rock middle 50%)
    "AE": 0.627, "JEPA": 0.703,
}


def _mode_or_other(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "other"
    return s.value_counts().idxmax()


def build_real_wells():
    print(f"loading {PARQUET} ...", flush=True)
    df = pd.read_parquet(PARQUET)
    df = df[df["dataset"] == "NLOG"]
    # depth convention the simulator/encoder trained on = burial depth below
    # surface; fall back to `depth` where burial is missing.
    depth = df["burial_depth_m"].fillna(df["depth"])
    df = df.assign(_depth=depth)
    df = df[(df["_depth"] >= 0) & (df["_depth"] < NZ * CELL_M)]
    df = df[df["measurement"].isin(VARS) |
            df["measurement"].isna()]  # keep label-bearing rows too
    print(f"  NLOG rows in depth range: {len(df):,}", flush=True)

    wells_all5, values, rocks, forms, well_ids = [], [], [], [], []
    n_total = df["borehole"].nunique()
    for wid, g in df.groupby("borehole", sort=False):
        present = set(g.loc[g["measurement"].isin(VARS), "measurement"].unique())
        if not set(VARS).issubset(present):
            continue
        wells_all5.append(wid)
        cell = np.clip((g["_depth"].to_numpy() / CELL_M).astype(int), 0, NZ - 1)
        g = g.assign(_cell=cell)
        # logs: mean value per (cell, measurement)
        logs = (g[g["measurement"].isin(VARS)]
                .pivot_table(index="_cell", columns="measurement",
                             values="value", aggfunc="mean"))
        logs = logs.reindex(columns=VARS)
        arr = np.full((len(VARS), NZ), np.nan, dtype=np.float32)
        for vi, v in enumerate(VARS):
            if v in logs.columns:
                col = logs[v]
                arr[vi, col.index.to_numpy()] = col.to_numpy(dtype=np.float32)
        n_pop = int(np.isfinite(arr).any(axis=0).sum())
        if n_pop < MIN_POPULATED_CELLS:
            continue
        # per-cell majority labels
        rk = g.groupby("_cell")["rock_type_fine"].agg(_mode_or_other)
        fm = g.groupby("_cell")["formation"].agg(_mode_or_other)
        rrow = np.full(NZ, "other", dtype=object)
        frow = np.full(NZ, "other", dtype=object)
        rrow[rk.index.to_numpy()] = rk.to_numpy()
        frow[fm.index.to_numpy()] = fm.to_numpy()
        values.append(arr); rocks.append(rrow); forms.append(frow)
        well_ids.append(wid)

    print(f"  NLOG wells total: {n_total}", flush=True)
    print(f"  with all 5 channels: {len(wells_all5)}", flush=True)
    print(f"  usable (>= {MIN_POPULATED_CELLS} populated cells): {len(values)}",
          flush=True)
    return (np.stack(values), np.stack(rocks), np.stack(forms), well_ids)


def main():
    device = "cpu"
    values, rocks, forms, well_ids = build_real_wells()
    print(f"  values: {values.shape}", flush=True)

    label_sets = compute_all_labels(rocks, forms, nz=NZ)
    # robust extra label: dominant rock over the well's POPULATED column
    rock_col = []
    for i in range(len(rocks)):
        c = Counter(rocks[i]); c.pop("other", None)
        rock_col.append(c.most_common(1)[0][0] if c else "other")
    label_sets["rock (whole column)"] = rock_col
    for name, labs in label_sets.items():
        print(f"  label '{name}': top {Counter(labs).most_common(4)}", flush=True)

    Z_by, z2_by, sil_by = {}, {}, {}
    for name, path, loader, enc in [
        ("AE",   "checkpoints/ae_depth.pt",   load_ae_checkpoint,   AEMOD.encode_with_ae),
        ("JEPA", "checkpoints/jepa_final.pt", load_jepa_checkpoint, AEMOD.encode_with_jepa),
    ]:
        print(f"\n── {name}: {path}", flush=True)
        model, stats, vars_ = loader(Path(path), device=device)
        Z = enc(model, values, stats, vars_, device)
        Z_by[name] = Z
        sil_by[name] = AEMOD.silhouette_table(Z, label_sets)
        print(f"  latents {Z.shape}", flush=True)
        for ls in ["rock (middle 50%)", "rock (whole column)",
                   "formation (middle 50%)"]:
            sc, n = sil_by[name][ls]
            knn = AEMOD.knn_label_accuracy(Z, label_sets[ls])
            prb = AEMOD.probe_accuracy(Z, label_sets[ls])
            print(f"  {ls:24s} sil={sc:+.3f} (n={n:3d})  knn={knn:.3f}  probe={prb:.3f}",
                  flush=True)
        z2_by[name] = reduce_to_2d(Z, method="umap")

    # UMAP figure, coloured by whole-column dominant rock (the mid-50% label is
    # mostly "mixed" on real wells, so it is not a usable colouring).
    key = "rock (whole column)"
    AEMOD.plot_jepa_vs_ae_umap(
        z2_by["JEPA"], z2_by["AE"], label_sets[key],
        sil_by["JEPA"][key][0], sil_by["AE"][key][0], OUT,
        suptitle="163 real NLOG wells, coloured by dominant rock "
                 "(encoder trained only on synthetic)",
    )

    print("\n=========== REAL-WELL TRANSFER (dominant rock, whole column) ===========")
    print(f"{'encoder':6s} {'sil_real':>8s} {'knn_real':>8s} {'probe_real':>10s} "
          f"{'n':>4s}   (synthetic ref: sil~+0.70 mid-50%, probe~1.00)")
    for enc in ("JEPA", "AE"):
        sc, n = sil_by[enc][key]
        knn = AEMOD.knn_label_accuracy(Z_by[enc], label_sets[key])
        prb = AEMOD.probe_accuracy(Z_by[enc], label_sets[key])
        print(f"{enc:6s} {sc:+8.3f} {knn:8.3f} {prb:10.3f} {n:4d}", flush=True)
    maj = Counter(label_sets[key]); maj.pop("other", None)
    base = maj.most_common(1)[0][1] / sum(v for k, v in maj.items())
    print(f"majority-class baseline (probe chance): {base:.3f}")
    print("========================================================================")


if __name__ == "__main__":
    main()
