"""
Add a `burial_depth_m` column to samples.parquet using the per-well
water-depth table produced by attach_water_depth.py.

Why
---
Today the parquet's `depth` column is MSL-relative for offshore
wells and NAP-relative for onshore wells. The DistributionBank /
DiscoveryPrior / FormationGeometry bin both into the same depth
bins, which mixes water-column rows with shallow rock from onshore.

The fix
-------
burial_depth = depth - water_depth_m  (water_depth_m fillna 0)
  * Onshore: water_depth_m is NaN -> 0 -> burial_depth == depth.
  * Offshore: burial_depth = depth - water_depth.

We write `burial_depth_m` as a NEW column and leave `depth` alone
so the change is reversible and easy to diff.

Inputs
------
data/clean/samples.parquet       must carry `borehole`, `depth`,
                                  `location_type`
data/clean/well_water_depth.parquet  output of attach_water_depth.py

Outputs
-------
data/clean/samples.parquet            rewritten with new column
data/clean/samples.parquet.bak_pre_burial   backup of the original
plots/analysis/depth_vs_burial_hist.png     before/after histograms
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PARQUET_SAMPLES = Path("data/clean/samples.parquet")
PARQUET_WATER = Path("data/clean/well_water_depth.parquet")
PLOT_OUT = Path("plots/analysis/depth_vs_burial_hist.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, default=PARQUET_SAMPLES)
    p.add_argument("--water-depth", type=Path, default=PARQUET_WATER)
    p.add_argument("--plot", type=Path, default=PLOT_OUT)
    p.add_argument("--backup-suffix", default=".bak_pre_burial")
    p.add_argument("--no-overwrite", action="store_true",
                   help="emit samples.parquet.with_burial instead of "
                        "overwriting samples.parquet (useful for dry runs).")
    args = p.parse_args()

    print(f"loading samples: {args.samples}")
    df = pd.read_parquet(args.samples)
    n_rows = len(df)
    print(f"  {n_rows:,} rows, {df['borehole'].nunique():,} unique wells")

    print(f"loading water depths: {args.water_depth}")
    wd = pd.read_parquet(args.water_depth)
    print(f"  {len(wd):,} offshore wells with water_depth_m")

    # Left-join so onshore wells keep NaN water_depth.
    merged = df.merge(wd[["borehole", "water_depth_m"]],
                       on="borehole", how="left")
    n_onshore_rows = int(merged["water_depth_m"].isna().sum())
    n_offshore_rows = int((~merged["water_depth_m"].isna()).sum())
    print(f"\nrow counts after join:")
    print(f"  onshore / NA water-depth: {n_onshore_rows:,}")
    print(f"  offshore (with water-depth): {n_offshore_rows:,}")

    # burial depth -------------------------------------------------------
    wd_fill = merged["water_depth_m"].fillna(0.0)
    merged["burial_depth_m"] = merged["depth"] - wd_fill

    # Drop the auxiliary water_depth_m column from the parquet — it
    # belongs in well_water_depth.parquet, not samples.parquet. The
    # burial_depth_m column carries all the information downstream
    # consumers need.
    merged = merged.drop(columns=["water_depth_m"])

    print("\nbefore/after summary:")
    for loc in ("onshore", "offshore"):
        sub = merged[merged["location_type"] == loc]
        if len(sub) == 0:
            continue
        d_med = sub["depth"].median()
        b_med = sub["burial_depth_m"].median()
        d_max = sub["depth"].max()
        b_max = sub["burial_depth_m"].max()
        print(f"  {loc:8s}  depth   p50={d_med:8.1f} m   max={d_max:8.1f} m")
        print(f"            burial  p50={b_med:8.1f} m   max={b_max:8.1f} m   "
              f"  shift p50={b_med - d_med:+.1f} m")

    # Sanity: for onshore, burial == depth exactly.
    onshore_diff = (
        merged.loc[merged["location_type"] == "onshore", "depth"]
        - merged.loc[merged["location_type"] == "onshore", "burial_depth_m"]
    )
    if len(onshore_diff) > 0:
        max_abs = float(onshore_diff.abs().max())
        if max_abs > 1e-6:
            raise RuntimeError(
                f"onshore burial_depth differs from depth by up to {max_abs}"
                f" — water-depth join leaked onto onshore wells.")

    # Backup + write -----------------------------------------------------
    if args.no_overwrite:
        out_path = args.samples.with_suffix(".parquet.with_burial")
    else:
        bak = args.samples.with_name(args.samples.name + args.backup_suffix)
        if not bak.exists():
            print(f"\nbacking up {args.samples} -> {bak}")
            shutil.copy2(args.samples, bak)
        else:
            print(f"\nbackup already exists at {bak} (keeping it)")
        out_path = args.samples

    merged.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({len(merged):,} rows, "
          f"{len(merged.columns)} cols)")
    print(f"new column: burial_depth_m")

    # Plot ---------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    bins = np.linspace(0, 5000, 80)
    for ax, col, title in [
        (axes[0], "depth", "Original `depth` column (raw RT - KB)"),
        (axes[1], "burial_depth_m",
         "Corrected `burial_depth_m` (offshore: subtract water depth)"),
    ]:
        for loc, colour in (("onshore", "tab:orange"),
                             ("offshore", "tab:blue")):
            sub = merged[merged["location_type"] == loc][col].dropna()
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=bins, histtype="step", linewidth=1.4,
                    label=f"{loc} (n={len(sub):,})", color=colour)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("rows")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("depth [m]")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
