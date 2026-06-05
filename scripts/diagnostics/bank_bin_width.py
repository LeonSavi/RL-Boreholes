"""
Distribution-bank depth-bin width tradeoff.

For each candidate bin width, compute the count of unique (well, depth)
rows per (rock_type_fine, depth_bin) cell. Plot two diagnostics:

  (a) fraction of (rock, bin) cells with >=30 samples — the threshold
      DistributionBank.fit uses to admit a cell.
  (b) median samples-per-populated-cell.

The current code uses 400 m bins (15 cells). Goal: pick the smallest bin
width that keeps coverage acceptable while resolving more compaction
detail than 400 m. The DistributionBank already has a nearest-cell
fallback so empty cells aren't fatal — they're just resolved at the
nearest populated depth.

Also writes a (rock, bin) sample-count heatmap at the chosen width.

Outputs:
  plots/bank_bin_width_tradeoff.png
  plots/bank_sample_count_heatmap.png
  plots/analysis/bank_bin_width_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARQUET = Path("data/clean/samples.parquet")
OUT_TRADEOFF = Path("plots/bank_bin_width_tradeoff.png")
OUT_HEATMAP  = Path("plots/bank_sample_count_heatmap.png")
OUT_CSV      = Path("plots/analysis/bank_bin_width_summary.csv")

VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
DEPTH_MAX = 4400.0     # encoder window — bins past this never used
MIN_SAMPLES = 30       # matches DistributionBank.fit default
WIDTHS_M = [1, 2, 5, 10, 20, 25, 50, 100, 200, 400]


def load_wide() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = df[df["measurement"].isin(VARIABLES)]
    wide = df.pivot_table(
        index=["dataset", "borehole", "depth", "rock_type_fine"],
        columns="measurement",
        values="value",
        aggfunc="mean",
    ).reset_index()
    # one row per (well, depth, rock) — matches what bank.fit sees
    wide = wide[wide["depth"] <= DEPTH_MAX]
    wide = wide[wide["rock_type_fine"].notna()]
    return wide


def coverage_for_width(wide: pd.DataFrame, width: float) -> dict:
    edges = np.arange(0, DEPTH_MAX + width, width)
    bins = pd.cut(wide["depth"], bins=edges,
                  labels=list(range(len(edges) - 1)), include_lowest=True)
    counts = (
        wide.assign(_bin=bins)
            .groupby(["rock_type_fine", "_bin"], observed=True)
            .size()
    )
    n_rocks = wide["rock_type_fine"].nunique()
    n_bins = len(edges) - 1
    total_cells = n_rocks * n_bins
    populated = (counts >= MIN_SAMPLES).sum()
    median_n = float(counts[counts >= MIN_SAMPLES].median()) if populated else 0.0
    return {
        "width_m": width,
        "n_bins": n_bins,
        "total_cells": total_cells,
        "populated_cells": int(populated),
        "coverage_pct": 100.0 * populated / total_cells,
        "median_n_populated": median_n,
    }


def heatmap_for_width(wide: pd.DataFrame, width: float, out: Path) -> None:
    edges = np.arange(0, DEPTH_MAX + width, width)
    bins = pd.cut(wide["depth"], bins=edges,
                  labels=list(range(len(edges) - 1)), include_lowest=True)
    rocks = sorted(wide["rock_type_fine"].dropna().unique(),
                   key=lambda r: -(wide["rock_type_fine"] == r).sum())
    grid = np.zeros((len(rocks), len(edges) - 1), dtype=float)
    counts = (
        wide.assign(_bin=bins)
            .groupby(["rock_type_fine", "_bin"], observed=True)
            .size()
    )
    for (rock, b), c in counts.items():
        if rock in rocks:
            grid[rocks.index(rock), int(b)] = c

    fig, ax = plt.subplots(figsize=(11, 5))
    show = np.log10(grid + 1)
    im = ax.imshow(show, aspect="auto", cmap="viridis",
                   extent=[0, edges[-1], len(rocks), 0])
    cb = fig.colorbar(im, ax=ax, label="log10(samples + 1)")
    ax.set_yticks(np.arange(len(rocks)) + 0.5)
    ax.set_yticklabels(rocks, fontsize=8)
    ax.set_xlabel("depth [m]")
    ax.set_title(f"DistributionBank sample count per (rock, {int(width)} m bin) — "
                 f"darker = sparser (cells need >={MIN_SAMPLES} for KDE)")
    # mark the threshold with a contour at log10(30+1)≈1.49
    contour_val = np.log10(MIN_SAMPLES + 1)
    ax.contour(np.linspace(0, edges[-1], grid.shape[1]),
               np.arange(grid.shape[0]) + 0.5, show,
               levels=[contour_val], colors="white", linewidths=1.0, alpha=0.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_tradeoff(rows: list[dict], chosen: float, out: Path) -> None:
    df = pd.DataFrame(rows).sort_values("width_m")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(df["width_m"], df["coverage_pct"],
             marker="o", color="#4C72B0", linewidth=1.8)
    ax1.axvline(chosen, color="#d6604d", linestyle="--", linewidth=1)
    ax1.text(chosen + 8, df["coverage_pct"].min(),
             f"chosen: {int(chosen)} m", color="#d6604d", fontsize=9)
    ax1.set_xlabel("bin width [m]")
    ax1.set_ylabel("% of (rock × bin) cells with ≥30 samples")
    ax1.set_title("Coverage")
    ax1.grid(alpha=0.3)
    for _, r in df.iterrows():
        ax1.annotate(f"{r['coverage_pct']:.0f}%",
                     (r["width_m"], r["coverage_pct"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.plot(df["width_m"], df["median_n_populated"],
             marker="o", color="#55A868", linewidth=1.8)
    ax2.axvline(chosen, color="#d6604d", linestyle="--", linewidth=1)
    ax2.axhline(MIN_SAMPLES, color="#888", linestyle=":", linewidth=1)
    ax2.text(WIDTHS_M[0] + 5, MIN_SAMPLES + 5, f"min {MIN_SAMPLES}", color="#888", fontsize=9)
    ax2.set_xlabel("bin width [m]")
    ax2.set_ylabel("median samples per populated cell")
    ax2.set_title("Density (log scale)")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("DistributionBank bin-width tradeoff "
                 "(empty cells fall back to nearest populated bin at sample time)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    print("loading samples.parquet...")
    wide = load_wide()
    print(f"  {len(wide):,} (well, depth) rows in [0, {DEPTH_MAX}] m")

    rows = []
    for w in WIDTHS_M:
        r = coverage_for_width(wide, w)
        rows.append(r)
        print(f"  width={w:>4d} m  bins={r['n_bins']:>3d}  "
              f"coverage={r['coverage_pct']:>5.1f}%  "
              f"median-n={r['median_n_populated']:>7.0f}")

    # Decision rule (v3): pick the smallest width where median samples
    # per populated cell stays >= 100 (KDE bandwidth comfort) AND
    # coverage >= 30% (the rest of the cells fall through the nearest-
    # populated-cell fallback in DistributionBank._resolve_cell).
    df = pd.DataFrame(rows).sort_values("width_m")
    viable = df[(df["median_n_populated"] >= 100) & (df["coverage_pct"] >= 30)]
    if len(viable):
        chosen = float(viable["width_m"].min())
    else:
        chosen = float(df["width_m"].min())
    base = df[df["width_m"] == 400]["coverage_pct"].iloc[0]
    crow = df[df["width_m"] == chosen].iloc[0]
    print(f"\nchosen bin width: {int(chosen)} m  "
          f"(coverage {crow['coverage_pct']:.1f}%, "
          f"median-n {crow['median_n_populated']:.0f}; "
          f"400m baseline {base:.1f}%)")

    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    plot_tradeoff(rows, chosen, OUT_TRADEOFF)
    heatmap_for_width(wide, chosen, OUT_HEATMAP)


if __name__ == "__main__":
    main()
