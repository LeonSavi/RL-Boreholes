"""Visualise the k-means basins on the Dutch RD coordinate system (Task 5b).

Two-panel figure saved to plots/validation/basin_distribution.png:
  (left)  scatter of all NLOG wells coloured by basin cluster, with
          k-means centroids marked as black crosses
  (right) histogram of well count by basin, restricted to the 139-well
          combination pool used by FormationGeometry — this is the plot
          that reveals over-representation (e.g. Groningen Platform).

Run:
    python scripts/validate_basins.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.formation_geometry import FormationGeometry


GEOM_PATH = Path("data/clean/formation_geometry.pkl")
OUT_PATH = Path("plots/validation/basin_distribution.png")


def main() -> None:
    geom = FormationGeometry.load(GEOM_PATH)
    if geom.basin_centers is None:
        print("No basin clustering attached (re-run save_distributions.py).")
        return

    df = pd.read_parquet("data/clean/samples.parquet")
    df = df[df["dataset"] == "NLOG"]
    coords = (df.groupby("borehole")[["x_rd", "y_rd"]].mean().dropna())
    coords["basin"] = coords.index.map(geom.basin_labels_per_well)
    coords = coords.dropna(subset=["basin"])
    coords["basin"] = coords["basin"].astype(int)

    pool_wells = set()
    for combos in geom.combinations_by_basin.values():
        for combo, _ in combos:
            pass
    pool_basin_counts = {
        b: sum(c for _, c in combos)
        for b, combos in geom.combinations_by_basin.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw=dict(width_ratios=[1.4, 1.0]))
    cmap = plt.get_cmap("tab10")
    for b in sorted(coords["basin"].unique()):
        sub = coords[coords["basin"] == b]
        axes[0].scatter(sub["x_rd"] / 1e3, sub["y_rd"] / 1e3,
                        s=8, alpha=0.6,
                        color=cmap(b),
                        label=f"basin {b} (n={len(sub)})")
    cx, cy = geom.basin_centers[:, 0] / 1e3, geom.basin_centers[:, 1] / 1e3
    axes[0].scatter(cx, cy, marker="x", color="black", s=80, lw=2,
                    label="centroid")
    axes[0].set_xlabel("x_rd  /  km")
    axes[0].set_ylabel("y_rd  /  km")
    axes[0].set_title(f"NLOG wells by basin "
                      f"({len(coords):,} wells, k={len(geom.basin_centers)})")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_aspect("equal", adjustable="datalim")

    basins = sorted(pool_basin_counts.keys())
    n_pool = [pool_basin_counts[b] for b in basins]
    n_total = [int((coords["basin"] == b).sum()) for b in basins]
    pos = np.arange(len(basins))
    axes[1].bar(pos - 0.18, n_total, width=0.36, color="lightgrey",
                edgecolor="black", label="full corpus")
    axes[1].bar(pos + 0.18, n_pool, width=0.36, color="tab:orange",
                edgecolor="black", label="139-well combination pool")
    for i, (n, m) in enumerate(zip(n_total, n_pool)):
        axes[1].text(pos[i] - 0.18, n, str(n), ha="center", va="bottom",
                     fontsize=8)
        axes[1].text(pos[i] + 0.18, m, str(m), ha="center", va="bottom",
                     fontsize=8)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([f"basin {b}" for b in basins])
    axes[1].set_ylabel("well count")
    axes[1].set_title(f"basin coverage  "
                      f"(pool total = {sum(n_pool)} wells)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"saved -> {OUT_PATH}")
    print()
    print("basin coverage in 139-well combination pool:")
    for b in basins:
        pct_pool = 100 * pool_basin_counts[b] / max(sum(n_pool), 1)
        pct_full = 100 * n_total[basins.index(b)] / max(sum(n_total), 1)
        print(f"  basin {b}: pool={pool_basin_counts[b]:>3d} ({pct_pool:>5.1f}%)   "
              f"full corpus={n_total[basins.index(b)]:>4d} ({pct_full:>5.1f}%)")


if __name__ == "__main__":
    main()
