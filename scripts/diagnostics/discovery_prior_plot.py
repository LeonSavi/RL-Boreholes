"""
DiscoveryPrior visualisation.

Shows P(rock_type_fine | depth, hc_discovery=True) as a stacked bar
chart over depth. Each bar is a depth bin; segments inside the bar are
rock types sized by their conditional probability.

This makes it visible where in the column the simulator preferentially
places ore bodies (high-probability cells become attractive hosts).

Outputs:
    plots/discovery_prior_bars.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# thesis-legible default fonts
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

from simulator.distributions import DiscoveryPrior
from simulator.visualize import ROCK_COLOURS

FINE_ROCKS = ["anhydrite", "chalk", "clay", "claystone_cool", "claystone_hot",
              "dolomite", "halite_pure", "sandstone_clean", "sandstone_shaly"]

OUT = Path("plots/discovery_prior_bars.png")
DEPTH_MAX = 4400.0


def main() -> None:
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")

    n_bins = prior.prob.shape[1]
    edges = np.asarray(prior.depth_bins)
    keep = edges[:-1] < DEPTH_MAX
    p = prior.prob[:, keep]
    edges = edges[:sum(keep) + 1]
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    rocks = list(prior.rock_types)

    # Keep ONLY the 9 fine rock classes (the reclassification already dropped
    # everything else); renormalise each depth bin so the stack sums to 1.
    # No spurious "(other)" bucket.
    fine_idx = [rocks.index(r) for r in FINE_ROCKS if r in rocks]
    pf = p[fine_idx, :]
    col_sums = pf.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    pf = pf / col_sums
    fine_names = [rocks[i] for i in fine_idx]
    order = np.argsort(pf.sum(axis=1))[::-1]      # tidy stack: biggest first

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bottom = np.zeros_like(centres, dtype=float)
    for j in order:
        name = fine_names[j]
        ax.bar(centres, pf[j], width=widths * 0.95, bottom=bottom,
               label=name, color=ROCK_COLOURS.get(name, "#888888"),
               edgecolor="white", linewidth=0.2)
        bottom += pf[j]

    ax.set_xlim(0, DEPTH_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel("depth [m]")
    ax.set_ylabel(r"P(rock $\mid$ depth, hc$^{+}$)")
    ax.set_title(
        f"Discovery prior: host-rock probability vs depth "
        f"({prior.n_positive_wells} positive NLOG wells)",
        fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              framealpha=0.9, title="rock type", title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
