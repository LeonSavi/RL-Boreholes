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
import matplotlib as mpl

from simulator.distributions import DiscoveryPrior

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
    rocks = prior.rock_types

    # rank rocks by their max probability anywhere — show top-N + Other
    top_idx = np.argsort(p.max(axis=1))[::-1][:8]
    top_idx = sorted(top_idx, key=lambda i: -p[i].sum())
    rest = np.array([i for i in range(len(rocks)) if i not in top_idx])

    palette = mpl.colormaps.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11, 5.2))

    bottom = np.zeros_like(centres, dtype=float)
    for c, i in enumerate(top_idx):
        ax.bar(centres, p[i], width=widths * 0.95, bottom=bottom,
               label=rocks[i], color=palette[c], edgecolor="white", linewidth=0.2)
        bottom += p[i]
    if len(rest):
        rest_p = p[rest].sum(axis=0)
        ax.bar(centres, rest_p, width=widths * 0.95, bottom=bottom,
               label="(other)", color="#bbbbbb",
               edgecolor="white", linewidth=0.2)
        bottom += rest_p

    ax.set_xlim(0, DEPTH_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel("depth [m]")
    ax.set_ylabel("P(rock | depth, hc_discovery=True)")
    ax.set_title(
        f"Discovery prior — {prior.n_positive_wells} positive NLOG wells, "
        f"{prior.n_positive_rows:,} sample rows.  "
        "Tall bands = rocks that host hydrocarbons in that depth range."
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2,
              framealpha=0.9, frameon=True)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
