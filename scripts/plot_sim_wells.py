"""
Plot simulator columns with formation bands highlighted, in the same
style as plot_real_wells.py so you can compare side by side.

Picks a few cells from each generated map and produces one PNG per
column showing the 6 petrophysical variables with formation bands
shaded behind the curves.

Usage:
    python plot_sim_wells.py                        # 5 columns
    python plot_sim_wells.py --n 3 --highlight ZE
    python plot_sim_wells.py --highlight RO
    python plot_sim_wells.py --cell 8 8             # specific (x,y)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from simulator import (
    DistributionBank, DiscoveryPrior, FormationGeometry,
    MapGenerator, SimConfig,
)


VARS = ["rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log"]
VAR_LABELS = {
    "rhob": "RHOB (g/cc)",
    "gr_api": "GR (API)",
    "dt_us_ft": "DT (us/ft)",
    "nphi": "NPHI",
    "pef": "PEF",
    "res_deep_log": "RES (log10)",
}

VAR_LIMS = {
    "rhob": (1.5, 3.2),
    "gr_api": (0.0, 200.0),
    "dt_us_ft": (40.0, 200.0),
    "nphi": (-0.05, 0.6),
    "pef": (0.0, 8.0),
    "res_deep_log": (-1.0, 4.0),
}

FM_COLOR = {
    "NU": "#f5f5f5",
    "NM": "#ebebeb",
    "NL": "#dcdcdc",
    "CK": "#fff5e1",
    "KN": "#e6ddc8",
    "SL": "#d4c8a8",
    "SG": "#c8baa0",
    "AT": "#bda88c",
    "RN": "#b09578",
    "RB": "#a08560",
    "ZE": "#9bcfb8",
    "RO": "#cc9978",
    "DC": "#9c7050",
}


def _formation_intervals(
    formations: np.ndarray, depths: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Contiguous-run encoding of a formation column."""
    if len(formations) == 0:
        return []
    intervals = []
    cur_fm = str(formations[0])
    cur_top = float(depths[0])
    for i in range(1, len(formations)):
        fm = str(formations[i])
        if fm != cur_fm:
            intervals.append((cur_fm, cur_top, float(depths[i - 1])))
            cur_fm = fm
            cur_top = float(depths[i])
    intervals.append((cur_fm, cur_top, float(depths[-1])))
    return intervals


def _plot_column(
    formations: np.ndarray,
    depths: np.ndarray,
    var_arrays: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    highlight: str = "ZE",
) -> None:
    n_vars = len(VARS)
    fig, axes = plt.subplots(
        1, n_vars,
        figsize=(2.6 * n_vars + 2.0, 11),
        sharey=True,
    )

    intervals = _formation_intervals(formations, depths)
    fms_present = sorted({fm for fm, _, _ in intervals if fm})

    d_min = float(depths.min())
    d_max = float(depths.max())

    for ax_i, var in enumerate(VARS):
        ax = axes[ax_i]

        for fm, top, bot in intervals:
            if not fm:
                continue
            color = FM_COLOR.get(fm, "#dddddd")
            alpha = 0.95 if fm == highlight else 0.55
            ax.axhspan(top, bot, facecolor=color, alpha=alpha,
                       edgecolor="none", zorder=0)

        if var == "res_deep_log":
            vals = np.log10(np.abs(var_arrays[var]) + 1e-3)
        else:
            vals = var_arrays[var]

        lo, hi = VAR_LIMS[var]
        vals_clipped = np.clip(vals, lo, hi)
        ax.plot(vals_clipped, depths, color="black",
                linewidth=0.6, zorder=2)

        for fm, top, bot in intervals:
            if fm == highlight:
                ax.axhline(top, color="darkgreen", linewidth=1.0,
                            alpha=0.7, zorder=3)
                ax.axhline(bot, color="darkgreen", linewidth=1.0,
                            alpha=0.7, zorder=3)

        ax.set_xlim(lo, hi)
        ax.set_xlabel(VAR_LABELS[var])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        if ax_i == 0:
            ax.set_ylabel("Depth (m)")

    axes[0].set_ylim(d_max, d_min)

    legend_handles = [
        Patch(
            facecolor=FM_COLOR.get(fm, "#dddddd"),
            alpha=0.95 if fm == highlight else 0.55,
            edgecolor="black" if fm == highlight else "none",
            linewidth=1.5 if fm == highlight else 0,
            label=fm + (" (highlighted)" if fm == highlight else ""),
        )
        for fm in fms_present
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=True,
        title="Formation",
    )

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _extract_var_arrays(
    sim_map: dict, ix: int, iy: int,
) -> dict[str, np.ndarray]:
    """Pull the 6 variable curves from a sim map at cell (ix, iy).

    The simulator stores variables in a dict keyed by variable name
    with arrays of shape (nx, ny, n_depth). Common alternatives are
    handled too.
    """
    out = {}

    # Try several layouts the simulator might use
    if "variables" in sim_map and isinstance(sim_map["variables"], dict):
        vd = sim_map["variables"]
        for v in VARS:
            if v in vd:
                arr = np.asarray(vd[v])
                if arr.ndim == 3:
                    out[v] = arr[ix, iy, :]
                elif arr.ndim == 1:
                    out[v] = arr
                else:
                    out[v] = np.full(sim_map["depth_axis"].shape, np.nan)
            else:
                out[v] = np.full(sim_map["depth_axis"].shape, np.nan)
        return out

    # Alternative: top-level keys per variable
    for v in VARS:
        if v in sim_map:
            arr = np.asarray(sim_map[v])
            if arr.ndim == 3:
                out[v] = arr[ix, iy, :]
            elif arr.ndim == 1:
                out[v] = arr
            else:
                out[v] = np.full(sim_map["depth_axis"].shape, np.nan)
        else:
            out[v] = np.full(sim_map["depth_axis"].shape, np.nan)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5,
                   help="number of simulator columns to plot")
    p.add_argument("--cell", type=int, nargs=2, default=[16, 16],
                   metavar=("IX", "IY"),
                   help="(x,y) cell to extract from each map")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--highlight", default="ZE")
    p.add_argument("--require_formation", default=None,
                   help="only plot columns containing this formation; "
                        "defaults to --highlight")
    p.add_argument("--out_dir", default="plots/sim_wells")
    args = p.parse_args()

    require = args.require_formation or args.highlight

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading simulator components...")
    bank = DistributionBank.load("data/clean/distributions.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")

    n_maps_to_generate = max(args.n * 4, 20)
    print(f"  generating up to {n_maps_to_generate} maps "
          f"to find {args.n} containing {require}...")
    gen = MapGenerator(
        bank, geom, SimConfig(), seed=args.seed, prior=prior,
    )

    ix, iy = args.cell
    n_plotted = 0
    for map_idx in range(n_maps_to_generate):
        if n_plotted >= args.n:
            break
        m = next(gen)
        formations_arr = m["formations"]
        depth_axis = m["depth_axis"]
        col_fm = formations_arr[ix, iy, :]
        if require not in set(col_fm.astype(str).tolist()):
            continue

        var_arrays = _extract_var_arrays(m, ix, iy)
        title = (f"Simulator column — map {map_idx}, cell ({ix},{iy})    "
                 f"highlight={args.highlight}")
        out_path = out_dir / f"sim_map{map_idx:02d}_cell{ix}_{iy}.png"
        _plot_column(
            col_fm, depth_axis, var_arrays, title, out_path,
            highlight=args.highlight,
        )
        n_plotted += 1

    if n_plotted == 0:
        print(f"  no maps contained {require} at cell ({ix},{iy})")
    else:
        print(f"\ndone — {n_plotted} plots in {out_dir}/")


if __name__ == "__main__":
    main()