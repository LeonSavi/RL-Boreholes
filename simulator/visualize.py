"""
Diagnostic plots for inspecting generated maps.

Usage
-----
    from simulator.visualize import plot_map
    from simulator import SimConfig, generate_map, DistributionBank

    bank = DistributionBank.load("data/clean/distributions.pkl")
    m = generate_map(bank, SimConfig())
    plot_map(m, out_path="map_check.png")

The figure has four rows:
  row 1: a vertical cross-section at y = ny//2 showing rock types + key vars
  row 2: three map-view slices (shallow, mid, deep) of a chosen variable
  row 3: orebody ground truth — yield_field, thickness_field, rock mask
  row 4: one representative "drilled" borehole (all variables vs depth)

Use it after fitting the bank and generating a map to sanity-check that
the simulator output looks geologically reasonable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# colours for rock types — chosen to be visually distinct and match the
# intuition people have (chalk=cream, halite=pink, sandstone=tan, etc.)
ROCK_COLOURS = {
    "clay":       "#b3a380",
    "claystone":  "#6b5544",
    "sandstone":  "#d4a85a",
    "chalk":      "#f5f0dc",
    "halite":     "#ff99cc",
    "anhydrite":  "#8a6fb0",
    "carbonate":  "#70c2a8",
    "other":      "#cccccc",
}


def _rock_image(rock_arr: np.ndarray) -> tuple[np.ndarray, list, ListedColormap]:
    """Encode string rock-type array as integer indices + colormap."""
    unique = sorted(set(rock_arr.ravel().tolist()))
    idx = {r: i for i, r in enumerate(unique)}
    int_arr = np.vectorize(idx.get)(rock_arr)
    colours = [ROCK_COLOURS.get(r, "#888888") for r in unique]
    return int_arr, unique, ListedColormap(colours)


def plot_map(
    m: dict[str, Any],
    out_path: str | Path | None = None,
    primary_variable: str = "rhob",
) -> plt.Figure:
    """Render a 4-row diagnostic figure of one generated map."""
    rock_types = m["rock_types"]
    variables = m["variables"]
    depth_axis = m["depth_axis"]
    nx, ny, nz = rock_types.shape

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.38, wspace=0.28)

    # --- row 1: cross-section at y = ny // 2 -----------------------------
    y_slice = ny // 2
    rock_xs = rock_types[:, y_slice, :].T           # (nz, nx)
    int_rocks, uniq_rocks, rock_cmap = _rock_image(rock_xs)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(int_rocks, aspect="auto", cmap=rock_cmap,
              extent=(0, nx, depth_axis[-1], depth_axis[0]),
              interpolation="nearest")
    ax.set_title(f"Rock types (cross-section at y={y_slice})", fontweight="bold")
    ax.set_xlabel("x (cell)"); ax.set_ylabel("depth (m)")

    # legend for rock types (manual, since imshow doesn't do it)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ROCK_COLOURS.get(r, "#888"), label=r)
               for r in uniq_rocks]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)

    # three variables alongside the cross-section
    for col, var in enumerate(["rhob", "gr_api", "dt_us_ft"], start=1):
        if var not in variables:
            continue
        ax = fig.add_subplot(gs[0, col])
        sl = variables[var][:, y_slice, :].T
        im = ax.imshow(sl, aspect="auto", cmap="viridis",
                       extent=(0, nx, depth_axis[-1], depth_axis[0]),
                       interpolation="nearest")
        ax.set_title(f"{var} (cross-section)", fontweight="bold")
        ax.set_xlabel("x (cell)"); ax.set_ylabel("depth (m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- row 2: map-view slices (shallow/mid/deep) -----------------------
    slice_depths = [depth_axis[nz // 6], depth_axis[nz // 2], depth_axis[-nz // 5]]
    slice_idxs = [nz // 6, nz // 2, nz - nz // 5]
    arr = variables.get(primary_variable)
    if arr is not None:
        vmin, vmax = np.nanpercentile(arr, [2, 98])
        for col, (si, sd) in enumerate(zip(slice_idxs, slice_depths)):
            ax = fig.add_subplot(gs[1, col])
            im = ax.imshow(arr[:, :, si].T, origin="lower",
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{primary_variable} @ {sd:.0f}m", fontweight="bold")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # rock-type map-view at mid-depth
        ax = fig.add_subplot(gs[1, 3])
        rock_mid = rock_types[:, :, nz // 2]
        int_rocks_mid, _, rock_cmap_mid = _rock_image(rock_mid)
        ax.imshow(int_rocks_mid.T, origin="lower", cmap=rock_cmap_mid,
                  interpolation="nearest")
        ax.set_title(f"rock_type @ {slice_depths[1]:.0f}m", fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # --- row 3: orebody ---------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(m["yield_field"].T, origin="lower", cmap="hot")
    ax.set_title(f"Ore yield ({len(m['bodies'])} bodies)", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(m["thickness_field"].T, origin="lower", cmap="hot")
    ax.set_title("Ore thickness", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # histogram of yield values
    ax = fig.add_subplot(gs[2, 2])
    yf = m["yield_field"].ravel()
    ax.hist(yf[yf > 0], bins=30, color="#cc4422", alpha=0.8)
    ax.set_xlabel("yield"); ax.set_ylabel("cells")
    ax.set_title("Yield histogram (cells with ore)", fontweight="bold")
    ax.grid(alpha=0.25)

    # histogram of thickness
    ax = fig.add_subplot(gs[2, 3])
    tf = m["thickness_field"].ravel()
    ax.hist(tf[tf > 0], bins=30, color="#2244cc", alpha=0.8)
    ax.set_xlabel("thickness"); ax.set_ylabel("cells")
    ax.set_title("Thickness histogram (cells with ore)", fontweight="bold")
    ax.grid(alpha=0.25)

    # --- row 4: one "drilled" borehole ------------------------------------
    # pick the cell with highest yield so the borehole is interesting
    if m["yield_field"].max() > 0:
        best = np.unravel_index(np.argmax(m["yield_field"]), m["yield_field"].shape)
    else:
        best = (nx // 2, ny // 2)

    var_groups = [
        (["rhob"], "density"),
        (["gr_api"], "gamma"),
        (["dt_us_ft", "nphi"], "dt / nphi"),
        (["pef", "res_deep_log"], "pef / resistivity"),
    ]
    for col, (vars_, title) in enumerate(var_groups):
        ax = fig.add_subplot(gs[3, col])
        for v in vars_:
            if v in variables:
                profile = variables[v][best[0], best[1], :]
                ax.plot(profile, depth_axis, label=v, linewidth=1.2)
        ax.invert_yaxis()
        ax.set_title(f"Borehole ({best[0]},{best[1]}) — {title}",
                     fontweight="bold", fontsize=10)
        ax.set_ylabel("depth (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Simulator diagnostic — {nx}×{ny} map, "
                 f"{nz} depth samples, {len(m['bodies'])} orebodies",
                 fontsize=14, fontweight="bold", y=0.995)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"wrote {out_path}")
    return fig
