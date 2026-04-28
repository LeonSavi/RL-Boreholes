"""
Diagnostic plots for inspecting generated maps.

Usage
-----
    from simulator.visualize import plot_map
    from simulator import SimConfig, generate_map, DistributionBank, DiscoveryPrior

    bank  = DistributionBank.load("data/clean/distributions.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    m = generate_map(bank, SimConfig(), prior=prior)
    plot_map(m, out_path="map_check.png")

The figure has four rows:
  row 1: a vertical cross-section at y = ny//2 showing rock types + key vars
  row 2: three map-view slices (shallow, mid, deep) of a chosen variable
  row 3: orebody ground truth — top-down yield max-projection,
         vertical yield cross-section, body z-centres, yield histogram
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


# colours for rock types — chosen to be visually distinct
ROCK_COLOURS = {
    "clay":            "#b3a380",
    "claystone":       "#6b5544",
    "claystone_cool":  "#8a7359",
    "claystone_hot":   "#3d2e1f",
    "sandstone":       "#d4a85a",
    "sandstone_clean": "#e8c474",
    "sandstone_shaly": "#a87f3d",
    "chalk":           "#f5f0dc",
    "halite":          "#ff99cc",
    "halite_pure":     "#ff66aa",
    "anhydrite":       "#8a6fb0",
    "carbonate":       "#70c2a8",
    "dolomite":        "#509070",
    "limestone":       "#a0d8c0",
    "mudstone":        "#7a6650",
    "siltstone":       "#c0b090",
    "basalt":          "#404040",
    "diatom_ooze":     "#dadada",
    "nanno_ooze":      "#e8e8e8",
    "other":           "#cccccc",
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
    yield_field = m["yield_field"]
    bodies = m["bodies"]
    nx, ny, nz = rock_types.shape

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

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

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ROCK_COLOURS.get(r, "#888"), label=r)
               for r in uniq_rocks]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)

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

        ax = fig.add_subplot(gs[1, 3])
        rock_mid = rock_types[:, :, nz // 2]
        int_rocks_mid, _, rock_cmap_mid = _rock_image(rock_mid)
        ax.imshow(int_rocks_mid.T, origin="lower", cmap=rock_cmap_mid,
                  interpolation="nearest")
        ax.set_title(f"rock_type @ {slice_depths[1]:.0f}m", fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # --- row 3: 3D orebody diagnostics ------------------------------------
    # 3a: top-down max projection of yield over depth
    ax = fig.add_subplot(gs[2, 0])
    yield_top = yield_field.max(axis=2)
    im = ax.imshow(yield_top.T, origin="lower", cmap="hot")
    # mark body centres
    for b in bodies:
        ax.scatter(b.center_x, b.center_y, s=40, marker="x",
                   color="cyan", linewidths=1.5)
    ax.set_title(f"Yield max-projection (top-down, "
                 f"{len(bodies)} bodies)", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3b: yield cross-section at y = ny // 2
    ax = fig.add_subplot(gs[2, 1])
    yield_xs = yield_field[:, y_slice, :].T  # (nz, nx)
    im = ax.imshow(yield_xs, aspect="auto", cmap="hot",
                   extent=(0, nx, depth_axis[-1], depth_axis[0]),
                   interpolation="nearest")
    # mark body z-centres if any are near this y slice
    for b in bodies:
        if abs(b.center_y - y_slice) < b.radius_y:
            ax.scatter(b.center_x, b.center_z, s=40, marker="x",
                       color="cyan", linewidths=1.5)
    ax.set_title(f"Yield cross-section (y={y_slice})", fontweight="bold")
    ax.set_xlabel("x (cell)"); ax.set_ylabel("depth (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 3c: histogram of body z-centres
    ax = fig.add_subplot(gs[2, 2])
    if bodies:
        ax.hist([b.center_z for b in bodies], bins=10, color="#cc4422",
                alpha=0.8, edgecolor="white")
        ax.axvline(np.mean([b.center_z for b in bodies]),
                   color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("body z-centre (m)")
    ax.set_ylabel("count")
    ax.set_title("Body depth distribution", fontweight="bold")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)

    # 3d: yield histogram for non-zero voxels
    ax = fig.add_subplot(gs[2, 3])
    yf_flat = yield_field.ravel()
    yf_nonzero = yf_flat[yf_flat > 1e-6]
    if len(yf_nonzero) > 0:
        ax.hist(yf_nonzero, bins=30, color="#cc4422", alpha=0.8,
                edgecolor="white")
    ax.set_xlabel("yield"); ax.set_ylabel("voxels")
    ax.set_title("Yield histogram (voxels > 0)", fontweight="bold")
    ax.grid(alpha=0.25)

    # --- row 4: one "drilled" borehole ------------------------------------
    # pick the (x, y) cell with highest column-integrated yield
    if yield_field.max() > 0:
        col_yield = yield_field.sum(axis=2)
        best = np.unravel_index(np.argmax(col_yield), col_yield.shape)
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
        # overlay the yield profile for this column on a twin axis
        if col == 3:
            twin = ax.twiny()
            twin.plot(yield_field[best[0], best[1], :], depth_axis,
                      color="red", linewidth=1.2, label="yield")
            twin.set_xlabel("yield", color="red")
            twin.tick_params(axis="x", colors="red")
        ax.invert_yaxis()
        ax.set_title(f"Borehole ({best[0]},{best[1]}) — {title}",
                     fontweight="bold", fontsize=10)
        ax.set_ylabel("depth (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Simulator diagnostic — {nx}×{ny} map, "
                 f"{nz} depth samples ({depth_axis[-1]:.0f}m), "
                 f"{len(bodies)} orebodies",
                 fontsize=14, fontweight="bold", y=0.995)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"wrote {out_path}")
    return fig