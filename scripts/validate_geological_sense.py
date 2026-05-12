"""Visual geological sanity check on generated maps.

Renders four panels to plots/validation/geological_sense.png:
  1. Vertical cross-section at one (x, y) — formations & rocks vs depth,
     plus three petrophysical variables.
  2. Side-by-side strips of 16 columns along x at fixed y=16 — lateral
     coherence: layer boundaries should wiggle smoothly, not jump.
  3. Per-formation rock-composition stacked bars: simulator vs full
     NLOG corpus.  Mismatches > a few percent at the formation level
     are flagged.
  4. Variable distributions per rock (rhob, gr_api) — sim should fall
     inside the real envelope.

Run:
    python scripts/validate_geological_sense.py [--seed 0] [--n-maps 4]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry, FORMATION_ORDER
from simulator.map_generator import MapGenerator, SimConfig


OUT_PATH = Path("plots/validation/geological_sense.png")


def _rock_color_map(all_rocks: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {r: cmap(i % 20) for i, r in enumerate(sorted(all_rocks))}


def _formation_color_map(formations: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("Set3")
    return {f: cmap(i % 12) for i, f in enumerate(formations + ["other"])}


def _plot_vertical_strip(ax, col_rocks, col_fms, depth_axis,
                         rock_colors, fm_colors, title=""):
    """Draw two horizontal strips: rocks (top) and formations (bottom)."""
    for i, (r, f, d) in enumerate(zip(col_rocks, col_fms, depth_axis)):
        ax.add_patch(plt.Rectangle((0, d), 0.5, 10,
                                   facecolor=rock_colors.get(r, "white"),
                                   edgecolor="none"))
        ax.add_patch(plt.Rectangle((0.5, d), 0.5, 10,
                                   facecolor=fm_colors.get(f, "white"),
                                   edgecolor="none"))
    ax.set_xlim(0, 1)
    ax.set_ylim(depth_axis.max() + 10, 0)
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(["rock", "fm"], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("depth (m)")


def panel_single_column(fig, gs_pos, m, rock_colors, fm_colors,
                        variables, ix=16, iy=16):
    ax_strip = fig.add_subplot(gs_pos[0, 0])
    depth_axis = m["depth_axis"]
    col_rocks = m["rock_types"][ix, iy, :].astype(str)
    col_fms = m["formations"][ix, iy, :].astype(str)
    _plot_vertical_strip(ax_strip, col_rocks, col_fms, depth_axis,
                         rock_colors, fm_colors,
                         title=f"column ({ix}, {iy})")

    # variables in side-by-side mini-panels
    for j, var in enumerate(variables):
        ax = fig.add_subplot(gs_pos[0, j + 1], sharey=ax_strip)
        vals = m["variables"][var][ix, iy, :]
        ax.plot(vals, depth_axis, lw=0.7, color="tab:blue")
        ax.set_xlabel(var, fontsize=8)
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        ax.tick_params(axis="y", labelleft=False)


def panel_lateral_strips(ax, m, rock_colors, n_strips=16, y_fixed=16):
    """Show n_strips columns at fixed y across x to assess lateral coherence."""
    depth_axis = m["depth_axis"]
    nz = len(depth_axis)
    img = np.empty((nz, n_strips, 4))  # RGBA
    xs = np.linspace(0, m["rock_types"].shape[0] - 1, n_strips).astype(int)
    for k, x in enumerate(xs):
        rocks_col = m["rock_types"][x, y_fixed, :].astype(str)
        for d in range(nz):
            img[d, k, :] = rock_colors.get(rocks_col[d], (1, 1, 1, 1))
    ax.imshow(img, aspect="auto",
              extent=(0, n_strips, depth_axis[-1], depth_axis[0]))
    ax.set_xlabel(f"x cell (at y={y_fixed})")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"lateral coherence: {n_strips} columns at y={y_fixed}")


def panel_formation_composition(ax, sim_maps, real_df, formations):
    """Stacked bars of rock fraction per formation, sim vs real."""
    # collect per-formation rock counts in sim
    sim_counts = {fm: Counter() for fm in formations}
    for m in sim_maps:
        rt = m["rock_types"]
        fm_arr = m["formations"]
        for fm in formations:
            mask = (fm_arr == fm)
            if mask.any():
                sim_counts[fm].update(rt[mask].astype(str).tolist())

    real_counts = {fm: Counter() for fm in formations}
    for fm in formations:
        sub = real_df[real_df["formation"] == fm]
        real_counts[fm].update(sub["rock_type_fine"].astype(str).tolist())

    all_rocks = sorted({r for c in list(sim_counts.values()) + list(real_counts.values())
                        for r in c})
    cmap = plt.get_cmap("tab20")
    rock_colors = {r: cmap(i % 20) for i, r in enumerate(all_rocks)}

    def stack(ax, label_offset, counts, rocks, alpha=1.0):
        positions = np.arange(len(formations))
        bottoms = np.zeros(len(formations))
        for r in rocks:
            heights = np.array([
                counts[fm].get(r, 0) / max(sum(counts[fm].values()), 1)
                for fm in formations
            ])
            ax.bar(positions + label_offset, heights, bottom=bottoms,
                   width=0.4, color=rock_colors[r], alpha=alpha,
                   edgecolor="black", linewidth=0.3)
            bottoms += heights

    stack(ax, -0.21, real_counts, all_rocks)
    stack(ax, 0.21, sim_counts, all_rocks, alpha=0.85)
    ax.set_xticks(np.arange(len(formations)))
    ax.set_xticklabels(formations)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rock fraction")
    ax.set_title("formation composition — left bar: real NLOG, "
                 "right bar: simulator")

    # legend at top
    handles = [plt.Rectangle((0, 0), 1, 1, color=rock_colors[r]) for r in all_rocks]
    ax.legend(handles, all_rocks, fontsize=7, ncol=min(len(all_rocks), 4),
              loc="upper center", bbox_to_anchor=(0.5, -0.10))


def panel_variable_distributions(ax, sim_maps, real_df, var, target_rocks):
    """Overlay sim vs real KDE-style histograms per (rock, var)."""
    # real
    real_vals_by_rock = {}
    sub = real_df[real_df["measurement"] == var]
    for r in target_rocks:
        v = sub[sub["rock_type_fine"] == r]["value"].dropna().to_numpy()
        real_vals_by_rock[r] = v

    # sim
    sim_vals_by_rock = {r: [] for r in target_rocks}
    for m in sim_maps:
        if var not in m["variables"]:
            continue
        rt = m["rock_types"].astype(str)
        arr = m["variables"][var]
        for r in target_rocks:
            mask = rt == r
            if mask.any():
                sim_vals_by_rock[r].extend(arr[mask].ravel().tolist())

    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(target_rocks):
        rv = real_vals_by_rock.get(r, np.array([]))
        sv = np.array(sim_vals_by_rock.get(r, []))
        sv = sv[np.isfinite(sv)]
        if rv.size < 30 and sv.size < 30:
            continue
        all_vals = np.concatenate([rv, sv]) if rv.size and sv.size else (rv if rv.size else sv)
        if all_vals.size == 0:
            continue
        bins = np.linspace(np.percentile(all_vals, 1),
                           np.percentile(all_vals, 99), 40)
        c = cmap(i)
        if rv.size:
            ax.hist(rv, bins=bins, alpha=0.4, color=c, density=True,
                    label=f"real {r}")
        if sv.size:
            ax.hist(sv, bins=bins, alpha=0.4, color=c, density=True,
                    histtype="step", linewidth=1.6, linestyle="--",
                    label=f"sim {r}")
    ax.set_xlabel(var)
    ax.set_ylabel("density")
    ax.set_title(f"{var} marginals (solid=real, dashed=sim)")
    ax.legend(fontsize=7, ncol=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-maps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bank = DistributionBank.load("data/clean/distributions.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    gen = MapGenerator(bank, geom, SimConfig(), seed=args.seed, prior=prior)
    sim_maps = [next(gen) for _ in range(args.n_maps)]

    # quick "other"-fraction telltale
    other_total = sum(int(np.sum(m["formations"] == "other")) for m in sim_maps)
    total_cells = sum(int(m["formations"].size) for m in sim_maps)
    other_frac = other_total / total_cells if total_cells else 0
    print(f"'other' fraction across {args.n_maps} maps: {other_frac:.4f}")

    # build colour maps
    all_rocks = set()
    for m in sim_maps:
        all_rocks.update(np.unique(m["rock_types"]).astype(str).tolist())
    rock_colors = _rock_color_map(sorted(all_rocks))
    fm_colors = _formation_color_map(list(FORMATION_ORDER))

    real_df = pd.read_parquet("data/clean/samples.parquet")
    real_df = real_df[real_df["dataset"] == "NLOG"]
    real_df = real_df.drop_duplicates(["borehole", "depth"], keep="first")

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.2], hspace=0.32)

    # row 1: single column + variable curves
    gs1 = gs[0].subgridspec(1, 4, width_ratios=[0.5, 1, 1, 1])
    panel_single_column(fig, gs1, sim_maps[0], rock_colors, fm_colors,
                        variables=("rhob", "gr_api", "nphi"))

    # row 2: lateral strip
    ax_lateral = fig.add_subplot(gs[1])
    panel_lateral_strips(ax_lateral, sim_maps[0], rock_colors,
                         n_strips=16, y_fixed=16)

    # row 3: composition + variable hist
    gs3 = gs[2].subgridspec(1, 2, width_ratios=[2.0, 1.0])
    ax_comp = fig.add_subplot(gs3[0])
    panel_formation_composition(
        ax_comp, sim_maps, real_df,
        formations=[fm for fm in FORMATION_ORDER if fm in geom.formations],
    )
    ax_dist = fig.add_subplot(gs3[1])
    panel_variable_distributions(
        ax_dist, sim_maps, real_df, var="rhob",
        target_rocks=["sandstone_clean", "claystone_hot", "halite_pure", "chalk"],
    )

    fig.suptitle(
        f"Geological sanity check ({args.n_maps} maps, seed {args.seed})",
        fontsize=14,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=110)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
