"""Compact, thesis-ready figure of one synthetic map that contains an ore
body. Three aligned panels:

  1. rock-type cross-section at the y-slice through the deposit,
  2. ore-body yield cross-section at the same slice (the bright lens),
  3. top-down yield max-projection (the deposit footprint).

The map generator draws 0--2 ore bodies per map, so we search seeds until
we hit one with a clearly-developed body. Writes plots/synthetic_map_ore.png
(plots/map_check_v2.png, the full 4-row diagnostic, is left untouched).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Poster-legible default font sizes (axis ticks, labels, colorbars).
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
})

from simulator import SimConfig, generate_map, DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.visualize import ROCK_COLOURS, _rock_image

OUT = Path("plots/synthetic_map_ore.png")
MIN_YIELD_VOXELS = 250   # avoid tiny / edge bodies; pick a clear deposit


def main() -> None:
    bank = DistributionBank.load("data/clean/distributions.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    cfg = SimConfig()

    def _best_column(yf, da):
        """Strongest borehole (by peak yield) and its deposit core
        (cells > 50% of the column's peak yield)."""
        peak = yf.max(axis=2)
        bx, by = np.unravel_index(int(np.argmax(peak)), peak.shape)
        col = yf[bx, by, :]
        core = (np.where(col > 0.5 * col.max())[0] if col.max() > 0
                else np.array([], dtype=int))
        thick = (da[core[-1]] - da[core[0]]) if core.size else 0.0
        return int(bx), int(by), core, thick

    # Prefer a map whose strongest borehole hits a vertically *compact*
    # deposit, so the well-log shows a clear, localised ore anomaly rather
    # than a column saturated by overlapping bodies + haloes.
    chosen = None
    for seed in range(300):
        m = generate_map(bank, geom, cfg, rng=np.random.default_rng(seed),
                          prior=prior)
        yf = m["yield_field"]
        if yf.max() <= 0:
            continue
        bx, by, core, thick = _best_column(yf, m["depth_axis"])
        if core.size and 100.0 <= thick <= 800.0:
            chosen = (seed, m, bx, by, core)
            break
    if chosen is None:                       # fallback: any ore-bearing map
        for seed in range(300):
            m = generate_map(bank, geom, cfg, rng=np.random.default_rng(seed),
                              prior=prior)
            if m["yield_field"].max() > 0:
                bx, by, core, _ = _best_column(m["yield_field"],
                                               m["depth_axis"])
                chosen = (seed, m, bx, by, core)
                break
    if chosen is None:
        raise RuntimeError("no ore-bearing map found")

    seed, m, bx, by, core = chosen
    rock_types = m["rock_types"]
    yield_field = m["yield_field"]
    depth_axis = m["depth_axis"]
    nx, ny, nz = rock_types.shape
    y_slice = by                       # cross-section through the borehole
    d_lo = depth_axis[core[0]] if core.size else None
    d_hi = depth_axis[core[-1]] if core.size else None
    print(f"seed={seed}  bodies={len(m['bodies'])}  borehole=({bx},{by})  "
          f"y_slice={y_slice}  ore_core="
          f"{('%.0f-%.0f m' % (d_lo, d_hi)) if core.size else 'n/a'}")

    variables = m["variables"]
    extent = (0, nx, depth_axis[-1], depth_axis[0])   # depth on y, deep down
    # Canonical channel labels (underscores are literal outside $...$).
    VARIABLE_LABELS = {
        "rhob": "rhob (g/cc)",
        "gr_api": "gr_api (API)",
        "dt_us_ft": r"dt_us_ft ($\mu$s/ft)",
        "nphi": "nphi (v/v)",
        "res_deep_log": r"res_deep_log ($\log_{10}\,\Omega$m)",
    }
    FINE_ROCKS = frozenset([
        "anhydrite", "chalk", "clay", "claystone_cool", "claystone_hot",
        "dolomite", "halite_pure", "sandstone_clean", "sandstone_shaly"])

    # Poster: a compact 1x4 storyboard (rock section, one gas-sensitive log,
    # ore body, top-down footprint) instead of the dense 2x4 montage. The
    # rock-type colour key is a horizontal strip beneath the panels, so the
    # four data panels share the full figure width.
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 4.3),
                             constrained_layout=True)

    def _xsec(ax, sl, title, cmap="viridis", clip=True):
        kw = {}
        if clip:
            lo, hi = np.nanpercentile(sl, [1, 99])
            if hi > lo:
                kw = {"vmin": lo, "vmax": hi}
        im = ax.imshow(sl, aspect="auto", cmap=cmap, extent=extent, **kw)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("x (cell)", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9)
        return im

    # (0) rock-type cross-section
    rock_xs = rock_types[:, y_slice, :].T
    int_rocks, uniq, rcmap = _rock_image(rock_xs)
    axes[0].imshow(int_rocks, aspect="auto", cmap=rcmap, extent=extent,
                   interpolation="nearest")
    axes[0].set_title("Rock types", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("depth (m)", fontsize=11)
    axes[0].set_xlabel("x (cell)", fontsize=11)
    axes[0].tick_params(labelsize=9)

    # (1) one gas-sensitive wireline channel
    _xsec(axes[1], variables["res_deep_log"][:, y_slice, :].T,
          VARIABLE_LABELS["res_deep_log"])

    # (2) ore-body yield cross-section (same slice)
    _xsec(axes[2], yield_field[:, y_slice, :].T, "ore-body yield",
          cmap="hot", clip=False)

    # (3) top-down yield footprint
    yield_top = yield_field.max(axis=2).T
    im = axes[3].imshow(yield_top, origin="lower", cmap="hot",
                        extent=(0, nx, 0, ny))
    axes[3].axhline(y_slice, color="cyan", lw=1.4, ls="--")
    axes[3].set_title("yield footprint (top-down)", fontweight="bold",
                      fontsize=12)
    axes[3].set_xlabel("x (cell)", fontsize=11)
    axes[3].set_ylabel("y (cell)", fontsize=11)
    axes[3].tick_params(labelsize=9)
    cb = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)

    # rock-type colour key as a horizontal strip beneath the four panels.
    leg_handles = [Patch(facecolor=ROCK_COLOURS.get(r, "#888"), label=r)
                   for r in uniq if r in FINE_ROCKS]
    fig.legend(handles=leg_handles, loc="outside lower center",
               ncol=min(len(leg_handles), 5), fontsize=12,
               title="rock type", title_fontsize=13, framealpha=0.0,
               handlelength=1.6, columnspacing=1.6, handletextpad=0.6)
    fig.suptitle("One synthetic map at the ore slice:\n"
                 "rock section, a gas-sensitive log, the ore body, and its footprint",
                 fontweight="bold", fontsize=18, x=0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    # ---- well-log montage of one borehole drilled through the ore -------
    # bx, by, core, d_lo, d_hi already chosen above (compact deposit). Shade
    # only the deposit core so the band marks where the ore actually is.
    yld = yield_field[bx, by, :]
    occ = core

    tracks = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
    fig2, axs = plt.subplots(1, 6, figsize=(8.5, 4.4), sharey=True,
                             constrained_layout=True)
    for ax, v in zip(axs, tracks):
        ax.plot(variables[v][bx, by, :], depth_axis, lw=1.3, color="#1f3b66")
        if occ.size:
            ax.axhspan(d_lo, d_hi, color="red", alpha=0.12)
        ax.set_title(v, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    axs[5].plot(yld, depth_axis, lw=1.3, color="#b00")
    if occ.size:
        axs[5].axhspan(d_lo, d_hi, color="red", alpha=0.12)
    axs[5].set_title("yield", fontsize=10, fontweight="bold")
    axs[5].tick_params(labelsize=8)
    axs[5].grid(alpha=0.25)
    axs[0].invert_yaxis()                        # shared y -> deep at bottom
    axs[0].set_ylabel("depth (m)", fontsize=11)
    fig2.suptitle("Example synthetic borehole through the ore body "
                  "(red band = ore)", fontweight="bold", fontsize=12)
    OUT_LOG = Path("plots/synthetic_borehole.png")
    fig2.savefig(OUT_LOG, dpi=400, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {OUT_LOG}")


if __name__ == "__main__":
    main()
