"""
A/B test of the gas-response shift.

For each of N test maps: generate it ONCE with the gas-shift table
enabled, and ONCE more with the shift disabled (same seed so the
underlying rock layout, ore bodies, and noise fields match cell-by-
cell). The difference is exactly the gas shift applied per cell.

For three picked maps we render:
  - rocks + ore body outline (overview)
  - rhob: with shift, without shift, delta (shift contribution)
  - res_deep_log: same triple
  - nphi: same triple
plus per-(rock inside body) mean(delta) annotations so you can
verify the shift table values are being applied with the right
direction and magnitude.

Output:
    plots/test_map_<i>.png   for i in PICK
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from simulator import SimConfig, generate_map, DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry

OUT_DIR = Path("plots")
N_MAPS = 10
PICK = (2, 3, 8)
GAS_VARS = ["rhob", "res_deep_log", "nphi"]


def main() -> None:
    bank  = DistributionBank.load("data/clean/distributions.pkl")
    geom  = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    cfg_on  = SimConfig()
    cfg_off = SimConfig(gas_shift_table_path=None)
    variables = cfg_on.variables
    var_idx = {v: i for i, v in enumerate(variables)}

    print(f"max_ore_bodies={cfg_on.max_ore_bodies}, "
          f"gas_shift={cfg_on.gas_shift_table_path}")
    print(f"A/B: same seed per map, shift ON vs shift OFF\n")

    for i in range(N_MAPS):
        seed = 1000 + i
        m_on  = generate_map(bank, geom, cfg_on,
                             rng=np.random.default_rng(seed), prior=prior)
        m_off = generate_map(bank, geom, cfg_off,
                             rng=np.random.default_rng(seed), prior=prior)

        in_ore = m_on["yield_field"] > 0
        n_in_ore = int(in_ore.sum())
        n_bodies = len(m_on.get("bodies", []))

        if i not in PICK:
            print(f"  map {i}: {n_bodies} bodies, {n_in_ore} in-ore cells")
            continue
        if n_in_ore == 0:
            print(f"  map {i}: no ore bodies, skipping figure")
            continue

        rock = m_on["rock_types"]
        depth_axis = m_on["depth_axis"]
        nx, ny, nz = rock.shape

        # per-rock shift report inside the body
        delta_per_rock_per_var = {}
        for v in GAS_VARS:
            d = m_on["variables"][v] - m_off["variables"][v]
            for r in np.unique(rock[in_ore]):
                if r is None: continue
                mask = (rock == r) & in_ore
                if mask.sum() < 30: continue
                delta_per_rock_per_var.setdefault(str(r), {})[v] = float(
                    np.nanmean(d[mask]))

        print(f"\n  map {i}: {n_bodies} body(ies), {n_in_ore} in-ore cells")
        print(f"  applied gas shifts (mean delta in-body):")
        for r, dvs in sorted(delta_per_rock_per_var.items()):
            parts = "  ".join(f"d{v}={dvs.get(v, 0):+.3f}" for v in GAS_VARS)
            print(f"    {r:18s}  {parts}")

        # ------ figure ------
        # pick the y-slice that contains the most in-ore cells so the
        # cross-section actually cuts through the body (a fixed
        # y = ny/2 misses bodies that sit off-centre and the user
        # only sees a thin sliver of red).
        per_y_count = in_ore.sum(axis=(0, 2))
        y_slice = int(np.argmax(per_y_count)) if per_y_count.max() > 0 else ny // 2
        fig, axes = plt.subplots(len(GAS_VARS), 4, figsize=(17, 4 * len(GAS_VARS)),
                                 gridspec_kw={"wspace": 0.30, "hspace": 0.3})

        for row, v in enumerate(GAS_VARS):
            on_section  = m_on["variables"][v][:, y_slice, :].T
            off_section = m_off["variables"][v][:, y_slice, :].T
            delta = on_section - off_section
            ore_section = in_ore[:, y_slice, :].T.astype(float)
            vmin = float(min(np.nanpercentile(off_section, 2),
                             np.nanpercentile(on_section, 2)))
            vmax = float(max(np.nanpercentile(off_section, 98),
                             np.nanpercentile(on_section, 98)))

            for col, (title, sec, cmap, vrange) in enumerate([
                ("rocks + ore" if row == 0 else "", None, None, None),
                (f"{v}  shift ON",  on_section,  "viridis", (vmin, vmax)),
                (f"{v}  shift OFF", off_section, "viridis", (vmin, vmax)),
                (f"delta = ON - OFF", delta, "RdBu_r", None),
            ]):
                ax = axes[row, col]
                if col == 0:
                    if row == 0:
                        rock_slice = rock[:, y_slice, :].T
                        uniq = sorted(set(rock_slice.ravel().tolist()))
                        rmap = {r: k for k, r in enumerate(uniq)}
                        ax.imshow(np.vectorize(rmap.get)(rock_slice),
                                  aspect="auto", cmap="tab20",
                                  extent=(0, nx*0.1, depth_axis[-1], depth_axis[0]),
                                  interpolation="nearest")
                        ax.contour(np.linspace(0, nx*0.1, nx), depth_axis,
                                   ore_section, levels=[0.5],
                                   colors="red", linewidths=1.6)
                        ax.set_title(f"Rocks + ore outline\n"
                                     f"{n_bodies} body(ies), {n_in_ore} cells",
                                     fontsize=10)
                        ax.set_ylabel("depth [m]")
                    else:
                        ax.set_visible(False)
                else:
                    if col == 3:
                        # delta uses centered colormap so 0 is white
                        amax = max(0.01, float(np.nanmax(np.abs(delta))))
                        norm = TwoSlopeNorm(vcenter=0, vmin=-amax, vmax=amax)
                        im = ax.imshow(sec, aspect="auto", cmap=cmap,
                                       norm=norm,
                                       extent=(0, nx*0.1, depth_axis[-1], depth_axis[0]),
                                       interpolation="nearest")
                    else:
                        im = ax.imshow(sec, aspect="auto", cmap=cmap,
                                       vmin=vrange[0], vmax=vrange[1],
                                       extent=(0, nx*0.1, depth_axis[-1], depth_axis[0]),
                                       interpolation="nearest")
                    ax.contour(np.linspace(0, nx*0.1, nx), depth_axis,
                               ore_section, levels=[0.5],
                               colors="red", linewidths=1.4)
                    ax.set_title(title, fontsize=10)
                    if row == len(GAS_VARS) - 1:
                        ax.set_xlabel("x [km]")
                    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        fig.suptitle(f"Test map {i}: gas-response A/B  "
                     f"(left columns: shift on, off, delta = shift contribution; "
                     f"red = inside ore body)",
                     fontsize=11, y=1.00)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"test_map_{i}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
