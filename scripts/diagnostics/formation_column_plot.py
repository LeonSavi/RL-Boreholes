"""
Formation-geometry visual: one base column + a 2D cross-section showing
the GRF-perturbed layer boundaries.

Left panel:  the 1-D base column drawn from FormationGeometry — flat
             horizontal layer boundaries, no spatial perturbation.
Right panel: a 2-D vertical cross-section through a generated map, where
             the same formations have been perturbed laterally by the
             anisotropic Gaussian random field (gstools).

Purpose: show John that the simulator's geometry is not "stack of flat
slabs" but a stack of layers whose boundaries wiggle with a controlled
lateral correlation length.

Outputs:
    plots/formation_column_grf.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from simulator import SimConfig, generate_map, DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.stratigraphy import StratigraphicColumn
from simulator.visualize import ROCK_COLOURS

OUT = Path("plots/formation_column_grf.png")

# 9 fine rock classes only; non-fine residuals (e.g. "other") are
# painted gray and excluded from the legend.
FINE_ROCKS = frozenset([
    "anhydrite", "chalk", "clay",
    "claystone_cool", "claystone_hot",
    "dolomite", "halite_pure",
    "sandstone_clean", "sandstone_shaly",
])
NON_FINE_COLOUR = "#dddddd"


def _rock_image(arr: np.ndarray) -> tuple[np.ndarray, list, ListedColormap]:
    unique = sorted(set(arr.ravel().tolist()))
    idx = {r: i for i, r in enumerate(unique)}
    return (np.vectorize(idx.get)(arr), unique,
            ListedColormap([ROCK_COLOURS.get(r, "#888888") for r in unique]))


def main() -> None:
    bank = DistributionBank.load("data/clean/distributions.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")

    rng = np.random.default_rng(42)

    # --- left: base column (no spatial perturbation) ---------------------
    layers = geom.sample_column(rng)
    column = StratigraphicColumn(layers=layers)
    z_axis = np.linspace(0, 4400, 440, dtype=np.float32)
    base_rocks, _ = column.rasterise(z_axis)

    # --- right: 2-D cross-section through a generated map ----------------
    cfg = SimConfig()
    m = generate_map(bank=bank, geometry=geom, config=cfg, rng=rng, prior=prior)
    rock_field = m["rock_types"]
    y_slice = rock_field.shape[1] // 2
    cross = rock_field[:, y_slice, :].T   # (nz, nx)
    z_axis_m = m["depth_axis"]

    # --- colours: shared across both panels ------------------------------
    # Restrict the legend to the 9 fine rock classes; any non-fine residual
    # (e.g. "other" from the catch-all formation) is painted as a neutral
    # background colour and dropped from the legend.
    observed = set(list(base_rocks) + cross.ravel().tolist())
    all_rocks = sorted(r for r in observed if r in FINE_ROCKS)
    rock_idx = {r: i for i, r in enumerate(all_rocks)}
    palette = [ROCK_COLOURS.get(r, "#888888") for r in all_rocks]
    palette.append(NON_FINE_COLOUR)
    non_fine_idx = len(all_rocks)
    cmap = ListedColormap(palette)

    def _to_idx(arr):
        return np.vectorize(lambda r: rock_idx.get(r, non_fine_idx))(arr)

    base_int = _to_idx(base_rocks)
    cross_int = _to_idx(cross)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5),
                                   gridspec_kw={"width_ratios": [1, 4]})

    axL.imshow(base_int.reshape(-1, 1), aspect="auto", cmap=cmap,
               extent=(0, 1, z_axis[-1], z_axis[0]),
               interpolation="nearest")
    axL.set_xticks([])
    axL.set_ylabel("depth [m]")
    axL.set_title("Base column\n(FormationGeometry +\ncalibrated Markov sampler)",
                  fontsize=10)

    axR.imshow(cross_int, aspect="auto", cmap=cmap,
               extent=(0, rock_field.shape[0] * 0.1,
                       z_axis_m[-1], z_axis_m[0]),
               interpolation="nearest")
    axR.set_xlabel("lateral distance [km]")
    axR.set_title("2-D cross-section through generated map\n"
                  "(layer boundaries perturbed by anisotropic 2-D GRF, "
                  "gstools)",
                  fontsize=10)
    axR.set_yticklabels([])

    handles = [Patch(facecolor=ROCK_COLOURS.get(r, "#888"), label=r)
               for r in all_rocks]
    axR.legend(handles=handles, loc="upper right", fontsize=7,
               framealpha=0.9, ncol=2)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
