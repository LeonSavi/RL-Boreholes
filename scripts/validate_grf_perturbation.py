"""Validate the Gaussian-random-field layer-boundary perturbation (Task 1).

Renders one realisation of the GRF used to wiggle a layer boundary across
the (x, y) grid and saves a heatmap to plots/validation/.  The output
should look like smooth random hills with the configured correlation
length — not a sine-wave pattern as in the previous implementation.

Run:
    python scripts/validate_grf_perturbation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulator.map_generator import SimConfig
from simulator.stratigraphy import sample_layer_boundary_grf


OUT_DIR = Path("plots/validation")


def main() -> None:
    cfg = SimConfig()
    rng = np.random.default_rng(42)

    fields = sample_layer_boundary_grf(
        rng=rng,
        n_x=cfg.n_x,
        n_y=cfg.n_y,
        n_boundaries=4,
        perturbation_std=cfg.layer_perturbation_std,
        range_cells=cfg.layer_perturbation_range_cells,
    )

    print(f"Generated {len(fields)} independent boundary perturbation fields")
    print(f"  shape:    {fields[0].shape}")
    print(f"  config:   std={cfg.layer_perturbation_std} m  "
          f"range={cfg.layer_perturbation_range_cells} cells")
    for i, f in enumerate(fields):
        print(f"  field {i}: mean={f.mean():+.2f} m  std={f.std():.2f} m  "
              f"min={f.min():+.2f}  max={f.max():+.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    vmax = float(np.max(np.abs(fields)))
    for i, ax in enumerate(axes):
        im = ax.imshow(fields[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower")
        ax.set_title(f"boundary {i} perturbation (m)")
        ax.set_xlabel("y cell")
        ax.set_ylabel("x cell")
    fig.colorbar(im, ax=axes, shrink=0.85, label="boundary shift (m)")
    fig.suptitle(
        f"GRF layer-boundary perturbations  "
        f"(std={cfg.layer_perturbation_std} m, range={cfg.layer_perturbation_range_cells} cells)"
    )
    out_path = OUT_DIR / "grf_perturbation_example.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
