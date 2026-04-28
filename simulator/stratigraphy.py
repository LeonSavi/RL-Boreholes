"""
Stratigraphic column — stage 1b.

Delegates layer-geometry sampling to FormationGeometry (data-driven from
samples.parquet). The hand-coded DUTCH_COLUMN has been removed in v6;
formation depth ranges, thicknesses, and facies probabilities are now
fit empirically per formation.

This module retains:
  * StratigraphicColumn dataclass — represents one realised column
  * sample_column(rng, geometry, max_depth) — wrapper around
    FormationGeometry.sample_column
  * sample_spatial_column_field — produces a 2D (x, y) field of columns
    with lateral continuity via per-boundary spatial wiggle

Why the spatial perturbation logic stays here: it operates on the
already-sampled base column and is independent of how the base column
was generated.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .formation_geometry import FormationGeometry


@dataclass
class StratigraphicColumn:
    """One realisation of a vertical stratigraphic profile — an ordered
    list of (formation, rock_type_fine, depth_top, depth_bottom)."""
    layers: list[tuple[str, str, float, float]]

    def rock_type_at(self, depth: float) -> str | None:
        for _, rock, top, bot in self.layers:
            if top <= depth < bot:
                return rock
        return None

    def formation_at(self, depth: float) -> str | None:
        for fm, _, top, bot in self.layers:
            if top <= depth < bot:
                return fm
        return None

    @property
    def total_depth(self) -> float:
        return max(bot for _, _, _, bot in self.layers)

    def rasterise(self, depth_array: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rock_types, formations) arrays aligned to depth_array."""
        rocks = np.empty(len(depth_array), dtype=object)
        forms = np.empty(len(depth_array), dtype=object)
        for i, d in enumerate(depth_array):
            rocks[i] = self.rock_type_at(d) or "other"
            forms[i] = self.formation_at(d) or "other"
        return rocks, forms


def sample_column(
    rng: np.random.Generator,
    geometry: FormationGeometry,
    max_depth: float = 4400.0,
) -> StratigraphicColumn:
    """Sample one vertical stratigraphic column from the empirical
    FormationGeometry.

    Parameters
    ----------
    rng : np.random.Generator
        Random state.
    geometry : FormationGeometry
        Fitted from samples.parquet via FormationGeometry.fit().
    max_depth : float
        Total column depth (metres). Layers are extended/clipped so the
        column covers [0, max_depth] with no gaps.
    """
    layers = geometry.sample_column(rng, max_depth=max_depth)
    return StratigraphicColumn(layers=layers)


def sample_spatial_column_field(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    geometry: FormationGeometry,
    max_depth: float = 4400.0,
    layer_waviness: float = 20.0,
) -> list[list[StratigraphicColumn]]:
    """Sample a 2D (x, y) grid of stratigraphic columns with lateral
    continuity.

    Strategy: pick a single base column from the geometry, then perturb
    each layer-boundary depth spatially so neighbouring (x, y) cells
    have similar but not identical layer depths. Variable values
    inside cells are filled later by map_generator.py via gstools-style
    GRFs; this function is responsible only for layer geometry.

    Returns
    -------
    columns : nested list, columns[x][y] -> StratigraphicColumn
    """
    base = sample_column(rng, geometry, max_depth=max_depth)
    n_layers = len(base.layers)

    # 2D smooth perturbation field per layer boundary: each boundary
    # shifts by up to ±layer_waviness metres across the (x, y) grid.
    boundary_perturbations = np.zeros((n_layers + 1, n_x, n_y))
    for i in range(1, n_layers):
        kx = rng.uniform(0.05, 0.3)
        ky = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        ampl = rng.uniform(0.4, 1.0) * layer_waviness
        xs = np.arange(n_x)[:, None]
        ys = np.arange(n_y)[None, :]
        boundary_perturbations[i] = ampl * np.sin(kx * xs + ky * ys + phase)

    columns = []
    for x in range(n_x):
        row = []
        for y in range(n_y):
            new_layers = []
            current = 0.0
            for li, (fm, rock, top, bot) in enumerate(base.layers):
                perturbed_bot = bot + boundary_perturbations[li + 1, x, y]
                # ensure positive thickness (>= 5m) after perturbation
                perturbed_bot = max(current + 5.0, perturbed_bot)
                new_layers.append((fm, rock, current, perturbed_bot))
                current = perturbed_bot
            # close the bottom to max_depth
            if new_layers and current < max_depth:
                fm, rock, top, _ = new_layers[-1]
                new_layers[-1] = (fm, rock, top, max_depth)
            row.append(StratigraphicColumn(layers=new_layers))
        columns.append(row)
    return columns