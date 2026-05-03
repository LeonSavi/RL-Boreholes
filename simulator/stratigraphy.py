"""
Stratigraphic column — stage 1b.

Delegates layer-geometry sampling to FormationGeometry. In v7 each
layer carries a list of rocks (one per cell at cell_height resolution)
rather than a single rock string. This allows realistic within-formation
facies alternation (e.g., halite/anhydrite/dolomite interbedding within
ZE), driven by a Markov chain in FormationStats.sample_rocks_markov.

Module contents
---------------
* StratigraphicColumn dataclass — represents one realised column
* sample_column(rng, geometry, max_depth, ...) — wrapper around
  FormationGeometry.sample_column
* sample_spatial_column_field — produces a 2D (x, y) field of columns
  with lateral continuity via per-boundary spatial wiggle. Each (x, y)
  column inherits the rock list from a single base column; the boundary
  perturbation only shifts layer top/bot positions, the rock sequence
  is preserved per layer (cells re-indexed under the perturbed range).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .formation_geometry import (
    FormationGeometry,
    DEFAULT_CELL_HEIGHT,
    DEFAULT_FACIES_PERSISTENCE,
)


# Layer tuple shape: (formation, rocks, top, bot)
#   formation : str
#   rocks     : list[str]   one entry per source cell (cell_height = 10m)
#   top, bot  : float       layer top and bottom in metres


@dataclass
class StratigraphicColumn:
    """One realisation of a vertical stratigraphic profile — an ordered
    list of (formation, rocks, top, bot) tuples."""
    layers: list[tuple[str, list[str], float, float]]
    cell_height: float = DEFAULT_CELL_HEIGHT

    def _layer_at(self, depth: float):
        for layer in self.layers:
            _, _, top, bot = layer
            if top <= depth < bot:
                return layer
        return None

    def rock_type_at(self, depth: float) -> str | None:
        layer = self._layer_at(depth)
        if layer is None:
            return None
        _, rocks, top, bot = layer
        if not rocks:
            return None
        # Map this depth to an index inside the layer's rocks list,
        # stretching/compressing if the layer's actual span differs from
        # the span used when the rocks list was generated.
        frac = (depth - top) / max(bot - top, 1e-6)
        idx = int(np.clip(int(frac * len(rocks)), 0, len(rocks) - 1))
        return rocks[idx]

    def formation_at(self, depth: float) -> str | None:
        layer = self._layer_at(depth)
        if layer is None:
            return None
        return layer[0]

    @property
    def total_depth(self) -> float:
        return max(bot for _, _, _, bot in self.layers)

    def rasterise(self, depth_array: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rock_types, formations) arrays aligned to depth_array.

        Each cell in depth_array gets a rock type and formation by
        looking up which layer contains the cell's depth and indexing
        into the layer's rocks list.
        """
        rocks = np.empty(len(depth_array), dtype=object)
        forms = np.empty(len(depth_array), dtype=object)
        for i, d in enumerate(depth_array):
            r = self.rock_type_at(d)
            f = self.formation_at(d)
            rocks[i] = r if r is not None else "other"
            forms[i] = f if f is not None else "other"
        return rocks, forms


def sample_column(
    rng: np.random.Generator,
    geometry: FormationGeometry,
    max_depth: float = 4400.0,
    cell_height: float = DEFAULT_CELL_HEIGHT,
    facies_persistence: float = DEFAULT_FACIES_PERSISTENCE,
) -> StratigraphicColumn:
    """Sample one vertical stratigraphic column from FormationGeometry."""
    layers = geometry.sample_column(
        rng,
        max_depth=max_depth,
        cell_height=cell_height,
        facies_persistence=facies_persistence,
    )
    return StratigraphicColumn(layers=layers, cell_height=cell_height)


def sample_spatial_column_field(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    geometry: FormationGeometry,
    max_depth: float = 4400.0,
    layer_waviness: float = 20.0,
    cell_height: float = DEFAULT_CELL_HEIGHT,
    facies_persistence: float = DEFAULT_FACIES_PERSISTENCE,
) -> list[list[StratigraphicColumn]]:
    """Sample a 2D (x, y) grid of stratigraphic columns with lateral
    continuity.

    Strategy: pick a single base column, then perturb each layer-
    boundary depth spatially so neighbouring (x, y) cells have similar
    but not identical layer depths. The base column's per-cell rock
    list is reused for each (x, y); the rasteriser maps each depth into
    that list, automatically stretching/compressing when the layer's
    perturbed span differs from the base span.
    """
    base = sample_column(
        rng, geometry, max_depth=max_depth,
        cell_height=cell_height, facies_persistence=facies_persistence,
    )
    n_layers = len(base.layers)

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
            for li, (fm, rocks, top, bot) in enumerate(base.layers):
                perturbed_bot = bot + boundary_perturbations[li + 1, x, y]
                perturbed_bot = max(current + 5.0, perturbed_bot)
                # rocks list is preserved; the perturbed [current,
                # perturbed_bot] range will be sampled into that list
                # via the stretch/compress index in rock_type_at().
                new_layers.append((fm, rocks, current, perturbed_bot))
                current = perturbed_bot
            if new_layers and current < max_depth:
                fm, rocks, top, _ = new_layers[-1]
                new_layers[-1] = (fm, rocks, top, max_depth)
            row.append(StratigraphicColumn(
                layers=new_layers, cell_height=cell_height,
            ))
        columns.append(row)
    return columns