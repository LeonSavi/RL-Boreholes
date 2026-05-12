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

import gstools as gs
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


def sample_layer_boundary_grf(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    n_boundaries: int,
    perturbation_std: float,
    range_cells: float | tuple[float, float],
) -> np.ndarray:
    """Sample independent 2D Gaussian random fields, one per layer boundary.

    Standard geostatistical practice for spatially correlated boundary
    wiggle: a GRF with a Gaussian variogram of given variance and
    lateral correlation length.  Returns a (n_boundaries, n_x, n_y)
    array of perturbations in metres.

    Parameters
    ----------
    range_cells :
        Lateral correlation length, in cells.  Pass a scalar for an
        isotropic field, or (rx, ry) for an anisotropic one.
    """
    if np.isscalar(range_cells):
        len_scale = float(range_cells)
    else:
        len_scale = [float(r) for r in range_cells]
    model = gs.Gaussian(dim=2, var=perturbation_std ** 2, len_scale=len_scale)
    xs = np.arange(n_x)
    ys = np.arange(n_y)
    out = np.zeros((n_boundaries, n_x, n_y), dtype=np.float64)
    for i in range(n_boundaries):
        # independent realisation per boundary; derive deterministic seed
        # from the outer RNG so the whole sample stream stays reproducible.
        seed = int(rng.integers(0, 2 ** 31 - 1))
        srf = gs.SRF(model, seed=seed)
        out[i] = srf((xs, ys), mesh_type="structured")
    return out


def sample_spatial_column_field(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    geometry: FormationGeometry,
    max_depth: float = 4400.0,
    layer_perturbation_std: float = 10.0,
    layer_perturbation_range_cells: float | tuple[float, float] = 8.0,
    cell_height: float = DEFAULT_CELL_HEIGHT,
    facies_persistence: float = DEFAULT_FACIES_PERSISTENCE,
) -> list[list[StratigraphicColumn]]:
    """Sample a 2D (x, y) grid of stratigraphic columns with lateral
    continuity.

    Strategy: pick a single base column, then perturb each layer-
    boundary depth spatially so neighbouring (x, y) cells have similar
    but not identical layer depths.  The perturbation field is a 2D
    Gaussian random field (anisotropic-capable) per boundary — the
    geostatistical standard — replacing earlier sinusoidal wiggle.
    The base column's per-cell rock list is reused for each (x, y);
    the rasteriser maps each depth into that list, automatically
    stretching/compressing when the layer's perturbed span differs
    from the base span.
    """
    base = sample_column(
        rng, geometry, max_depth=max_depth,
        cell_height=cell_height, facies_persistence=facies_persistence,
    )
    n_layers = len(base.layers)

    # boundary 0 = top of column (always 0), boundary i (1..n_layers-1) =
    # between layer i-1 and layer i, boundary n_layers = bottom (capped to
    # max_depth).  Only the interior boundaries get perturbed.
    boundary_perturbations = np.zeros((n_layers + 1, n_x, n_y))
    n_interior = max(0, n_layers - 1)
    if n_interior > 0:
        interior = sample_layer_boundary_grf(
            rng=rng,
            n_x=n_x, n_y=n_y,
            n_boundaries=n_interior,
            perturbation_std=layer_perturbation_std,
            range_cells=layer_perturbation_range_cells,
        )
        boundary_perturbations[1:n_layers] = interior

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
                # extend the deepest formation to fill the column — real
                # geology continues to the basement even when well logs
                # stop.  The calibrated transition matrix keeps run
                # lengths realistic over this longer stretch.
                fm, rocks, top, _ = new_layers[-1]
                new_layers[-1] = (fm, rocks, top, max_depth)
            row.append(StratigraphicColumn(
                layers=new_layers, cell_height=cell_height,
            ))
        columns.append(row)
    return columns