"""
Map generator — stage 1 end-to-end.

Assembles:
  1. a 2D (n_x, n_y) field of stratigraphic columns (rock_type per depth),
     sampled from FormationGeometry (data-driven empirical distributions)
     with per-cell Markov-chain facies inside each formation
  2. per-cell variable values sampled from DistributionBank, with lateral
     and vertical smoothness via 3D Gaussian random fields
  3. a 3D ore-yield field with 0-3 ellipsoidal bodies, placed by
     rock-conditioned scoring against the DiscoveryPrior

Single-map output shape: dict with arrays
    rock_types:   (n_x, n_y, n_depth) dtype=object
    formations:   (n_x, n_y, n_depth) dtype=object
    variables:    dict[var_name -> (n_x, n_y, n_depth) float32]
    yield_field:  (n_x, n_y, n_depth) float32
    depth_axis:   (n_depth,) float32
    bodies:       list of OreBody dataclasses (ground truth)

Maps are generated ON THE FLY: calling generate_map() gives you one map;
no disk writes. Suitable for use as a torch IterableDataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .distributions import DistributionBank, DiscoveryPrior, HARD_BOUNDS
from .formation_geometry import FormationGeometry
from .stratigraphy import sample_spatial_column_field, StratigraphicColumn
from .orebody import sample_orebodies, OreBody


@dataclass
class SimConfig:
    """Simulation parameters."""
    n_x: int = 32
    n_y: int = 32
    n_depth: int = 440          # 10 m per cell × 4400 m
    max_depth: float = 4400.0   # captures 90% of NLOG positive wells

    variables: tuple[str, ...] = (
        "rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log",
    )

    lateral_correlation_cells: float = 2.0
    vertical_correlation_cells: float = 3.0  # 30m vertical correlation length
    spatial_correlation_strength: float = 0.7 # smootheness across layers
    layer_waviness_m: float = 20.0

    # ----- within-formation facies alternation -----
    # Markov-chain persistence for per-cell rock type within a layer.
    # 0.85 → average run length ~6.7 cells (~67m), matching typical
    # bed-package thickness in Dutch sequences.
    facies_persistence: float = 0.85

    # ----- ore-body parameters -----
    ore_depth_window: tuple[float, float] = (1600.0, 4400.0)
    n_ore_candidates: int = 30
    ore_softmax_temperature: float = 1.0
    ore_radius_xy_range: tuple[float, float] = (3.0, 8.0)
    ore_radius_z_range: tuple[float, float] = (50.0, 200.0)
    ore_yield_peak_range: tuple[float, float] = (0.5, 5.0)

    @property
    def dz(self) -> float:
        return self.max_depth / self.n_depth


def generate_map(
    bank: DistributionBank,
    geometry: FormationGeometry,
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
    prior: DiscoveryPrior | None = None,
) -> dict[str, Any]:
    """Generate one map with full variables + 3D ore yield field."""
    if config is None:
        config = SimConfig()
    if rng is None:
        rng = np.random.default_rng()

    nx, ny, nz = config.n_x, config.n_y, config.n_depth
    depth_axis = np.linspace(0, config.max_depth, nz, dtype=np.float32)

    # --- 1. sample spatial stratigraphic columns --------------------------
    columns = sample_spatial_column_field(
        rng=rng,
        n_x=nx,
        n_y=ny,
        geometry=geometry,
        max_depth=config.max_depth,
        layer_waviness=config.layer_waviness_m,
        cell_height=config.dz,
        facies_persistence=config.facies_persistence,
    )

    rock_types = np.empty((nx, ny, nz), dtype=object)
    formations = np.empty((nx, ny, nz), dtype=object)
    for x in range(nx):
        for y in range(ny):
            r, f = columns[x][y].rasterise(depth_axis)
            rock_types[x, y, :] = r
            formations[x, y, :] = f

    # --- 2. sample variable values per cell -------------------------------
    variables_out = {
        v: np.full((nx, ny, nz), np.nan, dtype=np.float32)
        for v in config.variables
    }

    noise_fields = _make_noise_fields(
        rng, nx, ny, nz, config.variables,
        lateral_len_scale=config.lateral_correlation_cells,
        vertical_len_scale=config.vertical_correlation_cells,
    )

    for z_idx, depth in enumerate(depth_axis):
        for rock in bank.rock_types:
            mask = (rock_types[:, :, z_idx] == rock)
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            samples = bank.sample(rock, float(depth), n=n_cells, rng=rng)
            xs, ys = np.where(mask)
            for var in config.variables:
                if var not in samples:
                    continue
                iid = samples[var]
                if np.all(np.isnan(iid)):
                    continue
                cell = bank.cells.get((rock, _bin_for(bank, depth)))
                if cell is None or var not in cell.kdes:
                    variables_out[var][xs, ys, z_idx] = iid.astype(np.float32)
                    continue
                mu = cell.means.get(var, float(np.nanmean(iid)))
                sigma = cell.stds.get(var, float(np.nanstd(iid)))
                grf_values = noise_fields[var][xs, ys, z_idx]
                smooth_baseline = mu + sigma * grf_values
                alpha = config.spatial_correlation_strength
                blended = alpha * smooth_baseline + (1 - alpha) * iid
                lo, hi = HARD_BOUNDS.get(var, (-np.inf, np.inf))
                blended = np.clip(blended, lo, hi)
                variables_out[var][xs, ys, z_idx] = blended.astype(np.float32)

    # --- 3. 3D ore yield field --------------------------------------------
    yield_field, bodies = sample_orebodies(
        rng=rng,
        n_x=nx,
        n_y=ny,
        n_depth=nz,
        depth_axis=depth_axis,
        rock_types=rock_types,
        prior=prior,
        ore_depth_window=config.ore_depth_window,
        n_candidates=config.n_ore_candidates,
        softmax_temperature=config.ore_softmax_temperature,
        radius_xy_range=config.ore_radius_xy_range,
        radius_z_range=config.ore_radius_z_range,
        yield_peak_range=config.ore_yield_peak_range,
    )

    return {
        "rock_types": rock_types,
        "formations": formations,
        "variables": variables_out,
        "yield_field": yield_field,
        "depth_axis": depth_axis,
        "bodies": bodies,
        "config": asdict(config),
    }


def _make_noise_fields(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    nz: int,
    variables: tuple[str, ...],
    lateral_len_scale: float,
    vertical_len_scale: float,
) -> dict[str, np.ndarray]:
    """Generate per-variable 3D anisotropic Gaussian random fields."""
    from scipy.ndimage import gaussian_filter
    out = {}
    for var in variables:
        raw = rng.normal(0, 1, size=(nx, ny, nz)).astype(np.float32)
        raw = gaussian_filter(
            raw,
            sigma=(lateral_len_scale, lateral_len_scale, vertical_len_scale),
        )
        std = raw.std()
        if std > 1e-6:
            raw /= std
        out[var] = raw
    return out


def _bin_for(bank, depth: float) -> int:
    bin_idx = np.searchsorted(bank.depth_bins, depth, side="right") - 1
    return int(np.clip(bin_idx, 0, len(bank.depth_bins) - 2))


class MapGenerator:
    """Stateful wrapper for streaming map generation."""

    def __init__(
        self,
        bank: DistributionBank,
        geometry: FormationGeometry,
        config: SimConfig | None = None,
        seed: int | None = None,
        prior: DiscoveryPrior | None = None,
    ):
        self.bank = bank
        self.geometry = geometry
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.prior = prior

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        return generate_map(
            self.bank, self.geometry, self.config, self.rng,
            prior=self.prior,
        )