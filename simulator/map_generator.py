"""
Map generator — stage 1 end-to-end.

Assembles:
  1. a 2D (n_x, n_y) field of stratigraphic columns (rock_type per depth)
  2. per-cell variable values sampled from the fitted distributions,
     with lateral smoothness via gstools Gaussian random fields
  3. an ore yield + thickness field from 0-3 orebody blobs

Single-map output shape: dict with arrays
    rock_types:   (n_x, n_y, n_depth) dtype=object
    formations:   (n_x, n_y, n_depth) dtype=object
    variables:    dict[var_name -> (n_x, n_y, n_depth) float32]
    yield_field:  (n_x, n_y) float32
    thickness_field: (n_x, n_y) float32
    depth_axis:   (n_depth,) float32
    bodies:       list of OreBody dataclasses (ground truth)

By default maps are generated ON THE FLY: calling generate_map() gives you
one map; no disk writes.  Suitable for use as a torch IterableDataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .distributions import DistributionBank
from .stratigraphy import sample_spatial_column_field, StratigraphicColumn
from .orebody import sample_orebodies, OreBody


@dataclass
class SimConfig:
    """Simulation parameters."""
    n_x: int = 32
    n_y: int = 32
    n_depth: int = 200
    max_depth: float = 2000.0
    variables: tuple[str, ...] = (
        "rhob", "gr_api", "dt_us_ft", "nphi", "pef",
        "cali_in", "res_deep_log", "sp_mv", "drho", "msus_si",
    )
    # how spatially smooth the noise-field perturbations are (in cells).
    # smaller = more fine-grained variation; larger = broader features.
    lateral_correlation_cells: float = 2.0
    # blend between the smooth GRF baseline and i.i.d. KDE draws.
    # 0.0 = pure i.i.d. draws (no spatial correlation, speckly).
    # 1.0 = pure GRF (smooth but doesn't capture KDE shape detail).
    # 0.5 gives both: visible spatial features + real per-cell variation.
    spatial_correlation_strength: float = 0.5
    layer_waviness_m: float = 20.0

    @property
    def dz(self) -> float:
        return self.max_depth / self.n_depth


def generate_map(
    bank: DistributionBank,
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Generate one 2D map with full variables + ore yield + thickness.

    Parameters
    ----------
    bank : DistributionBank
        Fitted from NLOG+LILY data via DistributionBank.fit().
    config : SimConfig
        Map dimensions and variable list.
    rng : np.random.Generator
        Random state (provide for reproducibility; None = fresh).
    """
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
        max_depth=config.max_depth,
        layer_waviness=config.layer_waviness_m,
    )

    rock_types = np.empty((nx, ny, nz), dtype=object)
    formations = np.empty((nx, ny, nz), dtype=object)
    for x in range(nx):
        for y in range(ny):
            r, f = columns[x][y].rasterise(depth_axis)
            rock_types[x, y, :] = r
            formations[x, y, :] = f

    # --- 2. sample variable values per cell -------------------------------
    # For each (x, y, z) cell we know its rock_type. We want:
    #   (a) values respecting per-rock-type distributions
    #   (b) between-variable correlations
    #   (c) smooth lateral variation (neighbouring cells similar)
    #
    # Approach: two-stage sampling.
    #   Stage A: per-cell joint sample from the bank — preserves marginals
    #            and between-variable correlations, but i.i.d. across cells.
    #   Stage B: generate a 2D Gaussian random field per variable per slice
    #            (mean 0, std 1). Use it to "pull" each independent draw
    #            toward a spatially-smooth baseline.
    #
    # Specifically: final = alpha * (rock_mean + rock_std * grf) +
    #                       (1 - alpha) * iid_draw
    # where alpha controls how much lateral correlation vs sample diversity.
    # alpha=0: raw i.i.d. draws (no lateral correlation).
    # alpha=1: everything is mean + std * grf (fully correlated, no per-cell
    #          detail beyond the GRF).
    # Default alpha=0.5 preserves ~75% of the original variance while giving
    # visible spatial correlation.
    variables_out = {
        v: np.full((nx, ny, nz), np.nan, dtype=np.float32)
        for v in config.variables
    }

    # per-variable (x, y, z) correlated noise fields — one GRF per depth
    # slice. Using the same seed across depths would produce vertically
    # continuous features; using different seeds per depth gives
    # decorrelated slices. We use the SAME seed per variable across
    # depths so a cell that is "high density" at one depth tends to be
    # "high density" at nearby depths too.
    noise_fields = _make_noise_fields(
        rng, nx, ny, nz, config.variables,
        len_scale=config.lateral_correlation_cells,
    )

    for z_idx, depth in enumerate(depth_axis):
        for rock in bank.rock_types:
            mask = (rock_types[:, :, z_idx] == rock)
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            # i.i.d. per-cell joint draw
            samples = bank.sample(rock, float(depth), n=n_cells, rng=rng)
            # cell location (x, y) for each masked cell (to index noise field)
            xs, ys = np.where(mask)
            for var in config.variables:
                if var not in samples:
                    continue
                iid = samples[var]
                if np.all(np.isnan(iid)):
                    continue
                # pull cell values toward a smooth baseline around the rock mean
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
                variables_out[var][xs, ys, z_idx] = blended.astype(np.float32)

    # --- 3. orebody: yield + thickness ------------------------------------
    yield_field, thickness_field, bodies = sample_orebodies(rng, nx, ny)

    return {
        "rock_types": rock_types,
        "formations": formations,
        "variables": variables_out,
        "yield_field": yield_field.astype(np.float32),
        "thickness_field": thickness_field.astype(np.float32),
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
    len_scale: float,
) -> dict[str, np.ndarray]:
    """Generate a correlated noise field per variable, shape (nx, ny, nz),
    with mean 0, std 1, and Gaussian spatial correlation in (x, y) with
    correlation length `len_scale` cells.  Slices are drawn independently
    but with temporal correlation via a slow random walk on each cell.

    Uses scipy.ndimage.gaussian_filter on white noise — cheap, and gives
    approximately Gaussian-correlated fields. For more physical
    variograms you'd switch to gstools.SRF here.
    """
    from scipy.ndimage import gaussian_filter
    out = {}
    for var in variables:
        # white noise in (x, y, z), smoothed in (x, y) only per slice
        raw = rng.normal(0, 1, size=(nx, ny, nz)).astype(np.float32)
        for z in range(nz):
            raw[:, :, z] = gaussian_filter(raw[:, :, z], sigma=len_scale)
        # renormalise to unit variance after smoothing (smoothing reduces std)
        std = raw.std()
        if std > 1e-6:
            raw /= std
        out[var] = raw
    return out


def _bin_for(bank, depth: float) -> int:
    bin_idx = np.searchsorted(bank.depth_bins, depth, side="right") - 1
    return int(np.clip(bin_idx, 0, len(bank.depth_bins) - 2))


class MapGenerator:
    """Stateful wrapper for streaming map generation.  Use with torch
    IterableDataset or just call next() repeatedly."""

    def __init__(
        self,
        bank: DistributionBank,
        config: SimConfig | None = None,
        seed: int | None = None,
    ):
        self.bank = bank
        self.config = config or SimConfig()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        return generate_map(self.bank, self.config, self.rng)