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

import gstools as gs
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
    n_depth: int = 440  # 10 m per cell × 4400 m
    max_depth: float = 4400.0  # captures 90% of NLOG positive wells

    variables: tuple[str, ...] = (
        # pef removed: only 6% of NLOG wells log it, so the encoder spent
        # most of its capacity on a near-constant zero channel.
        "rhob",
        "gr_api",
        "dt_us_ft",
        "nphi",
        "res_deep_log",
    )

    lateral_correlation_cells: float = 2.0
    vertical_correlation_cells: float = 3.0  # 30m vertical correlation length
    spatial_correlation_strength: float = 0.7  # smootheness across layers

    # ----- variable-noise GRF (Task 4) -----
    # Two implementations of an anisotropic 3D Gaussian random field for
    # the per-variable spatial smoothness:
    #   "gaussian_filter": scipy gaussian_filter on white noise.  This IS
    #     spectral synthesis for the Gaussian variogram (the kernel and
    #     covariance function are Fourier duals), so it produces the same
    #     statistical field as gstools, ~15× faster.
    #   "gstools_srf": gstools.SRF with an explicit gs.Gaussian variogram.
    #     Slower; useful if you ever need to swap in exponential/spherical
    #     variograms or do conditional simulation.
    # The validation script confirms both produce variograms matching the
    # specified range (plots/validation/variogram_check.png).
    grf_method: str = "gaussian_filter"
    grf_mode_no: int = 64  # only used when grf_method=="gstools_srf"

    # ----- layer-boundary perturbation (Task 1) -----
    # Each interior layer boundary gets one independent 2D Gaussian random
    # field realisation (anisotropic-capable) added to its base depth.
    # Std controls vertical amplitude in metres; range_cells controls the
    # lateral correlation length (scalar -> isotropic, tuple -> anisotropic).
    layer_perturbation_std: float = 10.0  # m, replaces layer_waviness
    layer_perturbation_range_cells: float = 8.0  # lateral correlation length

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
    # max_ore_bodies caps the per-map body count drawn uniformly from
    # [0, max_ore_bodies]. User constraint: 2.
    max_ore_bodies: int = 2

    # ----- gas-response petrophysics (Phase O) -----
    # Path to the per-rock empirical shift table produced by
    # scripts/diagnostics/fit_gas_shift_table.py. When non-None the
    # map generator looks up shifts per rock_type and adds them to
    # rhob / nphi / res_deep_log inside ore-body cells (yield > 0).
    gas_shift_table_path: str | None = "data/clean/gas_shift_table.json"

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
        layer_perturbation_std=config.layer_perturbation_std,
        layer_perturbation_range_cells=config.layer_perturbation_range_cells,
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

    # --- 2. ore yield field (moved BEFORE variables, Phase O) ------------
    # Ore bodies are placed first so the variable-sampling loop knows
    # which cells should receive the gas-response shift.
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
        max_bodies=config.max_ore_bodies,
    )
    in_orebody = yield_field > 0

    # --- gas-response shift table (Phase O) ------------------------------
    # {rock_type: {variable: shift}}; "default" is the cross-rock mean
    # for rocks we have no entry for.
    gas_shifts: dict = {}
    if config.gas_shift_table_path:
        try:
            import json as _json

            with open(config.gas_shift_table_path) as _fh:
                gas_shifts = _json.load(_fh)
        except (OSError, ValueError) as _exc:
            print(
                f"[map_generator] could not load gas shift table "
                f"{config.gas_shift_table_path}: {_exc!r}; "
                "ore-body cells will NOT receive a gas shift."
            )

    # --- 3. sample variable values per cell ------------------------------
    variables_out = {
        v: np.full((nx, ny, nz), np.nan, dtype=np.float32) for v in config.variables
    }

    noise_fields = _make_noise_fields(
        rng,
        nx,
        ny,
        nz,
        config.variables,
        lateral_len_scale=config.lateral_correlation_cells,
        vertical_len_scale=config.vertical_correlation_cells,
        method=config.grf_method,
        mode_no=config.grf_mode_no,
    )

    for z_idx, depth in enumerate(depth_axis):
        # iterate over unique (rock, formation) pairs at this depth slice
        rock_slice = rock_types[:, :, z_idx]
        fm_slice = formations[:, :, z_idx]
        ore_slice = in_orebody[:, :, z_idx]
        pairs = set(zip(rock_slice.ravel().tolist(), fm_slice.ravel().tolist()))
        for rock, formation in pairs:
            if rock is None or formation is None:
                continue
            mask = (rock_slice == rock) & (fm_slice == formation)
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            samples = bank.sample(
                rock,
                float(depth),
                n=n_cells,
                rng=rng,
                formation=formation,
            )
            xs, ys = np.where(mask)
            cell_in_ore = ore_slice[xs, ys]
            rock_shifts = gas_shifts.get(rock) or gas_shifts.get("default") or {}
            for var in config.variables:
                if var not in samples:
                    continue
                iid = samples[var]
                if np.all(np.isnan(iid)):
                    continue
                cell = bank.cell_for(rock, float(depth), formation=formation)
                if cell is None or var not in cell.kdes:
                    blended = iid.astype(np.float32)
                else:
                    mu = cell.means.get(var, float(np.nanmean(iid)))
                    sigma = cell.stds.get(var, float(np.nanstd(iid)))
                    grf_values = noise_fields[var][xs, ys, z_idx]
                    smooth_baseline = mu + sigma * grf_values
                    alpha = config.spatial_correlation_strength
                    blended = alpha * smooth_baseline + (1 - alpha) * iid
                # gas-response shift in ore-body cells
                shift = float(rock_shifts.get(var, 0.0))
                if shift != 0.0 and cell_in_ore.any():
                    blended = np.where(cell_in_ore, blended + shift, blended)
                lo, hi = bank.bounds_for(var)
                blended = np.clip(blended, lo, hi)
                variables_out[var][xs, ys, z_idx] = blended.astype(np.float32)

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
    method: str = "gaussian_filter",
    mode_no: int = 64,
) -> dict[str, np.ndarray]:
    """Per-variable 3D anisotropic Gaussian random fields.

    For the Gaussian variogram, convolution of white noise with a Gaussian
    kernel is spectral synthesis of the corresponding random field — so
    `method="gaussian_filter"` and `method="gstools_srf"` produce
    statistically equivalent output (verified in validate_grf_3d.py).
    The filter path is ~15× faster.

    Outputs are renormalised to unit std so downstream blending math
    (`mu + sigma * field`) keeps its existing scale.
    """
    if method == "gaussian_filter":
        return _noise_gaussian_filter(
            rng,
            nx,
            ny,
            nz,
            variables,
            lateral_len_scale,
            vertical_len_scale,
        )
    elif method == "gstools_srf":
        return _noise_gstools_srf(
            rng,
            nx,
            ny,
            nz,
            variables,
            lateral_len_scale,
            vertical_len_scale,
            mode_no=mode_no,
        )
    else:
        raise ValueError(
            f"unknown grf_method={method!r} "
            "(expected 'gaussian_filter' or 'gstools_srf')"
        )


def _noise_gaussian_filter(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    nz: int,
    variables: tuple[str, ...],
    lateral_len_scale: float,
    vertical_len_scale: float,
) -> dict[str, np.ndarray]:
    """Convolution of white noise with a Gaussian kernel.

    Math note: convolving white noise with a Gaussian of std σ produces
    a field whose covariance function is a Gaussian with std σ√2, i.e.
    a Gaussian variogram with effective range ≈ σ√2.  To make the
    `*_len_scale` parameter mean the SAME range as gstools' Gaussian
    variogram (γ(h)=σ²[1−exp(−(h/L)²)]), divide the filter sigma by √2.
    The validation script confirms both methods then produce variograms
    with the same fitted range.
    """
    from scipy.ndimage import gaussian_filter

    sigma_lat = lateral_len_scale / np.sqrt(2.0)
    sigma_vert = vertical_len_scale / np.sqrt(2.0)
    out: dict[str, np.ndarray] = {}
    for var in variables:
        raw = rng.normal(0, 1, size=(nx, ny, nz)).astype(np.float32)
        raw = gaussian_filter(
            raw,
            sigma=(sigma_lat, sigma_lat, sigma_vert),
        )
        std = raw.std()
        if std > 1e-6:
            raw /= std
        out[var] = raw
    return out


def _noise_gstools_srf(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    nz: int,
    variables: tuple[str, ...],
    lateral_len_scale: float,
    vertical_len_scale: float,
    mode_no: int = 64,
) -> dict[str, np.ndarray]:
    """gstools.SRF with explicit anisotropic Gaussian variogram.  Slow;
    use only for validation or when swapping in non-Gaussian variograms.
    """
    model = gs.Gaussian(
        dim=3,
        var=1.0,
        len_scale=[lateral_len_scale, lateral_len_scale, vertical_len_scale],
    )
    xs = np.arange(nx)
    ys = np.arange(ny)
    zs = np.arange(nz)
    out: dict[str, np.ndarray] = {}
    for var in variables:
        seed = int(rng.integers(0, 2**31 - 1))
        srf = gs.SRF(model, seed=seed, mode_no=mode_no)
        field = srf((xs, ys, zs), mesh_type="structured").astype(np.float32)
        std = field.std()
        if std > 1e-6:
            field /= std
        out[var] = field
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
            self.bank,
            self.geometry,
            self.config,
            self.rng,
            prior=self.prior,
        )


# ---------------------------------------------------------------------------
# Formation-resolution variant
# ---------------------------------------------------------------------------


def generate_map_formation(
    bank,  # FormationDistributionBank
    geometry: FormationGeometry,
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
    prior: DiscoveryPrior | None = None,
) -> dict[str, Any]:
    """Same as `generate_map` but draws variable values from a formation-
    indexed `FormationDistributionBank` instead of a rock-indexed
    `DistributionBank`.

    Step 1 (stratigraphic columns) and step 3 (ore bodies) are identical
    to `generate_map`. Step 2 (per-cell variable sampling) iterates over
    formations and reads `formations[:, :, z_idx]` instead of
    `rock_types[:, :, z_idx]`. Rock labels are still produced by step 1
    and saved alongside formation labels so the dataset stays directly
    comparable to the rock-resolution one.
    """
    if config is None:
        config = SimConfig()
    if rng is None:
        rng = np.random.default_rng()

    nx, ny, nz = config.n_x, config.n_y, config.n_depth
    depth_axis = np.linspace(0, config.max_depth, nz, dtype=np.float32)

    # --- 1. sample spatial stratigraphic columns (unchanged) --------------
    columns = sample_spatial_column_field(
        rng=rng,
        n_x=nx,
        n_y=ny,
        geometry=geometry,
        max_depth=config.max_depth,
        layer_perturbation_std=config.layer_perturbation_std,
        layer_perturbation_range_cells=config.layer_perturbation_range_cells,
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

    # --- 2. sample variable values per cell, indexed by FORMATION ---------
    variables_out = {
        v: np.full((nx, ny, nz), np.nan, dtype=np.float32) for v in config.variables
    }

    noise_fields = _make_noise_fields(
        rng,
        nx,
        ny,
        nz,
        config.variables,
        lateral_len_scale=config.lateral_correlation_cells,
        vertical_len_scale=config.vertical_correlation_cells,
        method=config.grf_method,
        mode_no=config.grf_mode_no,
    )

    for z_idx, depth in enumerate(depth_axis):
        for fm in bank.formations:
            mask = formations[:, :, z_idx] == fm
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            samples = bank.sample(fm, float(depth), n=n_cells, rng=rng)
            xs, ys = np.where(mask)
            for var in config.variables:
                if var not in samples:
                    continue
                iid = samples[var]
                if np.all(np.isnan(iid)):
                    continue
                cell = bank.cells.get((fm, _bin_for(bank, depth)))
                if cell is None or var not in cell.kdes:
                    variables_out[var][xs, ys, z_idx] = iid.astype(np.float32)
                    continue
                mu = cell.means.get(var, float(np.nanmean(iid)))
                sigma = cell.stds.get(var, float(np.nanstd(iid)))
                grf_values = noise_fields[var][xs, ys, z_idx]
                smooth_baseline = mu + sigma * grf_values
                alpha = config.spatial_correlation_strength
                blended = alpha * smooth_baseline + (1 - alpha) * iid
                lo, hi = bank.bounds_for(var)
                blended = np.clip(blended, lo, hi)
                variables_out[var][xs, ys, z_idx] = blended.astype(np.float32)

    # --- 3. 3D ore yield field (unchanged, uses rock_types) ---------------
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


class FormationMapGenerator:
    """Stateful wrapper for formation-resolution map streaming."""

    def __init__(
        self,
        bank,  # FormationDistributionBank
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
        return generate_map_formation(
            self.bank,
            self.geometry,
            self.config,
            self.rng,
            prior=self.prior,
        )
