"""
Orebody — the target variables the POMDP wants to find.

Currently models hydrocarbon-discovery prospectivity in the Dutch
subsurface, calibrated against NLOG positive-discovery wells (gas, oil,
gas+oil) via the DiscoveryPrior. The framework generalises to other
deposit types given suitable training labels.

Per generated map:
  * 0–3 ellipsoidal "deposit" bodies, each in (x, y, z).
  * Body (cx, cy) is sampled by softmax over candidate columns scored
    against the prior. Columns whose rock signature matches historical
    positive-discovery wells score higher.
  * Body z-centre is sampled within a depth window (default 1600–4400m)
    weighted by the prior's per-cell probability — i.e. ore preferentially
    sits at depths where reservoir / seal / source-rock signatures occur.

Output:
  yield_field : (n_x, n_y, n_z) float32   — yield value at every voxel
  bodies      : list[OreBody]              — ground-truth body parameters

The 2D thickness_field from the previous version is removed; vertical
extent is now encoded via each body's `radius_z`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .distributions import DiscoveryPrior


@dataclass
class OreBody:
    """One ellipsoidal ore body in 3D map coordinates."""
    center_x: float                # cell index, x-axis
    center_y: float                # cell index, y-axis
    center_z: float                # depth in metres
    radius_x: float                # cells
    radius_y: float                # cells
    radius_z: float                # metres (vertical extent)
    peak_yield: float              # grade at the centre
    orientation: float             # radians, rotation in (x, y) plane


def _trivariate_gaussian(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    body: OreBody,
) -> np.ndarray:
    """3D Gaussian falloff from body centre.

    Rotation is applied in the (x, y) plane only — z is independent.
    Z is in metres (use depth_axis), x/y are in cell indices.
    """
    dx = xx - body.center_x
    dy = yy - body.center_y
    dz = zz - body.center_z

    cos_t, sin_t = np.cos(body.orientation), np.sin(body.orientation)
    rot_x = cos_t * dx + sin_t * dy
    rot_y = -sin_t * dx + cos_t * dy

    return np.exp(
        -0.5 * (
            (rot_x / body.radius_x) ** 2
            + (rot_y / body.radius_y) ** 2
            + (dz / body.radius_z) ** 2
        )
    )


def _score_column(
    rock_column: np.ndarray,
    depth_axis: np.ndarray,
    prior: "DiscoveryPrior",
    depth_window: tuple[float, float],
) -> float:
    """Sum of log-probabilities under the prior, restricted to the
    depth window. Higher = column looks more like a positive-discovery
    well in the NLOG corpus."""
    return prior.score_column(
        rock_types_column=rock_column,
        depth_axis=depth_axis,
        depth_window=depth_window,
    )


def _sample_z_in_column(
    rock_column: np.ndarray,
    depth_axis: np.ndarray,
    prior: "DiscoveryPrior",
    depth_window: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """Sample a z-centre within the depth window, weighted by the
    prior's probability of seeing the local rock at each depth.

    Falls back to the depth-window centre if no cell in the window has
    finite log-prob (e.g. column is all 'other')."""
    lo, hi = depth_window
    in_window = (depth_axis >= lo) & (depth_axis < hi)

    weights = np.zeros(len(depth_axis), dtype=np.float64)
    for i, (r, d, m) in enumerate(zip(rock_column, depth_axis, in_window)):
        if not m:
            continue
        lp = prior.log_prob(str(r), float(d))
        if np.isfinite(lp):
            weights[i] = np.exp(lp)

    total = weights.sum()
    if total <= 0:
        return float(0.5 * (lo + hi))
    weights /= total
    chosen_idx = int(rng.choice(len(depth_axis), p=weights))
    return float(depth_axis[chosen_idx])


def sample_orebodies(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    n_depth: int,
    depth_axis: np.ndarray,
    rock_types: np.ndarray,
    prior: "DiscoveryPrior" | None = None,
    n_bodies: int | None = None,
    n_candidates: int = 30,
    ore_depth_window: tuple[float, float] = (1600.0, 4400.0),
    yield_peak_range: tuple[float, float] = (0.5, 5.0),
    radius_xy_range: tuple[float, float] = (3.0, 8.0),
    radius_z_range: tuple[float, float] = (50.0, 200.0),
    softmax_temperature: float = 1.0,
) -> tuple[np.ndarray, list[OreBody]]:
    """Generate a 3D yield field with 0-3 rock-conditioned ore bodies.

    Parameters
    ----------
    rng : np.random.Generator
        Random state.
    n_x, n_y, n_depth : int
        Map grid shape.
    depth_axis : (n_depth,) array
        Depth in metres for each z-index.
    rock_types : (n_x, n_y, n_depth) object array
        Rock-type label per voxel. Used to score candidate columns and
        to sample body z-centres.
    prior : DiscoveryPrior | None
        Calibration prior fitted from NLOG hc_discovery=True wells.
        If None, falls back to uniform random placement (legacy behaviour).
    n_bodies : int | None
        Number of bodies to place. None → sampled uniformly from {0, 1, 2, 3}.
    n_candidates : int
        Number of (x, y) candidate locations sampled per body, scored
        against the prior, and softmax-resolved to the chosen one.
    ore_depth_window : (lo, hi)
        Depth range (metres) within which ore body z-centres can be
        placed. Default 1600-4400m matches the realistic Dutch
        hydrocarbon depth range based on the prior.
    yield_peak_range, radius_xy_range, radius_z_range : (lo, hi)
        Uniform-random ranges for the body's peak yield and ellipsoid axes.
    softmax_temperature : float
        Temperature for the softmax over candidate scores. Higher =
        more uniform sampling; lower = more deterministic toward the
        best-scoring candidate. Default 1.0 (raw softmax).

    Returns
    -------
    yield_field : (n_x, n_y, n_depth) float32
    bodies : list[OreBody] (ground truth, length 0-3)
    """
    if n_bodies is None:
        n_bodies = int(rng.integers(0, 4))  # 0 to 3 bodies

    yield_field = np.zeros((n_x, n_y, n_depth), dtype=np.float32)
    bodies: list[OreBody] = []

    if n_bodies == 0:
        return yield_field, bodies

    # pre-build coordinate grids once (in the right units: x/y in cells,
    # z in metres so radius_z is also in metres)
    xs = np.arange(n_x, dtype=np.float32)[:, None, None]
    ys = np.arange(n_y, dtype=np.float32)[None, :, None]
    zs = depth_axis.astype(np.float32)[None, None, :]
    xx = np.broadcast_to(xs, (n_x, n_y, n_depth))
    yy = np.broadcast_to(ys, (n_x, n_y, n_depth))
    zz = np.broadcast_to(zs, (n_x, n_y, n_depth))

    for _ in range(int(n_bodies)):
        # ---------- pick (cx, cy) via score-weighted softmax ----------
        if prior is not None:
            cand_x = rng.uniform(0.1 * n_x, 0.9 * n_x, size=n_candidates)
            cand_y = rng.uniform(0.1 * n_y, 0.9 * n_y, size=n_candidates)
            scores = np.empty(n_candidates, dtype=np.float64)
            for i, (cx, cy) in enumerate(zip(cand_x, cand_y)):
                ix, iy = int(round(cx)), int(round(cy))
                ix = int(np.clip(ix, 0, n_x - 1))
                iy = int(np.clip(iy, 0, n_y - 1))
                col = rock_types[ix, iy, :]
                scores[i] = _score_column(col, depth_axis, prior,
                                          ore_depth_window)
            # softmax with numerical stability
            s = scores / max(softmax_temperature, 1e-6)
            s = s - s.max()
            probs = np.exp(s)
            probs /= probs.sum()
            chosen = int(rng.choice(n_candidates, p=probs))
            center_x = float(cand_x[chosen])
            center_y = float(cand_y[chosen])
        else:
            # legacy: uniform random
            center_x = float(rng.uniform(0.1 * n_x, 0.9 * n_x))
            center_y = float(rng.uniform(0.1 * n_y, 0.9 * n_y))

        # ---------- pick z-centre using prior weighting ----------
        ix = int(np.clip(round(center_x), 0, n_x - 1))
        iy = int(np.clip(round(center_y), 0, n_y - 1))
        col = rock_types[ix, iy, :]
        if prior is not None:
            center_z = _sample_z_in_column(
                col, depth_axis, prior, ore_depth_window, rng,
            )
        else:
            lo, hi = ore_depth_window
            center_z = float(rng.uniform(lo, hi))

        # ---------- build the body ----------
        body = OreBody(
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            radius_x=float(rng.uniform(*radius_xy_range)),
            radius_y=float(rng.uniform(*radius_xy_range)),
            radius_z=float(rng.uniform(*radius_z_range)),
            peak_yield=float(rng.uniform(*yield_peak_range)),
            orientation=float(rng.uniform(0, np.pi)),
        )
        bodies.append(body)

        falloff = _trivariate_gaussian(xx, yy, zz, body)
        # overlapping bodies: keep the richer one
        yield_field = np.maximum(
            yield_field, (body.peak_yield * falloff).astype(np.float32)
        )

    return yield_field, bodies