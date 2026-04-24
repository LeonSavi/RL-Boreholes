"""
Orebody — the target variables the POMDP wants to find.

Two target variables, per the thesis description:
  - ore_yield  (continuous, e.g. grade in g/t or wt% for base metals)
  - thickness (continuous, m)

These are spatially concentrated: a map has 0-3 ore bodies, each an
elliptical blob centred somewhere in the 2D footprint.  Yield decays
smoothly from the centre; thickness is zero outside the ore body.

The orebody also sits inside a "host rock" facies — usually sandstone or
carbonate — which constrains where it can appear geologically.  For now,
any rock type can host; if Charlie gives preferences later we can restrict.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OreBody:
    """One elliptical ore body in map coordinates."""
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    peak_yield: float       # grade at the centre
    peak_thickness: float   # thickness at the centre
    orientation: float      # radians


def _bivariate_gaussian(
    xx: np.ndarray,
    yy: np.ndarray,
    body: OreBody,
) -> np.ndarray:
    """Gaussian falloff from body centre, respecting ellipse + orientation."""
    dx = xx - body.center_x
    dy = yy - body.center_y
    cos_t, sin_t = np.cos(body.orientation), np.sin(body.orientation)
    rot_x = cos_t * dx + sin_t * dy
    rot_y = -sin_t * dx + cos_t * dy
    return np.exp(
        -0.5 * ((rot_x / body.radius_x) ** 2 + (rot_y / body.radius_y) ** 2)
    )


def sample_orebodies(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    n_bodies: int | None = None,
    yield_peak_range: tuple[float, float] = (0.5, 5.0),
    thickness_peak_range: tuple[float, float] = (5.0, 40.0),
    radius_range: tuple[float, float] = (3.0, 8.0),
) -> tuple[np.ndarray, np.ndarray, list[OreBody]]:
    """Generate a 2D field of (yield, thickness) with 0-3 ore bodies.

    Returns
    -------
    yield_field : (n_x, n_y) array of ore yield values
    thickness_field : (n_x, n_y) array of ore thickness values
    bodies : list of OreBody objects (ground truth for evaluation)
    """
    if n_bodies is None:
        n_bodies = rng.integers(0, 4)  # 0 to 3 bodies per map

    xx, yy = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="ij")

    yield_field = np.zeros((n_x, n_y))
    thickness_field = np.zeros((n_x, n_y))
    bodies = []

    for _ in range(n_bodies):
        body = OreBody(
            center_x=rng.uniform(0.1 * n_x, 0.9 * n_x),
            center_y=rng.uniform(0.1 * n_y, 0.9 * n_y),
            radius_x=rng.uniform(*radius_range),
            radius_y=rng.uniform(*radius_range),
            peak_yield=rng.uniform(*yield_peak_range),
            peak_thickness=rng.uniform(*thickness_peak_range),
            orientation=rng.uniform(0, np.pi),
        )
        bodies.append(body)
        falloff = _bivariate_gaussian(xx, yy, body)
        # add to fields; use max so overlapping bodies show the richer one
        yield_field = np.maximum(yield_field, body.peak_yield * falloff)
        thickness_field = np.maximum(thickness_field, body.peak_thickness * falloff)

    return yield_field, thickness_field, bodies
