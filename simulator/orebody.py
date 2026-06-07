"""
Orebody — stratabound bodies with grade heterogeneity, soft halos, and
gap-tolerant host-rock interval walking.

Key design choices:

1. Each body is hosted by a specific rock type chosen from the
   DiscoveryPrior. Reservoir-class rocks at appropriate depths are
   favoured.

2. The body's vertical extent at each (x, y) is the contiguous host-
   rock interval — but small (≤ `max_interbed_cells`) interbedded
   non-host runs within the interval are bridged. This matches how
   geologists describe reservoir intervals: a "sandstone reservoir"
   typically contains thin shaly stringers, but the whole interval is
   one reservoir for production / mapping purposes. Without this
   tolerance, per-cell facies alternation introduced by the Markov
   chain produces visually jarring "wave skipping" in cross-sections.

3. Where the column has no host rock at all near the body's depth
   band, the envelope falls back to a dim halo around the body's
   anchor (z_top, z_bot) at (cx, cy). This avoids hard cliffs across
   formation pinch-outs while keeping the body visually anchored to
   its host lithology.

4. Lateral Gaussian falloff in (x, y) attenuates the body away from
   (cx, cy).

5. Grade heterogeneity from a 3D GRF multiplied with the envelope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .distributions import DiscoveryPrior


@dataclass
class OreBody:
    """One stratabound ore body."""
    center_x: float
    center_y: float
    center_z: float
    host_rock: str
    z_top: float
    z_bot: float
    radius_x: float
    radius_y: float
    peak_yield: float
    orientation: float


# ---------- helpers ---------------------------------------------------------

def _walk_with_gap_tolerance(
    host_mask: np.ndarray,
    start_idx: int,
    direction: int,
    max_interbed_cells: int,
) -> int:
    """Walk along `host_mask` in `direction` (-1 up, +1 down), allowing
    runs of up to `max_interbed_cells` non-host cells before terminating.

    Returns the furthest index reached that contains host rock.
    """
    n = len(host_mask)
    last_host = start_idx
    consecutive_gap = 0
    i = start_idx + direction
    while 0 <= i < n:
        if host_mask[i]:
            last_host = i
            consecutive_gap = 0
        else:
            consecutive_gap += 1
            if consecutive_gap > max_interbed_cells:
                break
        i += direction
    return last_host


def _find_rock_interval(
    rock_column: np.ndarray,
    depth_axis: np.ndarray,
    target_rock: str,
    z_centre: float,
    max_interbed_cells: int = 3,
) -> tuple[float, float] | None:
    """Find the contiguous-with-gaps run of `target_rock` cells around
    z_centre. Up to `max_interbed_cells` non-host cells are bridged
    within the interval.
    """
    z_idx = int(np.argmin(np.abs(depth_axis - z_centre)))
    if str(rock_column[z_idx]) != target_rock:
        found = False
        for offset in range(1, 6):
            for sign in (-1, 1):
                k = z_idx + sign * offset
                if 0 <= k < len(rock_column) and \
                        str(rock_column[k]) == target_rock:
                    z_idx = k
                    found = True
                    break
            if found:
                break
        if not found:
            return None

    host_mask = np.array(
        [str(c) == target_rock for c in rock_column], dtype=bool,
    )

    top_idx = _walk_with_gap_tolerance(
        host_mask, z_idx, direction=-1,
        max_interbed_cells=max_interbed_cells,
    )
    bot_idx = _walk_with_gap_tolerance(
        host_mask, z_idx, direction=+1,
        max_interbed_cells=max_interbed_cells,
    )

    dz = (depth_axis[1] - depth_axis[0]) if len(depth_axis) > 1 else 10.0
    return (float(depth_axis[top_idx] - 0.5 * dz),
            float(depth_axis[bot_idx] + 0.5 * dz))


def _make_grade_grf(
    rng: np.random.Generator,
    nx: int, ny: int, nz: int,
    lateral_scale: float = 4.0,
    vertical_scale: float = 5.0,
) -> np.ndarray:
    """3D Gaussian random field, mapped to ~[0.3, 1.0] for grade
    multiplication."""
    from scipy.ndimage import gaussian_filter
    raw = rng.normal(0, 1, size=(nx, ny, nz)).astype(np.float32)
    raw = gaussian_filter(
        raw, sigma=(lateral_scale, lateral_scale, vertical_scale),
    )
    std = raw.std()
    if std > 1e-6:
        raw = (raw - raw.mean()) / std
    grade = 0.65 + 0.18 * np.tanh(raw / 1.5)
    return grade.astype(np.float32)


def _vertical_envelope_at_xy(
    rock_col: np.ndarray,
    host_mask: np.ndarray,
    in_window: np.ndarray,
    depth_axis: np.ndarray,
    body: OreBody,
    n_depth: int,
    dz: float,
    layer_edge_smoothness_m: float,
    halo_falloff_m: float,
    max_interbed_cells: int,
    halo_attenuation: float,
) -> np.ndarray:
    """Vertical envelope at one (x, y).

    Two regimes:

    1. Column has host_rock cells in the search window → use the local
       gap-tolerant contiguous host-rock run as the envelope's top/bot,
       with smooth `layer_edge_smoothness_m` edges. Bridges short
       interbeds (up to `max_interbed_cells`) so cross-sections don't
       drop to halo intensity for thin shaly stringers within the
       reservoir interval.

    2. Column has no host_rock cells in the search window → the body
       has bled out into adjacent lithology. Use the body's anchor
       (z_top, z_bot) from (cx, cy) as a reference, with a wider
       `halo_falloff_m` decay. Multiply by `halo_attenuation` so the
       halo is dimmer than the on-host-rock body.
    """
    candidates = np.where(host_mask & in_window)[0]
    d = depth_axis

    if len(candidates) > 0:
        # ---- regime 1: stratabound with gap tolerance ----
        centre_idx_global = int(np.argmin(np.abs(d - body.center_z)))
        centre_idx = centre_idx_global
        if not (host_mask[centre_idx] and in_window[centre_idx]):
            centre_idx = int(candidates[
                np.argmin(np.abs(candidates - centre_idx_global))
            ])

        top_idx = _walk_with_gap_tolerance(
            host_mask, centre_idx, direction=-1,
            max_interbed_cells=max_interbed_cells,
        )
        bot_idx = _walk_with_gap_tolerance(
            host_mask, centre_idx, direction=+1,
            max_interbed_cells=max_interbed_cells,
        )

        local_top = float(d[top_idx] - 0.5 * dz)
        local_bot = float(d[bot_idx] + 0.5 * dz)

        smooth = max(layer_edge_smoothness_m, 1e-3)
        top_drop = 0.5 * (1 + np.tanh((d - local_top) / smooth))
        bot_drop = 0.5 * (1 - np.tanh((d - local_bot) / smooth))
        return (top_drop * bot_drop).astype(np.float32)

    # ---- regime 2: halo (no host rock in window) ----
    smooth = max(halo_falloff_m, 1e-3)
    top_drop = 0.5 * (1 + np.tanh((d - body.z_top) / smooth))
    bot_drop = 0.5 * (1 - np.tanh((d - body.z_bot) / smooth))
    halo = halo_attenuation * (top_drop * bot_drop)
    return halo.astype(np.float32)


def _stratabound_envelope(
    n_x: int, n_y: int, n_depth: int,
    depth_axis: np.ndarray,
    rock_types: np.ndarray,
    body: OreBody,
    layer_edge_smoothness_m: float = 3.0,
    halo_falloff_m: float = 15.0,
    max_interbed_cells: int = 3,
    halo_attenuation: float = 0.2,
) -> np.ndarray:
    """Compute the body envelope (0..1) over the full grid."""
    env = np.zeros((n_x, n_y, n_depth), dtype=np.float32)

    xs = np.arange(n_x, dtype=np.float32)
    ys = np.arange(n_y, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    dx = xx - body.center_x
    dy = yy - body.center_y
    cos_t, sin_t = np.cos(body.orientation), np.sin(body.orientation)
    rx = cos_t * dx + sin_t * dy
    ry = -sin_t * dx + cos_t * dy
    lateral = np.exp(-0.5 * (
        (rx / body.radius_x) ** 2 + (ry / body.radius_y) ** 2
    )).astype(np.float32)
    lateral_threshold = 1e-3

    layer_thickness = body.z_bot - body.z_top
    z_search_lo = body.z_top - 0.5 * layer_thickness
    z_search_hi = body.z_bot + 0.5 * layer_thickness
    in_window = (depth_axis >= z_search_lo) & (depth_axis <= z_search_hi)

    dz = float(depth_axis[1] - depth_axis[0]) if len(depth_axis) > 1 else 10.0

    for ix in range(n_x):
        for iy in range(n_y):
            if lateral[ix, iy] < lateral_threshold:
                continue

            col = rock_types[ix, iy, :]
            host_mask = np.array(
                [str(c) == body.host_rock for c in col], dtype=bool,
            )
            vert = _vertical_envelope_at_xy(
                col, host_mask, in_window, depth_axis, body,
                n_depth, dz, layer_edge_smoothness_m, halo_falloff_m,
                max_interbed_cells, halo_attenuation,
            )
            env[ix, iy, :] = lateral[ix, iy] * vert

    return env


def _score_column(
    rock_column: np.ndarray,
    depth_axis: np.ndarray,
    prior: "DiscoveryPrior",
    depth_window: tuple[float, float],
) -> float:
    return prior.score_column(
        rock_types_column=rock_column,
        depth_axis=depth_axis,
        depth_window=depth_window,
    )


def _sample_z_and_rock(
    rock_column: np.ndarray,
    depth_axis: np.ndarray,
    prior: "DiscoveryPrior",
    depth_window: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[float, str] | None:
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
        return None
    weights /= total
    chosen_idx = int(rng.choice(len(depth_axis), p=weights))
    return float(depth_axis[chosen_idx]), str(rock_column[chosen_idx])


# ---------- main entry point ------------------------------------------------

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
    ore_depth_window: tuple[float, float] = (1200.0, 4100.0),
    yield_peak_range: tuple[float, float] = (0.5, 5.0),
    radius_xy_range: tuple[float, float] = (3.0, 8.0),
    radius_z_range: tuple[float, float] = (50.0, 200.0),  # API compat
    softmax_temperature: float = 1.0,
    layer_edge_smoothness_m: float = 3.0,
    halo_falloff_m: float = 15.0,
    halo_attenuation: float = 0.2,
    max_interbed_cells: int = 3,
    grade_lateral_scale: float = 4.0,
    grade_vertical_scale: float = 5.0,
    max_bodies: int = 2,
) -> tuple[np.ndarray, list[OreBody]]:
    """Generate a 3D yield field with 0-3 stratabound ore bodies.

    Defaults are calibrated for conventional-hydrocarbon-reservoir
    geometries:
      * `layer_edge_smoothness_m=3.0` — sharp top/bot contacts (3-5m
        transition zone)
      * `halo_falloff_m=15.0` — modest bleed across pinch-outs/contacts
      * `halo_attenuation=0.2` — halo at 20% of in-host intensity
      * `max_interbed_cells=3` — bridge thin shaly interbeds within
        the reservoir interval (up to 30m at 10m/cell)
    """
    if n_bodies is None:
        # +1 so the upper bound is inclusive (rng.integers high is exclusive)
        n_bodies = int(rng.integers(0, max(1, max_bodies + 1)))

    yield_field = np.zeros((n_x, n_y, n_depth), dtype=np.float32)
    bodies: list[OreBody] = []

    if n_bodies == 0:
        return yield_field, bodies

    grade_grf = _make_grade_grf(
        rng, n_x, n_y, n_depth,
        lateral_scale=grade_lateral_scale,
        vertical_scale=grade_vertical_scale,
    )

    max_attempts_per_body = 5

    for _ in range(int(n_bodies)):
        body_built = False
        for _attempt in range(max_attempts_per_body):
            if prior is not None:
                cand_x = rng.uniform(0.1 * n_x, 0.9 * n_x, size=n_candidates)
                cand_y = rng.uniform(0.1 * n_y, 0.9 * n_y, size=n_candidates)
                scores = np.empty(n_candidates, dtype=np.float64)
                for i, (cx, cy) in enumerate(zip(cand_x, cand_y)):
                    ix = int(np.clip(round(cx), 0, n_x - 1))
                    iy = int(np.clip(round(cy), 0, n_y - 1))
                    scores[i] = _score_column(
                        rock_types[ix, iy, :], depth_axis, prior,
                        ore_depth_window,
                    )
                s = scores / max(softmax_temperature, 1e-6)
                s = s - s.max()
                probs = np.exp(s)
                probs /= probs.sum()
                chosen = int(rng.choice(n_candidates, p=probs))
                center_x = float(cand_x[chosen])
                center_y = float(cand_y[chosen])
            else:
                center_x = float(rng.uniform(0.1 * n_x, 0.9 * n_x))
                center_y = float(rng.uniform(0.1 * n_y, 0.9 * n_y))

            ix = int(np.clip(round(center_x), 0, n_x - 1))
            iy = int(np.clip(round(center_y), 0, n_y - 1))
            col = rock_types[ix, iy, :]

            if prior is not None:
                sampled = _sample_z_and_rock(
                    col, depth_axis, prior, ore_depth_window, rng,
                )
                if sampled is None:
                    continue
                center_z, host_rock = sampled
            else:
                lo, hi = ore_depth_window
                center_z = float(rng.uniform(lo, hi))
                z_idx = int(np.argmin(np.abs(depth_axis - center_z)))
                host_rock = str(col[z_idx])

            interval = _find_rock_interval(
                col, depth_axis, host_rock, center_z,
                max_interbed_cells=max_interbed_cells,
            )
            if interval is None:
                continue
            z_top, z_bot = interval

            body = OreBody(
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                host_rock=host_rock,
                z_top=z_top,
                z_bot=z_bot,
                radius_x=float(rng.uniform(*radius_xy_range)),
                radius_y=float(rng.uniform(*radius_xy_range)),
                peak_yield=float(rng.uniform(*yield_peak_range)),
                orientation=float(rng.uniform(0, np.pi)),
            )
            bodies.append(body)

            envelope = _stratabound_envelope(
                n_x, n_y, n_depth, depth_axis, rock_types, body,
                layer_edge_smoothness_m=layer_edge_smoothness_m,
                halo_falloff_m=halo_falloff_m,
                max_interbed_cells=max_interbed_cells,
                halo_attenuation=halo_attenuation,
            )
            body_yield = body.peak_yield * envelope * grade_grf
            yield_field = np.maximum(yield_field, body_yield)
            body_built = True
            break


        if not body_built:
            continue
 
    return yield_field, bodies