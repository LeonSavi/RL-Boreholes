"""
Formation geometry — depth-based sampling with per-column facies
composition drawn from real NLOG wells, blended with the population
average to ensure every column has at least small probability of
showing the formation's typical interbedded facies.

Two-tier fit
------------
1. COMBINATIONS are fit from a depth-windowed subset of NLOG wells
   (default [4000, 4500m]).

2. PER-FORMATION TOP-DEPTH and FACIES-BY-WELL distributions are fit
   from the FULL NLOG corpus. For each formation we store:
   - top_depths, thicknesses (one per well that contains the formation)
   - facies (population-average composition: rock fraction over all cells)
   - well_compositions: list of per-well facies dicts, one per well

Sampling
--------
1. Pick a combination weighted by well count.
2. Sample each formation's top depth from its empirical KDE.
3. For each formation: pick ONE real well's composition uniformly at
   random and BLEND it with the population average using a small
   weight (default `composition_blend = 0.10`):
       mixed = (1 - eps) * well_comp + eps * population_facies
   This keeps the picked well's flavour dominant while ensuring every
   column has a small probability of producing the formation's typical
   interbedded facies. Without this blend, wells with 100%-one-facies
   composition (which exist in NLOG, particularly for narrow ZE
   intervals that crossed only one rock package) would lock the
   simulator's column to a single facies.
4. Run the Markov chain with the mixed composition.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


FORMATION_ORDER = [
    "NU", "NM", "NL",
    "CK", "KN",
    "SL", "SG", "AT",
    "RN", "RB",
    "ZE", "RO",
    "DC",
]

SURFACE_FORMATIONS = {"NU", "NM", "NL", "CK", "KN"}

DEFAULT_MIN_WELL_DEPTH = 4000.0
DEFAULT_MAX_WELL_DEPTH = 4500.0

MIN_LAYER_THICKNESS_DEFAULT = 50.0
MAX_RESAMPLES = 20

DEFAULT_CELL_HEIGHT = 10.0
DEFAULT_FACIES_PERSISTENCE = 0.95
DEFAULT_COMPOSITION_BLEND = 0.10  # weight on population average
MIN_WELLS_FOR_PER_WELL_COMPOSITION = 10  # below this, use population avg only

# Markov transition matrix fit (Task 2 — T-PROGS-style)
TRANSITION_LAPLACE = 1e-3            # smoothing on rare pairs (avoid P=0)
MIN_TRANSITIONS_FOR_FIT = 500        # below this, fall back to persistence
TRANSITION_BIN_STEP_M = 10.0         # resample wells to this step before fit

# Basin-aware stratification (Task 5b)
MIN_WELLS_PER_BASIN_FOR_STRATIFICATION = 5  # tiny basins are dropped — a
# basin with 1-2 wells in the pool would otherwise be picked 1/k of the
# time under uniform basin sampling, which over-amplifies it instead of
# correcting for the over-representation of dense basins.


@dataclass
class FormationStats:
    name: str
    n_wells: int
    top_depths: np.ndarray
    thicknesses: np.ndarray
    facies: dict[str, float] = field(default_factory=dict)
    well_compositions: list[dict[str, float]] = field(default_factory=list)
    # Task 2: empirical 1-step transition matrix at TRANSITION_BIN_STEP_M
    # (rows = from-rock, cols = to-rock, Laplace-smoothed and row-normalised).
    # None when fewer than MIN_TRANSITIONS_FOR_FIT observed transitions —
    # the sampler then falls back to DEFAULT_FACIES_PERSISTENCE.
    transition_matrix: pd.DataFrame | None = None
    n_transitions: int = 0
    _top_kde: gaussian_kde | None = None
    # cached arrays for fast sampling (rebuilt on demand)
    _trans_rocks: list[str] | None = None
    _trans_cumP: np.ndarray | None = None

    def sample_top_depth(self, rng: np.random.Generator) -> float:
        if self._top_kde is None:
            self._top_kde = gaussian_kde(self.top_depths)
        v = float(self._top_kde.resample(1, seed=rng)[0, 0])
        return max(v, 0.0)

    def sample_thickness(self, rng: np.random.Generator) -> float:
        """Sample a thickness from the empirical distribution.

        Used by sample_column for the LAST formation in a combination,
        which would otherwise extend all the way to max_depth and
        produce unrealistic chain run lengths (DC issue).
        """
        if len(self.thicknesses) == 0:
            return 100.0
        # avoid heavy-tail extreme draws by sampling with replacement from
        # the empirical pool — simpler and more robust than fitting a KDE
        # on a positive-skewed distribution where a KDE would put mass
        # below zero.
        return float(rng.choice(self.thicknesses))

    def sample_well_composition(
        self,
        rng: np.random.Generator,
        blend_with_population: float = DEFAULT_COMPOSITION_BLEND,
    ) -> dict[str, float]:
        """Pick one real well's facies composition and blend it with
        the population average.

        `blend_with_population` is the weight on the population average:
        0 means pure well composition, 1 means pure population average.
        Default 0.10 keeps the picked well's flavour dominant while
        ensuring every facies has at least small probability of being
        sampled, so that 100%-one-facies wells don't lock the column.

        Small-pool guard (Task 11): when fewer than
        MIN_WELLS_FOR_PER_WELL_COMPOSITION wells are available, the
        per-well draw becomes a coin flip on a tiny pool — one extreme
        well can dominate.  In that regime we ignore the per-well
        compositions and use the population marginal directly.
        """
        if (not self.well_compositions
                or len(self.well_compositions) < MIN_WELLS_FOR_PER_WELL_COMPOSITION):
            return dict(self.facies) if self.facies else {"other": 1.0}

        idx = int(rng.integers(0, len(self.well_compositions)))
        well_comp = self.well_compositions[idx]

        # union of rocks in well + population
        all_rocks = set(well_comp.keys()) | set(self.facies.keys())
        eps = float(blend_with_population)
        mixed = {}
        for r in all_rocks:
            w = well_comp.get(r, 0.0)
            p = self.facies.get(r, 0.0)
            mixed[r] = (1.0 - eps) * w + eps * p

        # renormalise (in case the population dict and well dict had
        # slightly different rock sets)
        total = sum(mixed.values())
        if total > 0:
            mixed = {r: v / total for r, v in mixed.items()}
        return mixed

    def sample_rocks_markov(
        self,
        rng: np.random.Generator,
        n_cells: int,
        persistence: float = DEFAULT_FACIES_PERSISTENCE,
        facies_override: dict[str, float] | None = None,
    ) -> list[str]:
        if n_cells <= 0:
            return []

        facies = (facies_override if facies_override is not None
                  else self.facies)
        if not facies:
            return ["other"] * n_cells

        rocks, probs = zip(*facies.items())
        rocks = list(rocks)
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()

        out: list[str] = []
        current = str(rng.choice(rocks, p=probs))
        out.append(current)
        for _ in range(1, n_cells):
            if rng.random() < persistence:
                out.append(current)
            else:
                current = str(rng.choice(rocks, p=probs))
                out.append(current)
        return out

    def has_transition_matrix(self) -> bool:
        return (self.transition_matrix is not None
                and self.n_transitions >= MIN_TRANSITIONS_FOR_FIT)

    def _ensure_transition_arrays(self) -> None:
        """Cache numpy views for fast inverse-CDF sampling."""
        if self._trans_rocks is not None and self._trans_cumP is not None:
            return
        if self.transition_matrix is None:
            return
        self._trans_rocks = list(self.transition_matrix.index)
        P = self.transition_matrix.values.astype(np.float64)
        self._trans_cumP = np.cumsum(P, axis=1)

    def sample_rocks_with_matrix(
        self,
        rng: np.random.Generator,
        n_cells: int,
    ) -> list[str]:
        """T-PROGS-style sampler driven by the fitted transition matrix.

        Initial state is drawn from the empirical marginal (`facies`);
        each subsequent cell uses inverse-CDF sampling on the row of
        the transition matrix corresponding to the current state.
        Falls back to sample_rocks_markov() if no matrix is attached.
        """
        if n_cells <= 0:
            return []
        if not self.has_transition_matrix():
            return self.sample_rocks_markov(rng, n_cells)
        self._ensure_transition_arrays()
        rocks = self._trans_rocks
        cumP = self._trans_cumP
        R = len(rocks)

        # initial state from the formation's empirical marginal (restricted
        # to rocks in the matrix; uniform if facies missing or all zero).
        init_p = np.array([self.facies.get(r, 0.0) for r in rocks],
                          dtype=np.float64)
        s = init_p.sum()
        if s > 0:
            init_p /= s
        else:
            init_p = np.full(R, 1.0 / R)

        out_idx = np.empty(n_cells, dtype=np.int64)
        out_idx[0] = int(rng.choice(R, p=init_p))
        u = rng.random(n_cells - 1) if n_cells > 1 else np.empty(0)
        for k in range(1, n_cells):
            out_idx[k] = int(np.searchsorted(cumP[out_idx[k - 1]], u[k - 1]))
            # clip in case of floating-point overshoot at u==1.0
            if out_idx[k] >= R:
                out_idx[k] = R - 1
        return [rocks[i] for i in out_idx]

    def mean_run_lengths(self) -> dict[str, float]:
        """Expected run length per rock under the fitted transition matrix.

        For a Markov chain with self-transition P(i→i)=p_ii, the run
        length of state i is geometric with mean 1/(1 - p_ii).
        """
        if self.transition_matrix is None:
            return {}
        diag = np.diag(self.transition_matrix.values.astype(np.float64))
        denom = np.clip(1.0 - diag, 1e-9, None)
        run = 1.0 / denom
        return dict(zip(self.transition_matrix.index, run.tolist()))

    @property
    def median_top(self) -> float:
        return float(np.median(self.top_depths))

    @property
    def median_thickness(self) -> float:
        return float(np.median(self.thicknesses))


def _empirical_run_lengths(
    df: pd.DataFrame,
    rocks: list[str],
    bin_step_m: float,
) -> dict[str, float]:
    """Mean within-well run length (in cells) for each rock at bin_step_m.

    Used to calibrate the transition-matrix diagonal so the simulator
    reproduces the *observed* run-length distribution rather than the
    matrix's untruncated steady state — which can be much longer when
    the within-formation/within-well sequence is short relative to the
    chain's natural run length (DC and SL exhibit this).
    """
    out: dict[str, list[int]] = {r: [] for r in rocks}
    work = df[["borehole", "depth", "rock_type_fine"]].copy()
    work["rock_type_fine"] = work["rock_type_fine"].astype(str)
    work = work[work["rock_type_fine"].isin(rocks)]
    work = work.sort_values(["borehole", "depth"], kind="mergesort")
    work["_bin"] = np.floor(work["depth"].to_numpy() / bin_step_m).astype(np.int64)
    work = work.drop_duplicates(["borehole", "_bin"], keep="first")
    for _, well_df in work.groupby("borehole"):
        seq = well_df["rock_type_fine"].tolist()
        if not seq:
            continue
        cur = seq[0]
        n = 1
        for r in seq[1:]:
            if r == cur:
                n += 1
            else:
                if cur in out:
                    out[cur].append(n)
                cur = r
                n = 1
        if cur in out:
            out[cur].append(n)
    return {r: float(np.mean(v)) if v else 1.5 for r, v in out.items()}


def _fit_transition_matrix(
    df: pd.DataFrame,
    rock_set: set[str],
    bin_step_m: float = TRANSITION_BIN_STEP_M,
    laplace: float = TRANSITION_LAPLACE,
    calibrate_to_empirical_runs: bool = True,
    max_p_self: float = 0.97,
    max_gap_cells: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Fit a 1-step Markov transition matrix at the given depth step.

    `df` must contain columns ['borehole', 'depth', 'rock_type_fine']
    and be restricted to a single formation.  Per-well sequences are
    resampled to `bin_step_m` (drop duplicates within each bin, keep
    first); only contiguous (Δbin == 1) rock pairs contribute.  Laplace
    smoothing avoids hard-zero transitions for rare pairs.

    When `calibrate_to_empirical_runs=True` (default), each row's
    diagonal is shrunk so that 1/(1−P[i,i]) matches the rock's
    empirical mean run length in cells; the off-diagonal mass is
    rescaled to keep the row stochastic.  This addresses the failure
    mode where the raw within-well counts produce P_self ≈ 1 (because
    empirical runs are truncated by well/formation boundaries rather
    than internal transitions), causing the simulator to generate
    untruncated chains that are unrealistically long.

    Returns (P, n_transitions).
    """
    rocks = sorted(rock_set)
    n = len(rocks)
    if n == 0 or df.empty:
        return pd.DataFrame(np.eye(max(n, 1)), index=rocks, columns=rocks), 0
    idx = {r: i for i, r in enumerate(rocks)}
    counts = np.zeros((n, n), dtype=np.float64)
    n_obs = 0

    # Vectorise per well: bin depths, drop dupes, shift to get pairs, count
    # only where consecutive bins differ by exactly one step.
    work = df[["borehole", "depth", "rock_type_fine"]].copy()
    work["rock_type_fine"] = work["rock_type_fine"].astype(str)
    work = work[work["rock_type_fine"].isin(idx)]
    work = work.sort_values(["borehole", "depth"], kind="mergesort")
    work["_bin"] = np.floor(work["depth"].to_numpy() / bin_step_m).astype(np.int64)
    work = work.drop_duplicates(subset=["borehole", "_bin"], keep="first")
    work["_prev_bin"] = work.groupby("borehole")["_bin"].shift(1)
    work["_prev_rock"] = work.groupby("borehole")["rock_type_fine"].shift(1)
    pairs = work.dropna(subset=["_prev_bin", "_prev_rock"])
    # v3.3 (Phase S): allow up to MAX_GAP_CELLS missing bins between
    # observed bins. The previous strict +1 filter dropped rare-rock
    # transitions that survive across small log gaps -- making the
    # bank's matrix miss e.g. RB sandstone entirely.
    pair_gap = pairs["_bin"] - pairs["_prev_bin"]
    pairs = pairs[(pair_gap >= 1) & (pair_gap <= max_gap_cells)]
    if not pairs.empty:
        ij = pairs[["_prev_rock", "rock_type_fine"]].to_numpy()
        for r_prev, r_cur in ij:
            counts[idx[str(r_prev)], idx[str(r_cur)]] += 1
        n_obs = int(len(pairs))

    smoothed = counts + laplace
    row_sums = smoothed.sum(axis=1, keepdims=True)
    P = smoothed / row_sums

    if calibrate_to_empirical_runs and n_obs > 0:
        runs = _empirical_run_lengths(df, rocks, bin_step_m)
        for i, r in enumerate(rocks):
            target_p_self = 1.0 - 1.0 / max(runs[r], 1.5)
            target_p_self = float(min(target_p_self, max_p_self))
            current_p_self = float(P[i, i])
            if current_p_self <= target_p_self:
                continue
            off_sum = 1.0 - current_p_self
            if off_sum > 1e-12:
                # rescale off-diagonal to sum to (1 - target_p_self),
                # preserving its empirical relative shape.
                scale = (1.0 - target_p_self) / off_sum
                for j in range(n):
                    if j != i:
                        P[i, j] *= scale
                P[i, i] = target_p_self
            else:
                # degenerate: no observed transitions out of this rock
                # (e.g. SL claystone_cool).  Spread the deficit across
                # the other rocks weighted by their marginal frequency
                # within the formation.
                marginal = counts.sum(axis=0) + laplace
                marginal[i] = 0.0
                if marginal.sum() > 0:
                    marginal /= marginal.sum()
                else:
                    marginal = np.ones(n) / max(n - 1, 1)
                    marginal[i] = 0.0
                P[i, :] = marginal * (1.0 - target_p_self)
                P[i, i] = target_p_self

    return pd.DataFrame(P, index=rocks, columns=rocks), n_obs


class FormationGeometry:
    def __init__(self, formation_order: list[str]):
        self.formation_order = list(formation_order)
        self._strat_idx = {fm: i for i, fm in enumerate(formation_order)}
        self.formations: dict[str, FormationStats] = {}
        self.combinations: list[tuple[tuple[str, ...], int]] = []
        self.n_wells_total: int = 0
        self.n_wells_full_corpus: int = 0
        self.min_well_depth: float = DEFAULT_MIN_WELL_DEPTH
        self.max_well_depth: float = DEFAULT_MAX_WELL_DEPTH
        # Task 5b: basin clusters derived from (x_rd, y_rd) on the full
        # NLOG corpus.  combinations_by_basin[basin] is the same shape as
        # `combinations` but restricted to wells in that cluster, used by
        # sample_column() for stratified combination sampling.
        self.basin_centers: np.ndarray | None = None
        self.basin_labels_per_well: dict[str, int] = {}
        self.combinations_by_basin: dict[
            int, list[tuple[tuple[str, ...], int]]
        ] = {}

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        formation_order: Sequence[str] = FORMATION_ORDER,
        min_well_depth: float = DEFAULT_MIN_WELL_DEPTH,
        max_well_depth: float = DEFAULT_MAX_WELL_DEPTH,
        require_surface: bool = True,
        n_basins: int = 5,
        basin_seed: int = 42,
    ) -> "FormationGeometry":
        df = pd.read_parquet(parquet_path)
        df = df[df["dataset"] == "NLOG"]
        df = df.drop_duplicates(subset=["dataset", "borehole", "depth"],
                                 keep="first")

        well_max_depth = df.groupby("borehole")["depth"].max()
        in_window = well_max_depth[
            (well_max_depth >= min_well_depth) &
            (well_max_depth <= max_well_depth)
        ].index
        df_window = df[df["borehole"].isin(in_window)].copy()
        n_in_window = df_window["borehole"].nunique()
        print(f"FormationGeometry: combinations fit from "
              f"[{min_well_depth:.0f}, {max_well_depth:.0f}]m → "
              f"{n_in_window:,} wells")


        geom = cls(list(formation_order))
        geom.min_well_depth = min_well_depth
        geom.max_well_depth = max_well_depth
 
        # ---- Task 5b: cluster wells into basins via k-means on (x_rd, y_rd)
        # over the FULL NLOG corpus, so basin definitions are stable
        # regardless of the depth window used for combinations.
        if {"x_rd", "y_rd"}.issubset(df.columns):
            from sklearn.cluster import KMeans
            coords = (df.groupby("borehole")[["x_rd", "y_rd"]]
                        .mean().dropna())
            k = max(2, min(n_basins, len(coords)))
            km = KMeans(n_clusters=k, random_state=basin_seed, n_init=10)
            labels = km.fit_predict(coords.values)
            geom.basin_centers = km.cluster_centers_
            geom.basin_labels_per_well = dict(
                zip(coords.index.astype(str).tolist(), labels.tolist())
            )
            counts = Counter(labels.tolist())
            print(f"  basin k-means k={k} fit on {len(coords):,} wells: "
                  + ", ".join(f"basin{b}={c}" for b, c in sorted(counts.items())))
        else:
            print("  (no x_rd/y_rd columns in parquet — basin clustering disabled)")

        combinations = Counter()
        # combinations_per_basin[basin][combo] = count
        from collections import defaultdict as _dd
        combo_per_basin: dict[int, dict[tuple[str, ...], int]] = _dd(
            lambda: _dd(int)
        )
        n_kept = 0
        n_dropped_no_surface = 0
        for borehole, well_df in df_window.groupby("borehole"):
            present = set(well_df["formation"].unique())
            valid = [fm for fm in formation_order if fm in present]
            if not valid:
                continue
            if require_surface and valid[0] not in SURFACE_FORMATIONS:
                n_dropped_no_surface += 1
                continue
            combo = tuple(valid)
            combinations[combo] += 1
            basin = geom.basin_labels_per_well.get(str(borehole))
            if basin is not None:
                combo_per_basin[int(basin)][combo] += 1
            n_kept += 1

        geom.n_wells_total = n_kept
        print(f"  kept {n_kept} wells, dropped {n_dropped_no_surface} "
              f"(no surface formation)")
        print(f"  unique combinations: {len(combinations)}")
        geom.combinations = sorted(
            combinations.items(), key=lambda kv: -kv[1]
        )
        # drop basins below MIN_WELLS_PER_BASIN_FOR_STRATIFICATION (their
        # uniform-sampling weight would over-amplify them); fall back to
        # global frequency sampling if no basins survive.
        kept = {
            b: d for b, d in combo_per_basin.items()
            if sum(d.values()) >= MIN_WELLS_PER_BASIN_FOR_STRATIFICATION
        }
        geom.combinations_by_basin = {
            b: sorted(d.items(), key=lambda kv: -kv[1])
            for b, d in kept.items()
        }
        for b in sorted(combo_per_basin):
            n = sum(combo_per_basin[b].values())
            note = "" if b in kept else f"  [dropped: < {MIN_WELLS_PER_BASIN_FOR_STRATIFICATION} wells]"
            print(f"    basin {b}: {n} wells, "
                  f"{len(combo_per_basin[b])} unique combos{note}")
        n_unmapped = n_kept - sum(
            sum(d.values()) for d in combo_per_basin.values()
        )
        if n_unmapped > 0:
            print(f"    ({n_unmapped} pool wells lacked x_rd/y_rd, "
                  "ignored for stratified sampling — they remain in the "
                  "global combinations list as fallback)")

        n_full_corpus = df["borehole"].nunique()
        geom.n_wells_full_corpus = n_full_corpus
        print(f"\n  per-formation depth + facies fit from full "
              f"NLOG corpus: {n_full_corpus:,} wells")

        for fm in formation_order:
            sub = df[df["formation"] == fm]
            if len(sub) == 0:
                continue

            tops, thicks = [], []
            well_compositions: list[dict[str, float]] = []

            for borehole, well_df in sub.groupby("borehole"):
                top = float(well_df["depth"].min())
                bot = float(well_df["depth"].max())
                tops.append(top)
                thicks.append(bot - top)

                rock_counts = well_df.groupby(
                    "rock_type_fine", observed=True,
                ).size()
                rock_total = float(rock_counts.sum())
                if rock_total > 0:
                    comp = {str(r): float(c / rock_total)
                            for r, c in rock_counts.items()}
                    well_compositions.append(comp)

            tops = np.array(tops, dtype=np.float64)
            thicks = np.array(thicks, dtype=np.float64)
            if len(tops) < 2:
                continue

            facies_counts = sub.groupby(
                "rock_type_fine", observed=True,
            ).size()
            facies_total = float(facies_counts.sum())
            facies = {str(r): float(c / facies_total)
                      for r, c in facies_counts.items()}

            # Task 2: fit empirical transition matrix at TRANSITION_BIN_STEP_M
            trans_matrix, n_trans = _fit_transition_matrix(
                sub, rock_set=set(facies.keys()),
            )

            geom.formations[fm] = FormationStats(
                name=fm,
                n_wells=len(tops),
                top_depths=tops,
                thicknesses=thicks,
                facies=facies,
                well_compositions=well_compositions,
                transition_matrix=trans_matrix,
                n_transitions=n_trans,
            )
            fit_mode = ("matrix" if n_trans >= MIN_TRANSITIONS_FOR_FIT
                        else "persistence-fallback")
            print(f"  {fm:<4s}  n={len(tops):>4d}  "
                  f"top P5/50/95=[{np.percentile(tops, 5):>5.0f}, "
                  f"{np.percentile(tops, 50):>5.0f}, "
                  f"{np.percentile(tops, 95):>5.0f}]m  "
                  f"thk med={np.median(thicks):>5.0f}m  "
                  f"compositions={len(well_compositions)}  "
                  f"transitions={n_trans:,} ({fit_mode})")

        return geom

    # ---------- sampling -----------------------------------------------

    def sample_column(
        self,
        rng: np.random.Generator,
        max_depth: float = 4400.0,
        cell_height: float = DEFAULT_CELL_HEIGHT,
        facies_persistence: float = DEFAULT_FACIES_PERSISTENCE,
        composition_blend: float = DEFAULT_COMPOSITION_BLEND,
        min_layer_thickness: float = MIN_LAYER_THICKNESS_DEFAULT,
        max_resamples: int = MAX_RESAMPLES,
    ) -> list[tuple[str, list[str], float, float]]:
        if not self.combinations:
            raise RuntimeError("No combinations fit — call .fit() first")

        # Task 5b: when basin-stratified data is available, equalize basins
        # first (uniform draw over basin labels), then sample within the
        # basin weighted by combination frequency.  This mitigates bias
        # from over-represented basins (e.g. Groningen) in the 139-well
        # combination pool.  Falls back to global frequency sampling
        # when no basin metadata was attached.
        if self.combinations_by_basin:
            basin_keys = sorted(self.combinations_by_basin.keys())
            basin = basin_keys[int(rng.integers(0, len(basin_keys)))]
            combos, counts = zip(*self.combinations_by_basin[basin])
        else:
            combos, counts = zip(*self.combinations)
        weights = np.array(counts, dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(combos), p=weights))
        combination = combos[idx]

        tops = self._sample_tops_with_constraints(
            combination, rng, max_depth, min_layer_thickness, max_resamples,
        )

        layers: list[tuple[str, list[str], float, float]] = []
        for i, fm in enumerate(combination):
            top = tops[i]
            stats = self.formations.get(fm)
            # The last formation extends all the way to max_depth: in real
            # geology DC / basement do continue downward, even though many
            # wells don't log that interval.  Calibration of the Markov
            # diagonal (in _fit_transition_matrix) is what keeps run
            # lengths realistic when the slice is long, NOT truncating
            # the slice (which would leave large "other" gaps at the
            # column bottom and make every map look geologically broken).
            bot = tops[i + 1] if i + 1 < len(tops) else max_depth
            n_cells = max(1, int(np.ceil((bot - top) / cell_height)))
            if stats is None:
                rocks = ["other"] * n_cells
            elif stats.has_transition_matrix():
                # T-PROGS-style: empirical transition matrix drives both
                # marginals and run-length structure (Task 2).
                rocks = stats.sample_rocks_with_matrix(rng, n_cells)
            else:
                # Fallback for formations with too few observed transitions:
                # per-well composition blended with population average, then
                # uniform persistence Markov chain.
                comp = stats.sample_well_composition(
                    rng, blend_with_population=composition_blend,
                )
                rocks = stats.sample_rocks_markov(
                    rng, n_cells,
                    persistence=facies_persistence,
                    facies_override=comp,
                )
            layers.append((fm, rocks, top, bot))

        return layers

    def _sample_tops_with_constraints(
        self,
        combination: tuple[str, ...],
        rng: np.random.Generator,
        max_depth: float,
        min_layer_thickness: float,
        max_resamples: int,
    ) -> list[float]:
        n = len(combination)

        for attempt in range(max_resamples):
            sampled = []
            for fm in combination:
                stats = self.formations.get(fm)
                if stats is None:
                    sampled.append(None)
                else:
                    sampled.append(stats.sample_top_depth(rng))

            for i, t in enumerate(sampled):
                if t is None:
                    if i == 0:
                        sampled[i] = 0.0
                    else:
                        sampled[i] = sampled[i - 1] + min_layer_thickness

            max_allowed_last = max_depth - min_layer_thickness
            if sampled[-1] > max_allowed_last:
                sampled[-1] = max_allowed_last

            ok = True
            for i in range(1, n):
                if sampled[i] < sampled[i - 1] + min_layer_thickness:
                    ok = False
                    break
            if sampled[0] < 0:
                sampled[0] = 0.0

            if ok:
                return sampled

        sampled = []
        for fm in combination:
            stats = self.formations.get(fm)
            sampled.append(stats.median_top if stats else 0.0)
        sampled.sort()
        for i in range(1, n):
            if sampled[i] < sampled[i - 1] + min_layer_thickness:
                sampled[i] = sampled[i - 1] + min_layer_thickness
        if sampled[-1] > max_depth - min_layer_thickness:
            sampled[-1] = max_depth - min_layer_thickness
        for i in range(1, n):
            if sampled[i] < sampled[i - 1] + min_layer_thickness:
                sampled[i] = sampled[i - 1] + min_layer_thickness
        return sampled

    # ---------- IO -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        for s in self.formations.values():
            s._top_kde = None
            # Drop derived sampling caches — they're cheap to rebuild and
            # bloat the pickle.  The matrix itself is kept.
            s._trans_rocks = None
            s._trans_cumP = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved FormationGeometry to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FormationGeometry":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj._strat_idx = {fm: i for i, fm in enumerate(obj.formation_order)}
        for s in obj.formations.values():
            if not hasattr(s, "well_compositions"):
                s.well_compositions = []
            # backfill Task 2 attrs when loading older pickles
            if not hasattr(s, "transition_matrix"):
                s.transition_matrix = None
            if not hasattr(s, "n_transitions"):
                s.n_transitions = 0
            if not hasattr(s, "_trans_rocks"):
                s._trans_rocks = None
            if not hasattr(s, "_trans_cumP"):
                s._trans_cumP = None
        # backfill Task 5b basin attrs
        if not hasattr(obj, "basin_centers"):
            obj.basin_centers = None
        if not hasattr(obj, "basin_labels_per_well"):
            obj.basin_labels_per_well = {}
        if not hasattr(obj, "combinations_by_basin"):
            obj.combinations_by_basin = {}
        return obj

    def summary(self) -> pd.DataFrame:
        rows = []
        for fm in self.formation_order:
            s = self.formations.get(fm)
            if s is None:
                continue
            top_facies = sorted(s.facies.items(), key=lambda x: -x[1])[:2]
            facies_str = ", ".join(f"{r}({p:.0%})" for r, p in top_facies)
            t_p5, t_med, t_p95 = np.percentile(s.top_depths, [5, 50, 95])
            thk_med = np.median(s.thicknesses)
            rows.append({
                "fm": fm,
                "n_wells": s.n_wells,
                "n_compositions": len(s.well_compositions),
                "top_P5": f"{t_p5:.0f}",
                "top_med": f"{t_med:.0f}",
                "top_P95": f"{t_p95:.0f}",
                "thk_med": f"{thk_med:.0f}",
                "top_facies": facies_str,
            })
        return pd.DataFrame(rows)

    def top_combinations(self, n: int = 20) -> pd.DataFrame:
        rows = []
        total = sum(c for _, c in self.combinations)
        cumulative = 0
        for combo, count in self.combinations[:n]:
            cumulative += count
            rows.append({
                "combination": "→".join(combo),
                "n_formations": len(combo),
                "n_wells": count,
                "pct": f"{100*count/total:.1f}%",
                "cum_pct": f"{100*cumulative/total:.1f}%",
            })
        return pd.DataFrame(rows)


if __name__ == '__main__':
    print("=" * 60)
    print("Fitting FormationGeometry (per-column real-well facies, "
          "blended)")
    print("=" * 60)
    geom = FormationGeometry.fit(
        "data/clean/samples.parquet",
        min_well_depth=4000.0,
        max_well_depth=4500.0,
        require_surface=True,
    )
    geom.save("data/clean/formation_geometry.pkl")
    print("\nFormation summary:")
    print(geom.summary().to_string(index=False))