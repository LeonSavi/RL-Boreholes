"""
Distribution fitting — stage 1a of the simulator pipeline.

Two artefacts are produced from samples.parquet:

  * DistributionBank: P(variable | rock_type_fine, depth_bin), used to
    sample petrophysical values per cell during map generation.

  * DiscoveryPrior: P(rock_type_fine | depth_bin) conditional on
    hc_discovery=True, used to score candidate ore-body locations
    during map generation so that ore preferentially sits where the
    column's rock signature matches historical NLOG positive-discovery
    wells.

Both are fit from the same parquet but saved as separate pickles so they
can be regenerated independently.

Usage
-----
    from simulator.distributions import DistributionBank, DiscoveryPrior

    bank = DistributionBank.fit(
        "data/clean/samples.parquet",
        variables=["rhob", "gr_api", "dt_us_ft", "nphi", "pef"],
        depth_bins=[0, 400, 800, 1200, 1600, 2000, 2400, 2800,
                    3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000],
    )
    bank.save("data/clean/distributions.pkl")

    prior = DiscoveryPrior.fit(
        "data/clean/samples.parquet",
        depth_bins=bank.depth_bins,
        rock_types=bank.rock_types,
    )
    prior.save("data/clean/discovery_prior.pkl")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# physically-plausible clamps so rogue samples can't inject junk into the
# simulator. Match DISPLAY_BOUNDS in plots.py where sensible.
HARD_BOUNDS = {
    "rhob":         (1.20, 3.20),   # g/cc
    "gr_api":       (0.0, 300.0),   # API
    "dt_us_ft":     (40.0, 240.0),  # µs/ft
    "nphi":         (-0.05, 0.60),  # fractional
    "pef":          (1.0, 10.0),    # barns/e-
    "cali_in":      (4.0, 20.0),    # inches
    "res_deep_log": (-1.0, 5.0),    # log10 Ω·m
    "res_shal_log": (-1.0, 5.0),
    "sp_mv":        (-200.0, 200.0),
    "drho":         (-0.2, 0.2),
    "msus_si":      (1e-6, 1e-1),   # SI units
    "ngr_cps":      (0.0, 500.0),
}


# =============================================================================
# CellDistribution + DistributionBank
# =============================================================================

@dataclass
class CellDistribution:
    """Fitted distribution for one (rock_type_fine, depth_bin) cell."""
    rock_type: str
    depth_lo: float
    depth_hi: float
    variables: list[str]
    n_samples: int
    # per-variable marginals, fitted as KDE on clamped, finite values
    kdes: dict[str, gaussian_kde] = field(default_factory=dict)
    # per-variable support (empirical P5, P95 after clamping)
    supports: dict[str, tuple[float, float]] = field(default_factory=dict)
    # per-variable means/stds for standardisation in correlation sampling
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    # between-variable correlation matrix (Spearman, on ranks, robust to
    # heavy tails). Only computed across variables where this cell has
    # joint observations.
    corr_matrix: np.ndarray | None = None
    corr_variables: list[str] = field(default_factory=list)
    # Per-variable clipping bounds — overridden empirically at fit-time
    # (Task 5a) and falling back to HARD_BOUNDS for variables we never
    # observed enough of to set bounds for.
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw n joint samples across variables.

        Strategy: draw a correlated multivariate normal, map back to each
        variable's marginal distribution via rank-based transform. This is
        the Gaussian-copula approach — preserves the marginals (whatever
        their shape) while imposing the empirical correlation structure.
        """
        out = {}
        # defensive: ensure corr_matrix and corr_variables are consistent
        if (self.corr_matrix is not None
                and len(self.corr_variables) >= 2
                and self.corr_matrix.shape[0] == len(self.corr_variables)):
            d = len(self.corr_variables)
            mvn = rng.multivariate_normal(np.zeros(d), self.corr_matrix, size=n)
            from scipy.stats import norm
            u = norm.cdf(mvn)
            for i, var in enumerate(self.corr_variables):
                out[var] = self._inverse_cdf(var, u[:, i], rng)
        # variables not in the correlation matrix get independent marginal draws
        for var in self.variables:
            if var not in out:
                kde = self.kdes.get(var)
                if kde is None:
                    out[var] = np.full(n, np.nan)
                else:
                    out[var] = self._clamp(var, kde.resample(n, seed=rng)[0])
        return out

    def _inverse_cdf(self, var: str, u: np.ndarray, rng) -> np.ndarray:
        """Map uniform[0,1] samples to the variable's marginal via empirical
        quantiles drawn from the KDE. Approximate but fast."""
        kde = self.kdes.get(var)
        if kde is None:
            return np.full(len(u), np.nan)
        n_draws = 5000
        draws = self._clamp(var, kde.resample(n_draws, seed=rng)[0])
        draws.sort()
        return np.quantile(draws, u)

    def _clamp(self, var: str, values: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds.get(
            var, HARD_BOUNDS.get(var, (-np.inf, np.inf))
        )
        return np.clip(values, lo, hi)


class DistributionBank:
    """Collection of CellDistributions indexed by (rock_type, depth_bin)."""

    def __init__(self, variables: list[str], depth_bins: list[float]):
        self.variables = variables
        self.depth_bins = depth_bins
        self.cells: dict[tuple[str, int], CellDistribution] = {}
        self.rock_types: list[str] = []
        # [0.5%, 99.5%] empirical quantiles per variable, computed in fit()
        # from the full real corpus (Task 5a).  Replaces HARD_BOUNDS at
        # sampling time for variables present here; HARD_BOUNDS remains
        # the fallback for variables without enough data.
        self.empirical_bounds: dict[str, tuple[float, float]] = {}

    def bounds_for(self, var: str) -> tuple[float, float]:
        """Empirical bounds if fitted, otherwise HARD_BOUNDS, otherwise unbounded."""
        return self.empirical_bounds.get(
            var, HARD_BOUNDS.get(var, (-np.inf, np.inf))
        )

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        variables: Sequence[str],
        depth_bins: Sequence[float],
        min_samples_per_cell: int = 30,
        kde_bandwidth: str | float = "scott",
    ) -> "DistributionBank":
        """Fit all per-cell distributions from the cleaned samples parquet."""
        df = pd.read_parquet(parquet_path)
        bank = cls(list(variables), list(depth_bins))

        if "rock_type_fine" not in df.columns:
            raise ValueError("samples.parquet needs rock_type_fine (v4 schema)")

        df = df[df["measurement"].isin(variables)]
        # wide pivot so each row is (borehole, depth) with all variables
        # as columns — lets us compute correlations
        wide = df.pivot_table(
            index=["dataset", "borehole", "depth", "rock_type_fine"],
            columns="measurement",
            values="value",
            aggfunc="mean",
        ).reset_index()

        wide["depth_bin"] = pd.cut(
            wide["depth"],
            bins=depth_bins,
            labels=list(range(len(depth_bins) - 1)),
            include_lowest=True,
        )

        bank.rock_types = sorted(wide["rock_type_fine"].dropna().unique())
        print(f"Fitting distributions over {len(bank.rock_types)} rock types, "
              f"{len(depth_bins)-1} depth bins, {len(variables)} variables")

        # ---- Task 5a: empirical [0.5%, 99.5%] bounds per variable ----
        # Computed once, from the FULL real corpus (after winsorising to
        # HARD_BOUNDS so an upstream calibration glitch can't widen them).
        # Variables whose real corpus is too sparse fall back to HARD_BOUNDS
        # at sampling time via bank.bounds_for().
        print("\nempirical variable bounds (Task 5a):")
        for v in variables:
            if v not in wide.columns:
                continue
            vals = wide[v].dropna().to_numpy()
            hb_lo, hb_hi = HARD_BOUNDS.get(v, (-np.inf, np.inf))
            vals = vals[(vals >= hb_lo) & (vals <= hb_hi)]
            if len(vals) < 1000:
                print(f"  {v:14s}  (only {len(vals):,} clean samples — fall back to HARD_BOUNDS)")
                continue
            lo = float(np.quantile(vals, 0.005))
            hi = float(np.quantile(vals, 0.995))
            bank.empirical_bounds[v] = (lo, hi)
            print(f"  {v:14s}  empirical [{lo:>8.3f}, {hi:>8.3f}]   "
                  f"hard [{hb_lo:>7.2f}, {hb_hi:>7.2f}]   n={len(vals):,}")

        import time
        total_cells = 0
        t_start = time.time()
        for rock in bank.rock_types:
            for bin_idx in range(len(depth_bins) - 1):
                lo, hi = depth_bins[bin_idx], depth_bins[bin_idx + 1]
                sub = wide[
                    (wide["rock_type_fine"] == rock)
                    & (wide["depth_bin"] == bin_idx)
                ]
                if len(sub) < min_samples_per_cell:
                    continue
                t0 = time.time()
                cell = cls._fit_cell(
                    rock, lo, hi, list(variables), sub, kde_bandwidth,
                    bounds=bank.empirical_bounds,
                )
                dt = time.time() - t0
                bank.cells[(rock, bin_idx)] = cell
                total_cells += 1
                print(f"  {rock:14s} {lo:>5.0f}-{hi:<5.0f}m  "
                      f"n={cell.n_samples:>6d}  "
                      f"vars_corr={len(cell.corr_variables)}  "
                      f"({dt:.1f}s)")

        print(f"\nfitted {total_cells} cells in {time.time()-t_start:.0f}s")
        return bank

    @staticmethod
    def _fit_cell(
        rock: str,
        lo: float,
        hi: float,
        variables: list[str],
        sub: pd.DataFrame,
        bandwidth,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> CellDistribution:
        cell = CellDistribution(
            rock_type=rock,
            depth_lo=lo,
            depth_hi=hi,
            variables=variables,
            n_samples=len(sub),
            bounds=dict(bounds) if bounds else {},
        )
        # per-variable marginals.
        # Two-stage filtering before KDE fit:
        #   1. drop values outside HARD_BOUNDS (physically impossible)
        #   2. winsorise to empirical [P1, P99] (trims log-tool calibration
        #      glitches and bed-boundary cells that contaminate per-cell
        #      distributions). Without (2), the KDE picks up extreme
        #      outliers and produces unphysical samples in the simulator.
        for var in variables:
            if var not in sub.columns:
                continue
            vals = sub[var].dropna().values
            bounds = HARD_BOUNDS.get(var, (-np.inf, np.inf))
            vals = vals[(vals >= bounds[0]) & (vals <= bounds[1])]
            if len(vals) < 20: # min number of samples, if lower skip
                continue
            # winsorise to [P1, P99]
            p1, p99 = np.percentile(vals, [1, 99])
            vals = vals[(vals >= p1) & (vals <= p99)]
            if len(vals) < 20:
                continue
            try:
                cell.kdes[var] = gaussian_kde(vals, bw_method=bandwidth)
                cell.supports[var] = (
                    float(np.quantile(vals, 0.05)),
                    float(np.quantile(vals, 0.95)),
                )
                cell.means[var] = float(vals.mean())
                cell.stds[var] = float(vals.std())
            except np.linalg.LinAlgError:
                pass

        # between-variable correlations (Spearman, pairwise).
        # Pairwise (not full-dropna) because no NLOG well measures all
        # variables simultaneously. Correlation uses raw values (not the
        # winsorised marginals) — winsorisation per-variable would distort
        # the joint structure if extreme values are correlated across vars.
        varset = [v for v in variables if v in cell.kdes]
        if len(varset) >= 2:
            corr = np.eye(len(varset))
            for i in range(len(varset)):
                for j in range(i + 1, len(varset)):
                    vi, vj = varset[i], varset[j]
                    pair = sub[[vi, vj]].dropna()
                    if len(pair) < 30:
                        continue
                    ranks = pair.rank().values
                    if ranks[:, 0].std() < 1e-9 or ranks[:, 1].std() < 1e-9:
                        continue
                    c = np.corrcoef(ranks, rowvar=False)[0, 1]
                    if np.isfinite(c):
                        corr[i, j] = c
                        corr[j, i] = c
            corr = _nearest_psd(corr)
            cell.corr_matrix = corr
            cell.corr_variables = varset
        return cell

    def sample(
        self,
        rock_type: str,
        depth: float,
        n: int = 1,
        rng: np.random.Generator | None = None,
        interpolate: bool = True,
    ) -> dict[str, np.ndarray]:
        """Draw n joint samples of all variables for given rock type at depth.

        With `interpolate=True` (default), samples are blended between the
        two adjacent bin centres so the marginal distribution at depth d
        is a mixture of the floor-bin and ceiling-bin distributions.

        Three-level fallback when a requested cell/variable is missing:
          1. If the exact (rock_type, depth_bin) cell has no fit, use the
             nearest populated depth bin of the same rock type.
          2. For any variable that the chosen cell lacks a KDE for, borrow
             that variable's KDE from the nearest sibling cell.
          3. If the rock has no fitted cells anywhere, returns NaN.
        """
        if rng is None:
            rng = np.random.default_rng()

        if not interpolate:
            cell = self._resolve_cell(rock_type, depth)
            if cell is None:
                return {v: np.full(n, np.nan) for v in self.variables}
            return self._sample_cell_with_fallback(cell, rock_type, n, rng)

        centres = self._bin_centres()
        cell_lo, cell_hi, alpha = self._bracket_cells(rock_type, depth, centres)

        if cell_lo is None and cell_hi is None:
            return {v: np.full(n, np.nan) for v in self.variables}
        if cell_lo is None:
            return self._sample_cell_with_fallback(cell_hi, rock_type, n, rng)
        if cell_hi is None:
            return self._sample_cell_with_fallback(cell_lo, rock_type, n, rng)
        if cell_lo is cell_hi:
            return self._sample_cell_with_fallback(cell_lo, rock_type, n, rng)

        # binomial split preserves per-sample correlation structure
        n_hi = int(rng.binomial(n, alpha))
        n_lo = n - n_hi

        out = {v: np.empty(n, dtype=np.float32) for v in self.variables}

        if n_hi > 0:
            hi_samples = self._sample_cell_with_fallback(
                cell_hi, rock_type, n_hi, rng)
            for v in self.variables:
                out[v][:n_hi] = hi_samples.get(v, np.full(n_hi, np.nan))
        if n_lo > 0:
            lo_samples = self._sample_cell_with_fallback(
                cell_lo, rock_type, n_lo, rng)
            for v in self.variables:
                out[v][n_hi:] = lo_samples.get(v, np.full(n_lo, np.nan))

        idx = rng.permutation(n)
        for v in self.variables:
            out[v] = out[v][idx]
        return out

    def _resolve_cell(self, rock_type: str, depth: float
                      ) -> CellDistribution | None:
        bin_idx = np.searchsorted(self.depth_bins, depth, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, len(self.depth_bins) - 2))
        cell = self.cells.get((rock_type, bin_idx))
        if cell is None:
            cell = self._nearest_cell(rock_type, bin_idx)
        return cell

    def _bin_centres(self) -> list[float]:
        return [0.5 * (self.depth_bins[i] + self.depth_bins[i + 1])
                for i in range(len(self.depth_bins) - 1)]

    def _bracket_cells(
        self, rock_type: str, depth: float, centres: list[float],
    ) -> tuple[CellDistribution | None, CellDistribution | None, float]:
        populated = sorted(
            [(b, self.cells[(rock_type, b)])
             for (r, b) in self.cells if r == rock_type],
            key=lambda t: t[0],
        )
        if not populated:
            return None, None, 0.0

        for i, (b, c) in enumerate(populated):
            centre = centres[b]
            if depth <= centre:
                if i == 0:
                    return c, c, 0.0
                b_prev, c_prev = populated[i - 1]
                centre_prev = centres[b_prev]
                if centre == centre_prev:
                    return c_prev, c, 0.0
                alpha = (depth - centre_prev) / (centre - centre_prev)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                return c_prev, c, alpha

        _, c_last = populated[-1]
        return c_last, c_last, 0.0

    def _sample_cell_with_fallback(
        self,
        cell: CellDistribution,
        rock_type: str,
        n: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        bin_idx = 0
        for (r, b), c in self.cells.items():
            if c is cell and r == rock_type:
                bin_idx = b
                break

        donors = {}
        for v in self.variables:
            if v in cell.kdes:
                continue
            donor = self._nearest_cell_with_variable(rock_type, bin_idx, v)
            if donor is not None:
                donors[v] = donor

        if not donors:
            return cell.sample(n, rng)

        out = cell.sample(n, rng)
        for v, donor in donors.items():
            kde = donor.kdes.get(v)
            if kde is None:
                continue
            vals = kde.resample(n, seed=rng)[0]
            lo, hi = HARD_BOUNDS.get(v, (-np.inf, np.inf))
            out[v] = np.clip(vals, lo, hi)
        return out

    def _nearest_cell(self, rock_type: str, bin_idx: int
                      ) -> CellDistribution | None:
        candidates = [
            (abs(b - bin_idx), b)
            for (r, b) in self.cells
            if r == rock_type
        ]
        if not candidates:
            return None
        candidates.sort()
        _, nearest_bin = candidates[0]
        return self.cells[(rock_type, nearest_bin)]

    def _nearest_cell_with_variable(
        self, rock_type: str, bin_idx: int, variable: str,
    ) -> CellDistribution | None:
        candidates = []
        for (r, b), c in self.cells.items():
            if r != rock_type:
                continue
            if variable not in c.kdes:
                continue
            candidates.append((abs(b - bin_idx), c))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved DistributionBank to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DistributionBank":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # backfill Task 5a attrs on older pickles
        if not hasattr(obj, "empirical_bounds"):
            obj.empirical_bounds = {}
        for cell in obj.cells.values():
            if not hasattr(cell, "bounds"):
                cell.bounds = {}
        return obj

    def summary(self) -> pd.DataFrame:
        """Tabular summary of fit coverage."""
        rows = []
        for (rock, bin_idx), cell in self.cells.items():
            rows.append({
                "rock_type": rock,
                "depth_bin": f"{cell.depth_lo:.0f}-{cell.depth_hi:.0f}",
                "n_samples": cell.n_samples,
                "n_vars_marginal": len(cell.kdes),
                "n_vars_correlated": len(cell.corr_variables),
            })
        return pd.DataFrame(rows).sort_values(["rock_type", "depth_bin"])


# =============================================================================
# DiscoveryPrior — for rock-conditioned ore placement
# =============================================================================

@dataclass
class DiscoveryPrior:
    """P(rock_type_fine | depth_bin) conditional on hc_discovery=True.

    Built from samples.parquet, filtered to wells where a commercial
    hydrocarbon discovery was made (gas, oil, or gas+oil — see
    pull_data.py's hc_discovery=True definition).

    Stored as a 2D array `prob[rock_idx, bin_idx]` such that
    `prob[:, bin_idx].sum() == 1` for each populated bin.

    Used by orebody.py to score candidate ore-body centroids: a column
    whose rock types at each depth match high-probability cells of the
    prior gets a high score and is preferentially chosen as a host.
    """
    rock_types: list[str]            # ordered list, defines axis-0 of prob
    depth_bins: list[float]           # bin edges, defines axis-1 (n_bins-1)
    prob: np.ndarray                  # shape (n_rock_types, n_depth_bins)
    n_positive_wells: int
    n_positive_rows: int

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        depth_bins: Sequence[float],
        rock_types: Sequence[str] | None = None,
        smoothing: float = 1e-3,
    ) -> "DiscoveryPrior":
        """Compute P(rock | depth_bin, hc_discovery=True) from samples.parquet.

        Parameters
        ----------
        parquet_path : path to samples.parquet (must contain hc_discovery,
                       rock_type_fine, depth columns)
        depth_bins   : same edges as DistributionBank, for consistency
        rock_types   : optional ordering. If None, takes all unique values
                       present in positive-discovery rows.
        smoothing    : Laplace-smoothing constant added to every cell so
                       no rock-type / bin combination has zero probability
                       (avoids -inf in log-prob scoring).
        """
        df = pd.read_parquet(parquet_path)

        if "hc_discovery" not in df.columns:
            raise ValueError("samples.parquet missing hc_discovery column")
        positive = df[df["hc_discovery"] == True].copy()
        if len(positive) == 0:
            raise ValueError("no rows with hc_discovery=True found in parquet")

        # deduplicate to one row per (well, depth) — multi-variable rows
        # would otherwise over-count the same physical cell
        positive = positive.drop_duplicates(
            subset=["dataset", "borehole", "depth"], keep="first")

        positive["depth_bin"] = pd.cut(
            positive["depth"],
            bins=list(depth_bins),
            labels=list(range(len(depth_bins) - 1)),
            include_lowest=True,
        )

        if rock_types is None:
            rocks = sorted(positive["rock_type_fine"].dropna().unique().tolist())
        else:
            rocks = list(rock_types)
        rock_to_idx = {r: i for i, r in enumerate(rocks)}

        n_rocks = len(rocks)
        n_bins = len(depth_bins) - 1
        counts = np.zeros((n_rocks, n_bins), dtype=np.float64)

        for _, row in positive.iterrows():
            r = row.get("rock_type_fine")
            b = row.get("depth_bin")
            if r is None or pd.isna(r) or pd.isna(b):
                continue
            if r not in rock_to_idx:
                continue
            counts[rock_to_idx[r], int(b)] += 1.0

        # Laplace smoothing + per-bin normalisation
        counts += smoothing
        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        prob = counts / col_sums

        n_wells = positive["borehole"].nunique()
        n_rows = len(positive)
        print(f"DiscoveryPrior: {n_wells} positive wells, {n_rows:,} rows, "
              f"{n_rocks} rock types, {n_bins} depth bins")

        return cls(
            rock_types=rocks,
            depth_bins=list(depth_bins),
            prob=prob,
            n_positive_wells=n_wells,
            n_positive_rows=n_rows,
        )

    def log_prob(self, rock_type: str, depth: float) -> float:
        """Log-probability of seeing `rock_type` at `depth`, given positive
        hc_discovery. Returns -inf if rock_type is unknown to the prior."""
        if rock_type not in self.rock_types:
            return -np.inf
        ri = self.rock_types.index(rock_type)
        bi = int(np.clip(
            np.searchsorted(self.depth_bins, depth, side="right") - 1,
            0, len(self.depth_bins) - 2,
        ))
        p = self.prob[ri, bi]
        if p <= 0:
            return -np.inf
        return float(np.log(p))

    def score_column(
        self,
        rock_types_column: np.ndarray,   # (n_z,) of rock-type strings
        depth_axis: np.ndarray,          # (n_z,) of depths in metres
        depth_window: tuple[float, float] | None = None,
    ) -> float:
        """Sum of log-probabilities over a depth column.

        If `depth_window` is given (lo, hi), only the cells in that range
        contribute. Useful when scoring against a target reservoir
        interval (e.g., 2800-3400m for Rotliegend) rather than the full
        column.
        """
        if depth_window is not None:
            lo, hi = depth_window
            mask = (depth_axis >= lo) & (depth_axis < hi)
        else:
            mask = np.ones(len(depth_axis), dtype=bool)
        score = 0.0
        for r, d, m in zip(rock_types_column, depth_axis, mask):
            if not m:
                continue
            lp = self.log_prob(str(r), float(d))
            if np.isfinite(lp):
                score += lp
        return score

    def best_depth_in_column(
        self,
        rock_types_column: np.ndarray,
        depth_axis: np.ndarray,
        depth_window: tuple[float, float] | None = None,
    ) -> float:
        """Return the depth in `depth_axis` whose rock has the highest
        log-probability under the prior. Used to place the centre of an
        accepted ore body within a column."""
        if depth_window is not None:
            lo, hi = depth_window
            valid_mask = (depth_axis >= lo) & (depth_axis < hi)
        else:
            valid_mask = np.ones(len(depth_axis), dtype=bool)
        best_lp = -np.inf
        best_d = float(depth_axis[len(depth_axis) // 2])
        for r, d, m in zip(rock_types_column, depth_axis, valid_mask):
            if not m:
                continue
            lp = self.log_prob(str(r), float(d))
            if lp > best_lp:
                best_lp = lp
                best_d = float(d)
        return best_d

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved DiscoveryPrior to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "DiscoveryPrior":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> pd.DataFrame:
        """Per-bin top-3 rock types, useful for sanity-checking the prior
        looks geologically reasonable."""
        rows = []
        for bi in range(self.prob.shape[1]):
            lo, hi = self.depth_bins[bi], self.depth_bins[bi + 1]
            top_idx = np.argsort(self.prob[:, bi])[::-1][:3]
            top = [(self.rock_types[i], float(self.prob[i, bi]))
                   for i in top_idx]
            rows.append({
                "depth_bin": f"{lo:.0f}-{hi:.0f}",
                "top1": f"{top[0][0]} ({top[0][1]:.0%})",
                "top2": f"{top[1][0]} ({top[1][1]:.0%})",
                "top3": f"{top[2][0]} ({top[2][1]:.0%})",
            })
        return pd.DataFrame(rows)


# =============================================================================
# Helpers
# =============================================================================

def _nearest_psd(a: np.ndarray) -> np.ndarray:
    """Nearest positive semi-definite matrix via eigenvalue flooring.

    Correlation matrices from noisy rank data can have tiny negative
    eigenvalues that break multivariate_normal sampling.
    """
    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 1e-6)
    return (eigvecs * eigvals) @ eigvecs.T