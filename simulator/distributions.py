"""
Distribution fitting — stage 1a of the simulator pipeline.

Fits P(variable | rock_type_fine, depth_bin) from the cleaned NLOG+LILY
samples.parquet produced by pull_data.py v4.

The fitted distributions are the *parameters* of the simulator. Each rock
type at each depth gets a marginal KDE per variable, plus a correlation
matrix across variables for sampling physically-consistent joint values.

Usage
-----
    from simulator.distributions import DistributionBank
    bank = DistributionBank.fit(
        "data/clean/samples.parquet",
        variables=["rhob", "gr_api", "dt_us_ft", "nphi", "pef"],
        depth_bins=[0, 300, 800, 1500, 3000, 6000],
    )
    bank.save("data/clean/distributions.pkl")

    # later, at simulation time:
    bank = DistributionBank.load("data/clean/distributions.pkl")
    samples = bank.sample("sandstone", depth=1200, n=100)  # (100, n_vars)
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
# simulator.  match DISPLAY_BOUNDS in plots.py where sensible.
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
    # heavy tails).  only computed across variables where this cell has
    # joint observations.
    corr_matrix: np.ndarray | None = None
    corr_variables: list[str] = field(default_factory=list)

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw n joint samples across variables.

        Strategy: draw a correlated multivariate normal, map back to each
        variable's marginal distribution via rank-based transform.  This is
        the Gaussian-copula approach — preserves the marginals (whatever
        their shape) while imposing the empirical correlation structure.
        """
        out = {}
        # defensive: ensure corr_matrix and corr_variables are consistent.
        # can be violated if KDEs are edited after fitting.
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
        quantiles drawn from the KDE.  Approximate but fast."""
        kde = self.kdes.get(var)
        if kde is None:
            return np.full(len(u), np.nan)
        # build an empirical CDF by sampling the KDE heavily
        n_draws = 5000
        draws = self._clamp(var, kde.resample(n_draws, seed=rng)[0])
        draws.sort()
        # np.quantile is equivalent to empirical inverse CDF
        return np.quantile(draws, u)

    def _clamp(self, var: str, values: np.ndarray) -> np.ndarray:
        lo, hi = HARD_BOUNDS.get(var, (-np.inf, np.inf))
        return np.clip(values, lo, hi)


class DistributionBank:
    """Collection of CellDistributions indexed by (rock_type, depth_bin)."""

    def __init__(self, variables: list[str], depth_bins: list[float]):
        self.variables = variables
        self.depth_bins = depth_bins
        self.cells: dict[tuple[str, int], CellDistribution] = {}
        self.rock_types: list[str] = []

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

        # pivot long -> wide so each row is (borehole, depth) with all
        # variables as columns.  this lets us compute correlations.
        if "rock_type_fine" not in df.columns:
            raise ValueError("samples.parquet needs rock_type_fine (v4 schema)")

        # keep only rows for the variables of interest
        df = df[df["measurement"].isin(variables)]
        # wide pivot
        wide = df.pivot_table(
            index=["dataset", "borehole", "depth", "rock_type_fine"],
            columns="measurement",
            values="value",
            aggfunc="mean",
        ).reset_index()

        # bucket into depth bins
        wide["depth_bin"] = pd.cut(
            wide["depth"],
            bins=depth_bins,
            labels=list(range(len(depth_bins) - 1)),
            include_lowest=True,
        )

        bank.rock_types = sorted(wide["rock_type_fine"].dropna().unique())
        print(f"Fitting distributions over {len(bank.rock_types)} rock types, "
              f"{len(depth_bins)-1} depth bins, {len(variables)} variables")

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
                    rock, lo, hi, list(variables), sub, kde_bandwidth
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
    ) -> CellDistribution:
        cell = CellDistribution(
            rock_type=rock,
            depth_lo=lo,
            depth_hi=hi,
            variables=variables,
            n_samples=len(sub),
        )
        # per-variable marginals
        for var in variables:
            if var not in sub.columns:
                continue
            vals = sub[var].dropna().values
            bounds = HARD_BOUNDS.get(var, (-np.inf, np.inf))
            vals = vals[(vals >= bounds[0]) & (vals <= bounds[1])]
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
                # degenerate (all values identical) — skip
                pass

        # between-variable correlations (Spearman, pairwise).
        #
        # Previously used sub[varset].dropna() which required every row to
        # have all variables present.  In NLOG no single well measures all
        # 10 variables, so dropna() killed nearly every row for cells that
        # had plenty of marginal data — that's why well-populated cells
        # showed vars_corr=0 in the summary.
        #
        # Pairwise fix: for each variable pair, compute the Spearman
        # correlation using only rows where both are present.  Build the
        # d×d matrix pair by pair, then project to the nearest positive
        # semi-definite matrix so multivariate_normal sampling works.
        varset = [v for v in variables if v in cell.kdes]
        if len(varset) >= 2:
            corr = np.eye(len(varset))
            for i in range(len(varset)):
                for j in range(i + 1, len(varset)):
                    vi, vj = varset[i], varset[j]
                    pair = sub[[vi, vj]].dropna()
                    if len(pair) < 30:
                        # not enough joint observations — leave as 0
                        continue
                    # Spearman = Pearson on ranks
                    ranks = pair.rank().values
                    if ranks[:, 0].std() < 1e-9 or ranks[:, 1].std() < 1e-9:
                        continue
                    c = np.corrcoef(ranks, rowvar=False)[0, 1]
                    if np.isfinite(c):
                        corr[i, j] = c
                        corr[j, i] = c
            # regularise to positive semi-definite
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
        is a mixture of the floor-bin and ceiling-bin distributions,
        weighted by where d falls between their centres.  This removes
        the step-function artefact at bin boundaries.

        Three-level fallback when a requested cell/variable is missing:
          1. If the exact (rock_type, depth_bin) cell has no fit, use the
             nearest populated depth bin of the same rock type.
          2. For any variable that the chosen cell lacks a KDE for, borrow
             that variable's KDE from the nearest sibling cell (same rock,
             different depth bin) that *does* have it.
          3. If the rock has no fitted cells anywhere, returns NaN.
        """
        if rng is None:
            rng = np.random.default_rng()

        if not interpolate:
            cell = self._resolve_cell(rock_type, depth)
            if cell is None:
                return {v: np.full(n, np.nan) for v in self.variables}
            return self._sample_cell_with_fallback(cell, rock_type, n, rng)

        # interpolation: find the two bin centres bracketing `depth`
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

        # mix: n_hi samples from ceiling cell, rest from floor cell.
        # binomial split preserves per-sample correlation structure
        # (each sample is fully drawn from one cell, never mixed
        # mid-sample), while the *aggregate* distribution is the correct
        # alpha-weighted mixture.
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

        # shuffle so the hi/lo halves aren't in adjacent positions when
        # the caller assigns these to a spatial grid
        idx = rng.permutation(n)
        for v in self.variables:
            out[v] = out[v][idx]
        return out

    def _resolve_cell(self, rock_type: str, depth: float
                      ) -> CellDistribution | None:
        """Pick the cell for (rock_type, depth) with nearest-bin fallback."""
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
        """Return (cell_lo, cell_hi, alpha) where alpha ∈ [0,1] is the
        interpolation weight toward cell_hi.  Uses populated cells only:
        if rock's bin 2 is missing we interpolate between bins 1 and 3.

        alpha = 0  → use cell_lo  (depth at cell_lo's centre or below)
        alpha = 1  → use cell_hi  (depth at cell_hi's centre or above)
        """
        populated = sorted(
            [(b, self.cells[(rock_type, b)])
             for (r, b) in self.cells if r == rock_type],
            key=lambda t: t[0],
        )
        if not populated:
            return None, None, 0.0

        # compare `depth` against the populated bins' centres
        for i, (b, c) in enumerate(populated):
            centre = centres[b]
            if depth <= centre:
                if i == 0:
                    # below the shallowest populated centre — use it alone
                    return c, c, 0.0
                b_prev, c_prev = populated[i - 1]
                centre_prev = centres[b_prev]
                if centre == centre_prev:
                    return c_prev, c, 0.0
                alpha = (depth - centre_prev) / (centre - centre_prev)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                return c_prev, c, alpha

        # depth is above all populated centres — use the deepest alone
        _, c_last = populated[-1]
        return c_last, c_last, 0.0

    def _sample_cell_with_fallback(
        self,
        cell: CellDistribution,
        rock_type: str,
        n: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Run cell.sample(n) but fill in any variables the cell lacks by
        borrowing from the nearest sibling cell of the same rock type."""
        # locate this cell's bin index (needed for the borrow search)
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

    def _nearest_cell(self, rock_type: str, bin_idx: int) -> CellDistribution | None:
        """Find the populated cell for this rock type closest to bin_idx."""
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
        """Nearest cell (same rock) that has KDE fitted for `variable`."""
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
            return pickle.load(f)

    def summary(self) -> pd.DataFrame:
        """Tabular summary of fit coverage: which (rock, bin) cells have
        data, how many samples each, how many variables correlated."""
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


def _nearest_psd(a: np.ndarray) -> np.ndarray:
    """Nearest positive semi-definite matrix via eigenvalue flooring.

    Correlation matrices from noisy rank data can have tiny negative
    eigenvalues that break multivariate_normal sampling.
    """
    eigvals, eigvecs = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 1e-6)
    return (eigvecs * eigvals) @ eigvecs.T