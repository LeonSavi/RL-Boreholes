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
        if self.corr_matrix is not None and len(self.corr_variables) >= 2:
            # sample correlated standard normals, then transform to uniform
            # via the normal CDF, then to each variable's marginal via
            # inverse-CDF (which for KDE is approximated by empirical
            # quantiles).
            d = len(self.corr_variables)
            mvn = rng.multivariate_normal(np.zeros(d), self.corr_matrix, size=n)
            # normal CDF -> uniform(0,1)
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

        for rock in bank.rock_types:
            for bin_idx in range(len(depth_bins) - 1):
                lo, hi = depth_bins[bin_idx], depth_bins[bin_idx + 1]
                sub = wide[
                    (wide["rock_type_fine"] == rock)
                    & (wide["depth_bin"] == bin_idx)
                ]
                if len(sub) < min_samples_per_cell:
                    continue
                cell = cls._fit_cell(
                    rock, lo, hi, list(variables), sub, kde_bandwidth
                )
                bank.cells[(rock, bin_idx)] = cell
                print(f"  {rock:14s} {lo:>5.0f}-{hi:<5.0f}m  "
                      f"n={cell.n_samples:>6d}  "
                      f"vars_corr={len(cell.corr_variables)}")

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

        # between-variable correlations (Spearman rank corr, on pairs with
        # joint observations)
        varset = [v for v in variables if v in cell.kdes]
        joint = sub[varset].dropna()
        if len(joint) >= 30 and len(varset) >= 2:
            # rank-transform each column then compute Pearson == Spearman
            ranks = joint.rank().values
            corr = np.corrcoef(ranks, rowvar=False)
            # replace NaN (constant columns) with identity
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            # regularise: ensure positive-definite by adding small diagonal
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
    ) -> dict[str, np.ndarray]:
        """Draw n joint samples of all variables for given rock type at depth.

        Two-level fallback:
          1. If the exact (rock_type, depth_bin) cell has no fit, use the
             nearest populated depth bin of the same rock type.
          2. For any variable that the chosen cell lacks a KDE for, borrow
             that variable's KDE from the nearest sibling cell (same rock,
             different depth bin) that *does* have it. This is crucial for
             shallow cells where NLOG coverage is sparse — e.g. shallow
             chalk might lack NPHI, but deep chalk has it, and rock
             physics of NPHI vs. depth is predictable enough that the
             borrow is defensible.

        If the rock has no fitted cells at all, returns NaN.
        """
        if rng is None:
            rng = np.random.default_rng()

        bin_idx = np.searchsorted(self.depth_bins, depth, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, len(self.depth_bins) - 2))

        cell = self.cells.get((rock_type, bin_idx))
        if cell is None:
            cell = self._nearest_cell(rock_type, bin_idx)
        if cell is None:
            return {v: np.full(n, np.nan) for v in self.variables}

        # for any variable the chosen cell lacks, find a donor cell of
        # the same rock type that has it
        donors = {}
        for v in self.variables:
            if v in cell.kdes:
                continue
            donor = self._nearest_cell_with_variable(rock_type, bin_idx, v)
            if donor is not None:
                donors[v] = donor

        if not donors:
            return cell.sample(n, rng)

        # first get the cell's own samples (using its own KDEs + correlations)
        out = cell.sample(n, rng)
        # then fill missing variables from donor cells (uncorrelated draws)
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