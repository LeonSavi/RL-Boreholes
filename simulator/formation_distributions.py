"""
Formation-resolution distribution bank — parallel to DistributionBank.

The rock-resolution `DistributionBank` is indexed by
`(rock_type_fine, depth_bin)`. This module provides the formation-
resolution sibling indexed by `(formation, depth_bin)`. Everything
else — KDE marginals, Spearman copula, PSD projection, fallback
chain — is reused unchanged from `distributions.py`.

`CellDistribution` is reused as-is; its `rock_type` field carries the
formation name (e.g. "ZE") instead of a rock name. The field is only
used as a string label, so the overload is internal-only — the public
API uses `formation` throughout.

Usage
-----
    from simulator.formation_distributions import FormationDistributionBank

    bank = FormationDistributionBank.fit(
        "data/clean/samples.parquet",
        variables=["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"],
        depth_bins=[0, 200, 400, ..., 4400],
    )
    bank.save("data/clean/formation_distributions.pkl")
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd

from .distributions import (
    CellDistribution,
    DistributionBank,
    HARD_BOUNDS,
)


class FormationDistributionBank:
    """Collection of CellDistributions indexed by (formation, depth_bin)."""

    def __init__(self, variables: list[str], depth_bins: list[float]):
        self.variables = variables
        self.depth_bins = depth_bins
        self.cells: dict[tuple[str, int], CellDistribution] = {}
        self.formations: list[str] = []
        self.empirical_bounds: dict[str, tuple[float, float]] = {}

    def bounds_for(self, var: str) -> tuple[float, float]:
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
    ) -> "FormationDistributionBank":
        df = pd.read_parquet(parquet_path)
        bank = cls(list(variables), list(depth_bins))

        if "formation" not in df.columns:
            raise ValueError("samples.parquet needs a `formation` column")

        df = df[df["measurement"].isin(variables)]
        wide = df.pivot_table(
            index=["dataset", "borehole", "depth", "formation"],
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

        bank.formations = sorted(wide["formation"].dropna().unique())
        print(f"Fitting distributions over {len(bank.formations)} formations, "
              f"{len(depth_bins)-1} depth bins, {len(variables)} variables")

        # Empirical [0.5%, 99.5%] bounds per variable (shared with the rock
        # bank — they're a property of the variable, not the labelling).
        print("\nempirical variable bounds:")
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
        for fm in bank.formations:
            for bin_idx in range(len(depth_bins) - 1):
                lo, hi = depth_bins[bin_idx], depth_bins[bin_idx + 1]
                sub = wide[
                    (wide["formation"] == fm)
                    & (wide["depth_bin"] == bin_idx)
                ]
                if len(sub) < min_samples_per_cell:
                    continue
                t0 = time.time()
                # _fit_cell is a static method on DistributionBank; the
                # first arg ends up in cell.rock_type, which we overload
                # to hold the formation name. Internal-only — every
                # public API on this bank uses `formation`.
                cell = DistributionBank._fit_cell(
                    fm, lo, hi, list(variables), sub, kde_bandwidth,
                    bounds=bank.empirical_bounds,
                )
                dt = time.time() - t0
                bank.cells[(fm, bin_idx)] = cell
                total_cells += 1
                print(f"  {fm:14s} {lo:>5.0f}-{hi:<5.0f}m  "
                      f"n={cell.n_samples:>6d}  "
                      f"vars_corr={len(cell.corr_variables)}  "
                      f"({dt:.1f}s)")

        print(f"\nfitted {total_cells} cells in {time.time()-t_start:.0f}s")
        return bank

    def sample(
        self,
        formation: str,
        depth: float,
        n: int = 1,
        rng: np.random.Generator | None = None,
        interpolate: bool = True,
    ) -> dict[str, np.ndarray]:
        """Joint sample across variables for a (formation, depth) cell.

        Mirrors `DistributionBank.sample` exactly, including the
        depth-interpolation binomial split between adjacent bin centres
        and the three-level fallback for missing cells/variables.
        """
        if rng is None:
            rng = np.random.default_rng()

        if not interpolate:
            cell = self._resolve_cell(formation, depth)
            if cell is None:
                return {v: np.full(n, np.nan) for v in self.variables}
            return self._sample_cell_with_fallback(cell, formation, n, rng)

        centres = self._bin_centres()
        cell_lo, cell_hi, alpha = self._bracket_cells(formation, depth, centres)

        if cell_lo is None and cell_hi is None:
            return {v: np.full(n, np.nan) for v in self.variables}
        if cell_lo is None:
            return self._sample_cell_with_fallback(cell_hi, formation, n, rng)
        if cell_hi is None:
            return self._sample_cell_with_fallback(cell_lo, formation, n, rng)
        if cell_lo is cell_hi:
            return self._sample_cell_with_fallback(cell_lo, formation, n, rng)

        n_hi = int(rng.binomial(n, alpha))
        n_lo = n - n_hi

        out = {v: np.empty(n, dtype=np.float32) for v in self.variables}
        if n_hi > 0:
            hi_samples = self._sample_cell_with_fallback(
                cell_hi, formation, n_hi, rng)
            for v in self.variables:
                out[v][:n_hi] = hi_samples.get(v, np.full(n_hi, np.nan))
        if n_lo > 0:
            lo_samples = self._sample_cell_with_fallback(
                cell_lo, formation, n_lo, rng)
            for v in self.variables:
                out[v][n_hi:] = lo_samples.get(v, np.full(n_lo, np.nan))

        idx = rng.permutation(n)
        for v in self.variables:
            out[v] = out[v][idx]
        return out

    def _resolve_cell(self, formation: str, depth: float
                      ) -> CellDistribution | None:
        bin_idx = np.searchsorted(self.depth_bins, depth, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, len(self.depth_bins) - 2))
        cell = self.cells.get((formation, bin_idx))
        if cell is None:
            cell = self._nearest_cell(formation, bin_idx)
        return cell

    def _bin_centres(self) -> list[float]:
        return [0.5 * (self.depth_bins[i] + self.depth_bins[i + 1])
                for i in range(len(self.depth_bins) - 1)]

    def _bracket_cells(
        self, formation: str, depth: float, centres: list[float],
    ) -> tuple[CellDistribution | None, CellDistribution | None, float]:
        populated = sorted(
            [(b, self.cells[(formation, b)])
             for (f, b) in self.cells if f == formation],
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
        formation: str,
        n: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        bin_idx = 0
        for (f, b), c in self.cells.items():
            if c is cell and f == formation:
                bin_idx = b
                break

        donors = {}
        for v in self.variables:
            if v in cell.kdes:
                continue
            donor = self._nearest_cell_with_variable(formation, bin_idx, v)
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

    def _nearest_cell(self, formation: str, bin_idx: int
                      ) -> CellDistribution | None:
        candidates = [
            (abs(b - bin_idx), b)
            for (f, b) in self.cells
            if f == formation
        ]
        if not candidates:
            return None
        candidates.sort()
        _, nearest_bin = candidates[0]
        return self.cells[(formation, nearest_bin)]

    def _nearest_cell_with_variable(
        self, formation: str, bin_idx: int, variable: str,
    ) -> CellDistribution | None:
        candidates = []
        for (f, b), c in self.cells.items():
            if f != formation:
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
        print(f"Saved FormationDistributionBank to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FormationDistributionBank":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "empirical_bounds"):
            obj.empirical_bounds = {}
        for cell in obj.cells.values():
            if not hasattr(cell, "bounds"):
                cell.bounds = {}
        return obj

    def summary(self) -> pd.DataFrame:
        rows = []
        for (fm, bin_idx), cell in self.cells.items():
            rows.append({
                "formation": fm,
                "depth_bin": f"{cell.depth_lo:.0f}-{cell.depth_hi:.0f}",
                "n_samples": cell.n_samples,
                "n_vars_marginal": len(cell.kdes),
                "n_vars_correlated": len(cell.corr_variables),
            })
        return pd.DataFrame(rows).sort_values(["formation", "depth_bin"])
