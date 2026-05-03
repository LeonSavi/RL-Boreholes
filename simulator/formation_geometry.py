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


@dataclass
class FormationStats:
    name: str
    n_wells: int
    top_depths: np.ndarray
    thicknesses: np.ndarray
    facies: dict[str, float] = field(default_factory=dict)
    well_compositions: list[dict[str, float]] = field(default_factory=list)
    _top_kde: gaussian_kde | None = None

    def sample_top_depth(self, rng: np.random.Generator) -> float:
        if self._top_kde is None:
            self._top_kde = gaussian_kde(self.top_depths)
        v = float(self._top_kde.resample(1, seed=rng)[0, 0])
        return max(v, 0.0)

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
        """
        if not self.well_compositions:
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

    @property
    def median_top(self) -> float:
        return float(np.median(self.top_depths))

    @property
    def median_thickness(self) -> float:
        return float(np.median(self.thicknesses))


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

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        formation_order: Sequence[str] = FORMATION_ORDER,
        min_well_depth: float = DEFAULT_MIN_WELL_DEPTH,
        max_well_depth: float = DEFAULT_MAX_WELL_DEPTH,
        require_surface: bool = True,
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

        combinations = Counter()
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
            combinations[tuple(valid)] += 1
            n_kept += 1

        geom.n_wells_total = n_kept
        print(f"  kept {n_kept} wells, dropped {n_dropped_no_surface} "
              f"(no surface formation)")
        print(f"  unique combinations: {len(combinations)}")
        geom.combinations = sorted(
            combinations.items(), key=lambda kv: -kv[1]
        )

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

            geom.formations[fm] = FormationStats(
                name=fm,
                n_wells=len(tops),
                top_depths=tops,
                thicknesses=thicks,
                facies=facies,
                well_compositions=well_compositions,
            )
            print(f"  {fm:<4s}  n={len(tops):>4d}  "
                  f"top P5/50/95=[{np.percentile(tops, 5):>5.0f}, "
                  f"{np.percentile(tops, 50):>5.0f}, "
                  f"{np.percentile(tops, 95):>5.0f}]m  "
                  f"thk med={np.median(thicks):>5.0f}m  "
                  f"compositions={len(well_compositions)}")

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
            bot = tops[i + 1] if i + 1 < len(tops) else max_depth
            stats = self.formations.get(fm)
            n_cells = max(1, int(np.ceil((bot - top) / cell_height)))
            if stats is None:
                rocks = ["other"] * n_cells
            else:
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