"""
Formation geometry — stage 1c of the simulator pipeline.

Combination-based stratigraphic sampling. Built from NLOG wells whose
maximum drilled depth lies in a narrow window (default [4000, 4500] m),
so that:

  * Each well's formation set fits inside the simulator's max_depth (4400 m).
  * Each well's per-formation thickness is statistically consistent with
    the simulator output range.

For each well in the window we extract the ordered tuple of formations
present (in stratigraphic order, deduplicated). Tuples that don't begin
with a surface-touching formation (NU/NM/NL/CK/KN) are filtered out —
these come from re-entry/sidetrack wells whose interpretation skipped
the upper section.

Per formation, we store the empirical thickness array (filtered to the
same well set), KDE-fit at first sample.

Per combination, we store the well count for weighted sampling.

Column construction
-------------------
1. Sample a combination weighted by well count.
2. For each formation in the combination, sample a thickness from its
   empirical KDE (clamped at MIN_LAYER_THICKNESS).
3. If the total thickness > max_depth: resample thicknesses for the
   same combination, up to MAX_RESAMPLES attempts.
4. Final fallback if still > max_depth: scale all thicknesses
   proportionally to fit exactly.
5. Stack from depth 0 downward.
6. If total < max_depth: extend the deepest layer to max_depth.

This guarantees a continuous, gap-free column from 0 to max_depth, with
formations in correct stratigraphic order, drawn from real Dutch wells.
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

# Surface-touching formations: a combination must begin with one of these
# to be valid. Filters out re-entry/sidetrack wells with truncated
# interpretation.
SURFACE_FORMATIONS = {"NU", "NM", "NL", "CK", "KN"}

# Default well-depth window for fitting (metres).
DEFAULT_MIN_WELL_DEPTH = 4000.0
DEFAULT_MAX_WELL_DEPTH = 4500.0

# Minimum thickness (m) for any layer.
MIN_LAYER_THICKNESS_DEFAULT = 50.0

# Resample attempts before falling back to proportional scaling.
MAX_RESAMPLES = 20


@dataclass
class FormationStats:
    """Empirical thickness + facies for one formation, fit on the
    well-depth-windowed subset."""
    name: str
    n_wells: int
    thicknesses: np.ndarray
    facies: dict[str, float] = field(default_factory=dict)
    _thick_kde: gaussian_kde | None = None

    def sample_thickness(self, rng: np.random.Generator,
                          min_thickness: float = 50.0,
                          max_thickness: float | None = None) -> float:
        if self._thick_kde is None:
            self._thick_kde = gaussian_kde(self.thicknesses)
        v = float(self._thick_kde.resample(1, seed=rng)[0, 0])
        v = max(v, min_thickness)
        if max_thickness is not None:
            v = min(v, max_thickness)
        return v

    def sample_rock(self, rng: np.random.Generator) -> str:
        if not self.facies:
            return "other"
        rocks, probs = zip(*self.facies.items())
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()
        return str(rng.choice(rocks, p=probs))

    @property
    def median_thickness(self) -> float:
        return float(np.median(self.thicknesses))


class FormationGeometry:
    """Combination-based stratigraphic geometry, fit from NLOG wells in
    a target depth window."""

    def __init__(self, formation_order: list[str]):
        self.formation_order = list(formation_order)
        self._strat_idx = {fm: i for i, fm in enumerate(formation_order)}
        self.formations: dict[str, FormationStats] = {}
        # combinations: list of (formation_tuple, well_count) sorted by count
        self.combinations: list[tuple[tuple[str, ...], int]] = []
        self.n_wells_total: int = 0
        self.min_well_depth: float = DEFAULT_MIN_WELL_DEPTH
        self.max_well_depth: float = DEFAULT_MAX_WELL_DEPTH

    # ---------- fitting -------------------------------------------------

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        formation_order: Sequence[str] = FORMATION_ORDER,
        min_well_depth: float = DEFAULT_MIN_WELL_DEPTH,
        max_well_depth: float = DEFAULT_MAX_WELL_DEPTH,
        require_surface: bool = True,
    ) -> "FormationGeometry":
        """Fit combinations and per-formation thickness KDEs from NLOG.

        Parameters
        ----------
        parquet_path : path to samples.parquet
        formation_order : stratigraphic order, youngest → oldest
        min_well_depth, max_well_depth : keep only wells whose max cell
            depth is in [min, max] metres
        require_surface : if True, drop combinations whose first
            formation is not in SURFACE_FORMATIONS (re-entry wells)
        """
        df = pd.read_parquet(parquet_path)
        df = df[df["dataset"] == "NLOG"]
        df = df.drop_duplicates(subset=["dataset", "borehole", "depth"],
                                 keep="first")

        # filter to wells whose max depth is in [min, max]
        well_max_depth = df.groupby("borehole")["depth"].max()
        in_window = well_max_depth[
            (well_max_depth >= min_well_depth) &
            (well_max_depth <= max_well_depth)
        ].index
        df_window = df[df["borehole"].isin(in_window)].copy()
        n_in_window = df_window["borehole"].nunique()
        print(f"FormationGeometry: well-depth window "
              f"[{min_well_depth:.0f}, {max_well_depth:.0f}]m → "
              f"{n_in_window:,} wells")

        geom = cls(list(formation_order))
        geom.min_well_depth = min_well_depth
        geom.max_well_depth = max_well_depth

        # --- extract per-well ordered tuples + per-well thickness rows ---
        combinations = Counter()
        thickness_rows = {fm: [] for fm in formation_order}
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

            # per-formation thickness in this well
            for fm in valid:
                fm_cells = well_df[well_df["formation"] == fm]
                top = float(fm_cells["depth"].min())
                bot = float(fm_cells["depth"].max())
                thickness_rows[fm].append(bot - top)

        geom.n_wells_total = n_kept
        print(f"  kept {n_kept} wells, dropped {n_dropped_no_surface} "
              f"(no surface formation)")
        print(f"  unique combinations: {len(combinations)}")

        # --- store combinations sorted by count desc ---
        geom.combinations = sorted(
            combinations.items(), key=lambda kv: -kv[1]
        )

        # --- fit per-formation thickness + facies on the same well set ---
        for fm in formation_order:
            thicks = np.array(thickness_rows[fm], dtype=np.float64)
            if len(thicks) < 2:
                # not enough data — skip; combinations including this fm
                # will use a fallback thickness in sample_column
                print(f"  {fm:<4s}  only {len(thicks)} wells, skipped KDE fit")
                continue

            # filter NLOG cells in this formation, this well set
            sub = df_window[df_window["formation"] == fm]
            facies_counts = sub.groupby("rock_type_fine",
                                          observed=True).size()
            facies_total = float(facies_counts.sum())
            facies = {str(r): float(c / facies_total)
                      for r, c in facies_counts.items()}

            geom.formations[fm] = FormationStats(
                name=fm,
                n_wells=len(thicks),
                thicknesses=thicks,
                facies=facies,
            )
            print(f"  {fm:<4s}  n_wells={len(thicks):>3d}  "
                  f"thick median={np.median(thicks):>5.0f}m  "
                  f"thick P5/P95=[{np.percentile(thicks, 5):.0f}, "
                  f"{np.percentile(thicks, 95):.0f}]m")

        return geom

    # ---------- sampling ------------------------------------------------

    def sample_column(
        self,
        rng: np.random.Generator,
        max_depth: float = 4400.0,
        min_layer_thickness: float = MIN_LAYER_THICKNESS_DEFAULT,
        max_resamples: int = MAX_RESAMPLES,
    ) -> list[tuple[str, str, float, float]]:
        """Sample one stratigraphic column.

        1. Pick a combination weighted by its well count.
        2. Sample thicknesses for each formation; resample if total
           exceeds max_depth.
        3. Stack from depth 0; extend deepest if total < max_depth.

        Returns list of (formation, rock_type_fine, top, bot) tuples.
        """
        if not self.combinations:
            raise RuntimeError("No combinations fit — call .fit() first")

        # --- 1. sample a combination weighted by well count ------------
        combos, counts = zip(*self.combinations)
        weights = np.array(counts, dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(combos), p=weights))
        combination = combos[idx]

        # --- 2. sample thicknesses, resample until they fit ------------
        thicknesses = self._sample_thicknesses_fitting(
            combination, rng, max_depth, min_layer_thickness, max_resamples,
        )

        # --- 3. stack from depth 0 -------------------------------------
        layers: list[tuple[str, str, float, float]] = []
        current = 0.0
        for fm, thick in zip(combination, thicknesses):
            top = current
            bot = min(current + thick, max_depth)
            stats = self.formations.get(fm)
            rock = stats.sample_rock(rng) if stats else "other"
            layers.append((fm, rock, top, bot))
            current = bot
            if current >= max_depth:
                break

        # --- 4. extend deepest to max_depth if column is short ---------
        if layers and layers[-1][3] < max_depth:
            fm, rock, top, _ = layers[-1]
            layers[-1] = (fm, rock, top, max_depth)

        return layers

    def _sample_thicknesses_fitting(
        self,
        combination: tuple[str, ...],
        rng: np.random.Generator,
        max_depth: float,
        min_layer_thickness: float,
        max_resamples: int,
    ) -> list[float]:
        """Sample one thickness per formation in the combination.
        Resample until the total fits in max_depth, or fall back to
        proportional scaling after max_resamples attempts.

        Returns thicknesses in the same order as the combination.
        """
        for attempt in range(max_resamples):
            thicks = []
            for fm in combination:
                stats = self.formations.get(fm)
                if stats is None:
                    # fallback: fixed median guess
                    thicks.append(min_layer_thickness)
                else:
                    thicks.append(stats.sample_thickness(
                        rng, min_thickness=min_layer_thickness,
                    ))
            if sum(thicks) <= max_depth:
                return thicks

        # fallback: proportional scaling so the total equals max_depth
        # (still respects min_layer_thickness for each layer)
        total = sum(thicks)
        scale = max_depth / total
        thicks = [max(min_layer_thickness, t * scale) for t in thicks]

        # if min_layer_thickness floor pushed total back over max_depth,
        # truncate by dropping smallest layers (rare, only when the
        # combination has too many tiny formations)
        while sum(thicks) > max_depth and len(thicks) > 1:
            i_smallest = int(np.argmin(thicks))
            thicks.pop(i_smallest)

        return thicks

    # ---------- IO ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        for s in self.formations.values():
            s._thick_kde = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved FormationGeometry to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FormationGeometry":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj._strat_idx = {fm: i for i, fm in enumerate(obj.formation_order)}
        return obj

    def summary(self) -> pd.DataFrame:
        rows = []
        for fm in self.formation_order:
            s = self.formations.get(fm)
            if s is None:
                continue
            top_facies = sorted(s.facies.items(), key=lambda x: -x[1])[:2]
            facies_str = ", ".join(f"{r}({p:.0%})" for r, p in top_facies)
            thk_p5, thk_med, thk_p95 = np.percentile(
                s.thicknesses, [5, 50, 95]
            )
            rows.append({
                "fm": fm,
                "n_wells": s.n_wells,
                "thk_P5": f"{thk_p5:.0f}",
                "thk_med": f"{thk_med:.0f}",
                "thk_P95": f"{thk_p95:.0f}",
                "top_facies": facies_str,
            })
        return pd.DataFrame(rows)

    def top_combinations(self, n: int = 20) -> pd.DataFrame:
        """Return the top N combinations as a DataFrame."""
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
    print("\n" + "=" * 60)
    print("Fitting FormationGeometry from combinations of deep wells")
    print("=" * 60)
    
    geom = FormationGeometry.fit(
        "data/clean/samples.parquet",
        min_well_depth=4000.0,    # filter: well max depth ≥ 4000m
        max_well_depth=4500.0,    # filter: well max depth ≤ 4500m
        require_surface=True,      # drop wells whose first formation isn't
                                # NU/NM/NL/CK/KN (re-entry / sidetrack)
    )
    
    geom.save("data/clean/formation_geometry.pkl")
    
    print("\nTop 20 combinations:")
    print(geom.top_combinations(20).to_string(index=False))
    
    print("\nFormation summary:")
    print(geom.summary().to_string(index=False))
