"""
Formation geometry — stage 1c of the simulator pipeline.

Fits the empirical depth, thickness, and facies distributions of each
NLOG formation from samples.parquet. The fitted artefact replaces the
hand-coded DUTCH_COLUMN in stratigraphy.py with data-driven sampling.

Per formation, FormationGeometry stores:

  * prevalence            — fraction of NLOG wells where the formation appears
  * top_depths            — empirical array of per-well formation tops
  * thicknesses           — empirical array of per-well formation thicknesses
  * facies                — {rock_type_fine: prob} from cell-share within
                            the formation

Column construction strategy (independent-marginal with single-pass
stratigraphic gap-filling):

  1. For each formation, decide presence via Bernoulli(prevalence).
  2. For each present formation, sample top and thickness independently
     from their empirical KDEs.
  3. Sort by stratigraphic order and resolve overlaps. Drop layers that
     end up thinner than min_layer_thickness.
  4. Anchor the topmost present formation to depth 0.
  5. Fill any internal gaps with ONE formation whose stratigraphic index
     lies between the surrounding layers and whose typical depth centre
     is closest to the gap centre. If no valid filler exists, extend
     the previous layer downward to close the gap.
  6. Fill any bottom gap with ONE formation OLDER than the deepest
     current layer. Same single-formation rule. If no valid filler
     exists, extend the deepest layer to max_depth.

This keeps formation prevalences close to the empirical NLOG numbers,
since each filler is added at most once per column position rather
than iterating through every eligible formation.

Stratigraphic constraints prevent younger formations (e.g., NU at the
surface) from appearing below older ones (e.g., DC Carboniferous at depth).

Usage
-----
    from simulator.formation_geometry import FormationGeometry

    geom = FormationGeometry.fit("data/clean/samples.parquet")
    geom.save("data/clean/formation_geometry.pkl")

    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    column = geom.sample_column(rng, max_depth=4400.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# Dutch stratigraphic order, youngest at top → oldest at bottom.
FORMATION_ORDER = [
    "NU", "NM", "NL",
    "CK", "KN",
    "SL", "SG", "AT",
    "RN", "RB",
    "ZE", "RO",
    "DC",
]


# Minimum thickness (m) for any layer to be retained in the column.
# Layers thinner than this after overlap resolution are dropped.
MIN_LAYER_THICKNESS_DEFAULT = 50.0


@dataclass
class FormationStats:
    """Empirical geometry + facies for one formation."""
    name: str
    n_wells: int
    prevalence: float
    top_depths: np.ndarray
    thicknesses: np.ndarray
    facies: dict[str, float] = field(default_factory=dict)
    _top_kde: gaussian_kde | None = None
    _thick_kde: gaussian_kde | None = None

    def sample_top(self, rng: np.random.Generator,
                    max_depth: float | None = None) -> float:
        if self._top_kde is None:
            self._top_kde = gaussian_kde(self.top_depths)
        v = float(self._top_kde.resample(1, seed=rng)[0, 0])
        v = max(v, 0.0)
        if max_depth is not None:
            v = min(v, max_depth - 10.0)
        return v

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
    def median_top(self) -> float:
        return float(np.median(self.top_depths))

    @property
    def median_centre(self) -> float:
        return self.median_top + 0.5 * float(np.median(self.thicknesses))


class FormationGeometry:
    """Collection of FormationStats indexed by formation code."""

    def __init__(self, formation_order: list[str]):
        self.formation_order = list(formation_order)
        self.formations: dict[str, FormationStats] = {}
        self.n_wells_total: int = 0
        self._strat_idx = {fm: i for i, fm in enumerate(formation_order)}

    # ---------- fitting -------------------------------------------------

    @classmethod
    def fit(
        cls,
        parquet_path: str | Path,
        formation_order: Sequence[str] = FORMATION_ORDER,
        min_n_wells: int = 10,
    ) -> "FormationGeometry":
        df = pd.read_parquet(parquet_path)
        df = df[df["dataset"] == "NLOG"]
        df = df.drop_duplicates(subset=["dataset", "borehole", "depth"],
                                 keep="first")

        n_wells_total = df["borehole"].nunique()
        print(f"FormationGeometry: {len(df):,} cells, {n_wells_total:,} wells")

        geom = cls(list(formation_order))
        geom.n_wells_total = n_wells_total

        for fm in formation_order:
            sub = df[df["formation"] == fm]
            if len(sub) == 0:
                print(f"  {fm:<4s}  no cells — skipped")
                continue

            per_well = sub.groupby("borehole").agg(
                top=("depth", "min"),
                bot=("depth", "max"),
            )
            per_well["thickness"] = per_well["bot"] - per_well["top"]
            n_wells = len(per_well)
            if n_wells < min_n_wells:
                print(f"  {fm:<4s}  only {n_wells} wells — skipped")
                continue

            facies_counts = sub.groupby("rock_type_fine",
                                          observed=True).size()
            facies_total = float(facies_counts.sum())
            facies = {str(r): float(c / facies_total)
                      for r, c in facies_counts.items()}

            stats = FormationStats(
                name=fm,
                n_wells=n_wells,
                prevalence=n_wells / n_wells_total,
                top_depths=per_well["top"].values.astype(np.float64),
                thicknesses=per_well["thickness"].values.astype(np.float64),
                facies=facies,
            )
            geom.formations[fm] = stats
            print(f"  {fm:<4s}  wells={n_wells:>5d}  prev={stats.prevalence:.2f}  "
                  f"top median={np.median(stats.top_depths):>5.0f}m  "
                  f"thick median={np.median(stats.thicknesses):>5.0f}m")
        return geom

    # ---------- sampling ------------------------------------------------

    def sample_column(
        self,
        rng: np.random.Generator,
        max_depth: float = 4400.0,
        min_layer_thickness: float = MIN_LAYER_THICKNESS_DEFAULT,
    ) -> list[tuple[str, str, float, float]]:
        """Sample one stratigraphic column using independent marginals
        with single-pass stratigraphic gap-filling."""

        # 1-2: independent marginal sampling per formation
        sampled = []  # [fm, top, bot, strat_idx]
        for i, fm in enumerate(self.formation_order):
            stats = self.formations.get(fm)
            if stats is None:
                continue
            if rng.random() > stats.prevalence:
                continue
            top = stats.sample_top(rng, max_depth=max_depth)
            thick = stats.sample_thickness(rng,
                                            min_thickness=min_layer_thickness)
            bot = min(top + thick, max_depth)
            if bot - top < min_layer_thickness:
                continue
            sampled.append([fm, top, bot, i])

        if not sampled:
            best = max(self.formations.values(),
                        key=lambda s: s.prevalence, default=None)
            if best is None:
                return [("NU", "clay", 0.0, max_depth)]
            rock = best.sample_rock(rng)
            return [(best.name, rock, 0.0, max_depth)]

        # 3: sort by stratigraphic order
        sampled.sort(key=lambda t: t[3])

        # 4: resolve overlaps walking in stratigraphic order — push
        # younger formations down if they overlap older ones.  Use pop()
        # rather than None-marking so the previous-layer index is always
        # valid.
        i = 1
        while i < len(sampled):
            prev_bot = sampled[i - 1][2]
            curr_top, curr_bot = sampled[i][1], sampled[i][2]
            if curr_top < prev_bot:
                shift = prev_bot - curr_top
                sampled[i][1] = prev_bot
                sampled[i][2] = min(curr_bot + shift, max_depth)
            if sampled[i][2] - sampled[i][1] < min_layer_thickness:
                # too thin after shifting — drop. don't advance i, since
                # the next layer will now compare against sampled[i-1]
                sampled.pop(i)
            else:
                i += 1

        if not sampled:
            return [("NU", "clay", 0.0, max_depth)]

        # 5: anchor topmost to depth 0
        sampled[0][1] = 0.0

        # 6: assemble layers, filling internal gaps with stratigraphically
        # valid fillers (single layer per gap, not iterative)
        layers: list[tuple[str, str, float, float]] = []
        present_indices = {s[3] for s in sampled}

        for k, (fm, top, bot, strat_idx) in enumerate(sampled):
            stats = self.formations[fm]

            if layers:
                prev_bot = layers[-1][3]
                if top > prev_bot:
                    # one filler from formations whose strat index is
                    # strictly between previous (above) and current (below)
                    upper_idx = self._strat_idx[layers[-1][0]]
                    lower_idx = strat_idx
                    self._fill_gap_single(
                        layers, prev_bot, top, rng,
                        min_strat_idx=upper_idx + 1,
                        max_strat_idx=lower_idx - 1,
                        exclude_indices=present_indices,
                        min_layer_thickness=min_layer_thickness,
                    )

            rock = stats.sample_rock(rng)
            layers.append((fm, rock, top, bot))

        # 7: fill bottom gap with ONE formation older than deepest layer
        if layers and layers[-1][3] < max_depth:
            deepest_idx = self._strat_idx[layers[-1][0]]
            self._fill_gap_single(
                layers, layers[-1][3], max_depth, rng,
                min_strat_idx=deepest_idx + 1,
                max_strat_idx=len(self.formation_order) - 1,
                exclude_indices=present_indices,
                min_layer_thickness=min_layer_thickness,
            )

        # 8: if there's still a bottom gap (no valid older formation, or
        # the chosen filler's sampled thickness fell short), extend the
        # deepest layer to max_depth as a last resort
        if layers and layers[-1][3] < max_depth:
            fm, rock, top, _ = layers[-1]
            layers[-1] = (fm, rock, top, max_depth)

        return layers

    def _fill_gap_single(
        self,
        layers: list[tuple[str, str, float, float]],
        gap_top: float,
        gap_bot: float,
        rng: np.random.Generator,
        min_strat_idx: int,
        max_strat_idx: int,
        exclude_indices: set[int],
        min_layer_thickness: float = 50.0,
    ) -> None:
        """Fill the depth interval [gap_top, gap_bot] with ONE layer
        whose stratigraphic index lies in [min_strat_idx, max_strat_idx].

        The chosen filler is the formation whose typical depth centre is
        closest to the gap's centre. Its sampled thickness from its own
        KDE determines how much of the gap it covers — if less than the
        whole gap, the previous layer is extended to the filler's top.
        If no valid filler exists, the previous layer is extended to fill
        the gap entirely.

        Mutates `layers` and `exclude_indices` in place.
        """
        if gap_bot - gap_top < min_layer_thickness:
            # too small to be worth filling — extend previous layer
            if layers:
                fm_prev, rock_prev, top_prev, _ = layers[-1]
                layers[-1] = (fm_prev, rock_prev, top_prev, gap_bot)
            return

        gap_centre = 0.5 * (gap_top + gap_bot)
        filler_fm = self._best_formation_in_strat_range(
            gap_centre,
            min_strat_idx=min_strat_idx,
            max_strat_idx=max_strat_idx,
            exclude_indices=exclude_indices,
        )

        if filler_fm is None:
            # no valid filler — extend previous layer to close the gap
            if layers:
                fm_prev, rock_prev, top_prev, _ = layers[-1]
                layers[-1] = (fm_prev, rock_prev, top_prev, gap_bot)
            return

        stats = self.formations[filler_fm]
        thick = stats.sample_thickness(rng,
                                        min_thickness=min_layer_thickness)
        filler_bot = min(gap_top + thick, gap_bot)

        if filler_bot - gap_top < min_layer_thickness:
            # filler too thin even after sampling — extend previous instead
            if layers:
                fm_prev, rock_prev, top_prev, _ = layers[-1]
                layers[-1] = (fm_prev, rock_prev, top_prev, gap_bot)
            return

        rock = stats.sample_rock(rng)
        layers.append((filler_fm, rock, gap_top, filler_bot))
        exclude_indices.add(self._strat_idx[filler_fm])

        # if the filler doesn't reach gap_bot, extend it to do so. We
        # only ever add ONE layer per gap, so this trailing extension
        # is the price of single-pass filling. Empirical thicknesses
        # from the KDE will usually mean only a small extension here.
        if filler_bot < gap_bot:
            f_fm, f_rock, f_top, _ = layers[-1]
            layers[-1] = (f_fm, f_rock, f_top, gap_bot)

    def _best_formation_in_strat_range(
        self,
        depth: float,
        min_strat_idx: int,
        max_strat_idx: int,
        exclude_indices: set[int],
    ) -> str | None:
        """Find the formation whose typical depth centre is closest to
        `depth`, restricted to stratigraphic indices in
        [min_strat_idx, max_strat_idx], excluding indices already used."""
        best_fm = None
        best_dist = float("inf")
        for fm, stats in self.formations.items():
            idx = self._strat_idx.get(fm)
            if idx is None:
                continue
            if idx < min_strat_idx or idx > max_strat_idx:
                continue
            if idx in exclude_indices:
                continue
            dist = abs(stats.median_centre - depth)
            if dist < best_dist:
                best_dist = dist
                best_fm = fm
        return best_fm

    # ---------- IO ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        for s in self.formations.values():
            s._top_kde = None
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
            top_p5, top_med, top_p95 = np.percentile(s.top_depths, [5, 50, 95])
            thk_med = np.median(s.thicknesses)
            top_facies = sorted(s.facies.items(), key=lambda x: -x[1])[:2]
            facies_str = ", ".join(f"{r}({p:.0%})" for r, p in top_facies)
            rows.append({
                "fm": fm,
                "n_wells": s.n_wells,
                "prev": f"{s.prevalence:.2f}",
                "P5_top": f"{top_p5:.0f}",
                "med_top": f"{top_med:.0f}",
                "P95_top": f"{top_p95:.0f}",
                "med_thick": f"{thk_med:.0f}",
                "top_facies": facies_str,
            })
        return pd.DataFrame(rows)