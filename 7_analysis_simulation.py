"""Validate the simulator's fidelity vs the real NLOG+LILY corpus.

Generates a small sample (~30 maps) via `simulator.map_generator.MapGenerator`,
then compares those maps to `data/clean/samples.parquet` along four axes:

  A. Per-(rock × variable) marginals — KS statistic + Wasserstein distance.
  B. Per-formation rock composition — TVD (total variation distance).
  C. Vertical sequence statistics — Markov transition matrices + run lengths.
  D. Spatial structure — one sample map cross-section.

It also visualises the **DistributionBank** itself (the KDEs the simulator
draws from), so the thesis can justify why the empirical distributions are
realistic.

Outputs (plots/simulation/):
  SIMULATION_REPORT.md
  per_rock_marginal_distances.csv
  per_formation_composition.csv
  per_formation_transition_distance.csv
  01_distribution_bank_overview.png       (DistributionBank KDEs)
  02_marginal_sim_vs_real_rhob.png        (per-rock sim vs real rhob)
  03_marginal_sim_vs_real_gr.png          (per-rock sim vs real GR)
  04_formation_composition.png            (per-formation stacked bars)
  05_transition_matrices.png              (sim vs real Markov heatmaps)
  06_run_length_distributions.png         (sim vs real run lengths)
  07_sample_map_cross_section.png         (one simulated column + variables)
  08_lateral_coherence.png                (one simulated map's lateral strip)
  09_metric_summary.png                   (heatmap of all distances)

Run:
    python 7_analysis_simulation.py
"""
from __future__ import annotations

import os

# WSL memory pressure: keep BLAS single-threaded so peak RSS stays bounded.
# Must be set before numpy/scipy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
# poster-legible default fonts
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry, FORMATION_ORDER
from simulator.map_generator import MapGenerator, SimConfig


# ─── defaults ────────────────────────────────────────────────────────────

DEFAULT_SAMPLES = Path("data/clean/samples.parquet")
DEFAULT_BANK    = Path("data/clean/distributions.pkl")
DEFAULT_GEOM    = Path("data/clean/formation_geometry.pkl")
DEFAULT_PRIOR   = Path("data/clean/discovery_prior.pkl")
DEFAULT_DATASET = Path("data/dataset")
DEFAULT_OUT     = Path("plots/simulation")
DEFAULT_N_MAPS  = 500
DEFAULT_SEED    = 42

TARGET_VARS = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
N_X, N_Y, N_DEPTH = 32, 32, 440

ROCK_COLOURS = {
    "chalk":           "#a6d96a",
    "sandstone_clean": "#b15928",
    "sandstone_shaly": "#fb9a99",
    "sandstone":       "#cccc99",
    "claystone_hot":   "#762a83",
    "claystone_cool":  "#9970ab",
    "claystone":       "#8073ac",
    "clay":            "#ef8a62",
    "halite_pure":     "#fdbf6f",
    "anhydrite":       "#ff7f00",
    "dolomite":        "#33a02c",
    "limestone":       "#73c476",
    "siltstone":       "#fdae61",
    "mudstone":        "#b2abd2",
    "other":           "#bbbbbb",
    "basalt":          "#4d4d4d",
    "nanno_ooze":      "#67a9cf",
    "diatom_ooze":     "#2166ac",
    "halite":          "#ffd700",
}


# ─── simulation sample ────────────────────────────────────────────────────

def generate_sim_maps(bank: DistributionBank,
                      geom: FormationGeometry,
                      prior: DiscoveryPrior,
                      n_maps: int,
                      seed: int) -> list[dict]:
    """Pull `n_maps` synthetic maps off MapGenerator."""
    gen = MapGenerator(bank, geom, SimConfig(), seed=seed, prior=prior)
    out = []
    for i in range(n_maps):
        out.append(next(gen))
        if (i + 1) % 10 == 0 or (i + 1) == n_maps:
            print(f"    generated {i+1}/{n_maps}")
    return out


def dataset_has_labels(dataset_dir: Path) -> bool:
    """Saved dataset has labels iff labels_vocab.pkl + at least one
    labels_*.npz exist."""
    if not (dataset_dir / "labels_vocab.pkl").exists():
        return False
    return any(dataset_dir.glob("labels_*.npz"))


class _StreamingMapList:
    """Lazy, list-like view over saved boreholes_*.npy maps.

    Each __iter__ / __getitem__ call re-reads ONE map from disk and
    decodes it into the dict shape MapGenerator yields. Memory peak is
    bounded to one map at a time, which keeps WSL stable when n_maps is
    large (~thousands).

    Drop-in for the consumers that just iterate or index `maps`:
        for m in maps: ...        # streams one map at a time
        m0 = maps[0]               # single map for the cross-section plot
        len(maps)                  # number of maps
    """

    def __init__(self, dataset_dir: Path, n_maps: int,
                 variables: list[str]) -> None:
        with open(dataset_dir / "labels_vocab.pkl", "rb") as f:
            vocabs = pickle.load(f)
        self._inv_rocks = {i: r for r, i in vocabs["rocks"].items()}
        self._inv_fms = {i: f for f, i in vocabs["formations"].items()}

        with open(dataset_dir / "stats.pkl", "rb") as f:
            self._stats = pickle.load(f)
        with open(dataset_dir / "config.pkl", "rb") as f:
            cfg = pickle.load(f)
        self._saved_vars = list(cfg["variables"])
        self._variables = list(variables)

        all_bh = sorted(dataset_dir.glob("boreholes_*.npy"))
        bh_files: list[Path] = []
        for p in all_bh[:n_maps]:
            idx = int(p.stem.split("_")[-1])
            if (dataset_dir / f"labels_{idx:05d}.npz").exists():
                bh_files.append(p)
        if not bh_files:
            raise RuntimeError(f"no boreholes_*.npy in {dataset_dir}")
        self._bh_files = bh_files
        self._dataset_dir = dataset_dir
        self._depth_axis = np.arange(N_DEPTH, dtype=np.float64) * 10.0 + 5.0

    def __len__(self) -> int:
        return len(self._bh_files)

    def __iter__(self):
        for bh_path in self._bh_files:
            yield self._decode(bh_path)

    def __getitem__(self, idx: int) -> dict:
        return self._decode(self._bh_files[idx])

    def __bool__(self) -> bool:
        return len(self._bh_files) > 0

    def _decode(self, bh_path: Path) -> dict:
        idx = int(bh_path.stem.split("_")[-1])
        lbl_path = self._dataset_dir / f"labels_{idx:05d}.npz"
        bh = np.load(bh_path).astype(np.float32)          # (1024, V, D)
        with np.load(lbl_path) as z:
            rocks_int = z["rocks"]
            forms_int = z["formations"]

        var_arrays: dict[str, np.ndarray] = {}
        for i, v in enumerate(self._saved_vars):
            if v not in self._variables:
                continue
            mean, std = self._stats[v]
            raw = bh[:, i, :] * std + mean
            var_arrays[v] = raw.reshape(N_X, N_Y, N_DEPTH)
        del bh

        rock_str = np.vectorize(self._inv_rocks.get)(rocks_int).reshape(
            N_X, N_Y, N_DEPTH)
        fm_str = np.vectorize(self._inv_fms.get)(forms_int).reshape(
            N_X, N_Y, N_DEPTH)

        return {
            "rock_types": rock_str,
            "formations": fm_str,
            "variables":  var_arrays,
            "depth_axis": self._depth_axis,
        }


def load_sim_maps_from_disk(dataset_dir: Path,
                             n_maps: int,
                             variables: list[str],
                             ):
    """Streaming, list-like view over `n_maps` saved maps in `dataset_dir`.

    Returns a `_StreamingMapList` that decodes one map per iteration so
    memory peak stays bounded.  Each yielded map matches the dict shape
    MapGenerator emits ({rock_types, formations, variables, depth_axis});
    variable arrays are un-standardised back to raw units.
    """
    maps = _StreamingMapList(dataset_dir, n_maps, variables)
    print(f"    streaming {len(maps)} maps from {dataset_dir} "
          f"(one map in RAM at a time)")
    return maps


def streaming_aggregate(maps,
                         variables: list[str],
                         max_samples_per_rock_var: int = 50_000,
                         rng_seed: int = 17,
                         ) -> dict:
    """ONE pass over maps. Accumulates everything downstream needs without
    ever holding more than one map's arrays in RAM at a time.

    Returns a dict with:
      - sim_values:        {rock: {var: np.ndarray}}  reservoir-sampled
                           to <= max_samples_per_rock_var per (rock, var).
      - sim_by_fm:         {formation: Counter({rock: n})}
      - transitions_by_fm: {formation: Counter({(rock_a, rock_b): n})}
      - run_lengths:       {rock: [int, int, ...]}
      - first_map:         the first decoded map (for cross-section plots)
      - n_maps:            int, how many maps were aggregated

    The per-map work mirrors what pool_sim_values / pool_sim_formation_rock /
    transition_matrix_from_maps / plot_run_lengths used to do separately, but
    in one disk-pass instead of 4+ (which was catastrophic under the
    streaming map loader). Reservoir sampling keeps marginal-distance memory
    bounded; small Counter accumulators handle the rest.
    """
    rng = np.random.default_rng(rng_seed)
    sim_values: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    seen_counts: dict[tuple[str, str], int] = defaultdict(int)
    sim_by_fm: dict[str, Counter] = defaultdict(Counter)
    transitions_by_fm: dict[str, Counter] = defaultdict(Counter)
    run_lengths: dict[str, list] = defaultdict(list)
    first_map = None
    n_maps = 0

    for m in maps:
        n_maps += 1
        if first_map is None:
            first_map = m

        rt_obj = np.asarray(m["rock_types"]).astype(object)   # (X, Y, D)
        fm_obj = np.asarray(m["formations"]).astype(object)

        # ---- marginals: reservoir-sample per (rock, var) -----------------
        rt_flat = rt_obj.ravel()
        rt_str = np.where(rt_flat == None, "", rt_flat.astype(str))  # noqa: E711
        for v in variables:
            arr = m["variables"].get(v)
            if arr is None:
                continue
            v_flat = np.asarray(arr, dtype=np.float32).ravel()
            finite = np.isfinite(v_flat)
            valid = finite & (rt_str != "")
            if not valid.any():
                continue
            r_arr = rt_str[valid]
            x_arr = v_flat[valid]
            for rock in np.unique(r_arr):
                rock_mask = r_arr == rock
                xs = x_arr[rock_mask]
                if xs.size == 0:
                    continue
                # Algorithm R reservoir per (rock, v)
                key = (str(rock), v)
                bag = sim_values[str(rock)][v]
                cap = max_samples_per_rock_var
                seen_before = seen_counts[key]
                if seen_before < cap:
                    take = min(cap - seen_before, xs.size)
                    bag.extend(xs[:take].tolist())
                    seen_before += take
                    remaining = xs[take:]
                else:
                    remaining = xs
                if remaining.size:
                    # for each leftover sample, replace a random slot with
                    # prob cap / (seen_before + i)
                    idxs = seen_before + np.arange(remaining.size)
                    seen_before += remaining.size
                    probs = cap / (idxs + 1)
                    flips = rng.random(remaining.size) < probs
                    chosen = remaining[flips]
                    if chosen.size:
                        slots = rng.integers(0, cap, size=chosen.size)
                        for slot, val in zip(slots, chosen):
                            bag[int(slot)] = float(val)
                seen_counts[key] = seen_before

        # ---- composition: per-formation rock counts ----------------------
        rt_str_obj = rt_obj.astype(str)
        fm_str_obj = fm_obj.astype(str)
        rt_f = rt_str_obj.ravel()
        fm_f = fm_str_obj.ravel()
        valid_fm = (rt_f != "None") & (fm_f != "None")
        for r, f in zip(rt_f[valid_fm], fm_f[valid_fm]):
            sim_by_fm[f][r] += 1

        # ---- transitions + run lengths (column by column) ----------------
        # Iterating numpy axes-0,1 in Python is unavoidable here because the
        # state machine is per-column; but each column is ~440 ints, cheap.
        nx, ny, nz = rt_obj.shape
        for x in range(nx):
            for y in range(ny):
                col_r = rt_str_obj[x, y]    # (D,)
                col_f = fm_str_obj[x, y]
                cur_r = col_r[0]
                run = 1
                for k in range(1, nz):
                    r_prev = col_r[k - 1]
                    r_now = col_r[k]
                    f_prev = col_f[k - 1]
                    f_now = col_f[k]
                    # transitions (only within same formation)
                    if (r_prev != "None" and r_now != "None"
                            and f_prev == f_now and f_prev != "None"):
                        transitions_by_fm[f_prev][(r_prev, r_now)] += 1
                    # run lengths
                    if r_now == cur_r:
                        run += 1
                    else:
                        if cur_r != "None":
                            run_lengths[cur_r].append(run)
                        cur_r = r_now
                        run = 1
                if cur_r != "None":
                    run_lengths[cur_r].append(run)

        if n_maps % 25 == 0:
            print(f"    aggregated {n_maps} maps")

    print(f"    aggregated {n_maps} maps (single pass)")
    return {
        "sim_values": {r: {v: np.asarray(vs, dtype=np.float32)
                            for v, vs in vd.items()}
                        for r, vd in sim_values.items()},
        "sim_by_fm": sim_by_fm,
        "transitions_by_fm": transitions_by_fm,
        "run_lengths": run_lengths,
        "first_map": first_map,
        "n_maps": n_maps,
    }


# Note: pool_sim_values and pool_sim_formation_rock are removed -- their
# per-map work is now folded into streaming_aggregate (single pass).


def pool_real_formation_rock(real_df: pd.DataFrame) -> dict[str, Counter]:
    by_fm: dict[str, Counter] = defaultdict(Counter)
    nlog = real_df[real_df["dataset"] == "NLOG"].drop_duplicates(
        ["borehole", "depth"])
    for fm, grp in nlog.groupby("formation"):
        by_fm[str(fm)].update(grp["rock_type_fine"].astype(str).tolist())
    return by_fm


# ─── stats ────────────────────────────────────────────────────────────────

def marginal_distances(sim: dict[str, dict[str, np.ndarray]],
                       real_df: pd.DataFrame,
                       variables: list[str]) -> pd.DataFrame:
    """Per (rock, variable): KS statistic, Wasserstein distance, medians."""
    rows = []
    real_lookup = (real_df.groupby(["rock_type_fine", "measurement"])["value"]
                          .apply(lambda s: s.dropna().to_numpy())
                          .to_dict())
    for rock, vd in sim.items():
        for v in variables:
            sv = vd.get(v)
            if sv is None or len(sv) < 50:
                continue
            rv = real_lookup.get((rock, v))
            if rv is None or len(rv) < 50:
                continue
            ks_stat, ks_p = stats.ks_2samp(sv, rv)
            try:
                w = float(stats.wasserstein_distance(sv, rv))
            except Exception:
                w = float("nan")
            rows.append({
                "rock":        rock,
                "variable":    v,
                "n_sim":       int(len(sv)),
                "n_real":      int(len(rv)),
                "sim_p50":     round(float(np.median(sv)), 4),
                "real_p50":    round(float(np.median(rv)), 4),
                "ks_stat":     round(float(ks_stat), 4),
                "ks_p":        round(float(ks_p), 6),
                "wasserstein": round(w, 4),
            })
    return (pd.DataFrame(rows)
              .sort_values(["variable", "wasserstein"]))


# 9 fine rock classes; any other rock label is dropped from the
# composition chart (coarse residuals, "other", etc.).
FINE_ROCKS_FOR_COMPOSITION = frozenset([
    "anhydrite", "chalk", "clay",
    "claystone_cool", "claystone_hot",
    "dolomite", "halite_pure",
    "sandstone_clean", "sandstone_shaly",
])
# 13 named formations; the catch-all "other" formation is dropped.
NAMED_FORMATIONS_FOR_COMPOSITION = frozenset([
    "NU", "NM", "NL", "CK", "KN", "SL", "SG",
    "AT", "RN", "RB", "ZE", "RO", "DC",
])


def formation_composition_table(sim_by_fm: dict[str, Counter],
                                 real_by_fm: dict[str, Counter]
                                 ) -> pd.DataFrame:
    """Per (formation, rock) sim fraction vs real fraction, with TVD per formation.

    Restricted to the 13 named formations and the 9 fine rock classes;
    catch-all `"other"` formation and any non-fine rock are dropped."""
    rows = []
    formations = sorted(
        (set(sim_by_fm) | set(real_by_fm))
        & NAMED_FORMATIONS_FOR_COMPOSITION
    )
    for fm in formations:
        sim_rocks = {r: n for r, n in sim_by_fm[fm].items()
                     if r in FINE_ROCKS_FOR_COMPOSITION}
        real_rocks = {r: n for r, n in real_by_fm[fm].items()
                      if r in FINE_ROCKS_FOR_COMPOSITION}
        sim_total = sum(sim_rocks.values()) or 1
        real_total = sum(real_rocks.values()) or 1
        all_rocks = sorted(set(sim_rocks) | set(real_rocks))
        tvd_contrib = 0.0
        for r in all_rocks:
            sp = sim_rocks.get(r, 0) / sim_total
            rp = real_rocks.get(r, 0) / real_total
            rows.append({
                "formation":  fm,
                "rock":       r,
                "sim_pct":    round(100 * sp, 2),
                "real_pct":   round(100 * rp, 2),
                "abs_diff":   round(100 * abs(sp - rp), 2),
            })
            tvd_contrib += abs(sp - rp)
        rows.append({
            "formation": fm, "rock": "__TVD__",
            "sim_pct": None, "real_pct": None,
            "abs_diff": round(50 * tvd_contrib, 2),  # TVD in [0,100]%
        })
    return pd.DataFrame(rows)


def transition_matrix_from_counts(pairs: Counter) -> tuple[list[str], np.ndarray]:
    """Row-normalise a Counter({(a, b): n}) into a transition matrix.

    Replaces `transition_matrix_from_maps`: the per-map iteration is now
    done once inside streaming_aggregate, which yields the same Counter."""
    rocks_seen: set[str] = set()
    for a, b in pairs.keys():
        rocks_seen.add(a)
        rocks_seen.add(b)
    rocks = sorted(rocks_seen)
    n = len(rocks)
    M = np.zeros((n, n), dtype=np.float64)
    for (a, b), c in pairs.items():
        i, j = rocks.index(a), rocks.index(b)
        M[i, j] = c
    row_sum = M.sum(axis=1, keepdims=True)
    M_norm = np.divide(M, row_sum,
                        out=np.zeros_like(M), where=row_sum > 0)
    return rocks, M_norm


def transition_matrix_from_real(real_df: pd.DataFrame,
                                 formation: str) -> tuple[list[str], np.ndarray]:
    """Build a transition matrix from real NLOG data for one formation,
    using each well's 10m-resampled rock-type sequence."""
    nlog = real_df[(real_df["dataset"] == "NLOG")
                    & (real_df["formation"] == formation)]
    nlog = nlog.drop_duplicates(["borehole", "depth"])[
        ["borehole", "depth", "rock_type_fine"]
    ].dropna()
    if len(nlog) < 100:
        return [], np.zeros((0, 0))
    pairs = Counter()
    rocks_seen = set()
    for bh, grp in nlog.groupby("borehole"):
        g = grp.sort_values("depth")
        # bin depth at 10m intervals
        g["bin"] = (g["depth"] // 10).astype(int)
        binned = g.groupby("bin")["rock_type_fine"].first()
        seq = binned.tolist()
        for a, b in zip(seq[:-1], seq[1:]):
            pairs[(str(a), str(b))] += 1
            rocks_seen.add(str(a)); rocks_seen.add(str(b))
    rocks = sorted(rocks_seen)
    n = len(rocks)
    M = np.zeros((n, n))
    for (a, b), c in pairs.items():
        i, j = rocks.index(a), rocks.index(b)
        M[i, j] = c
    row_sum = M.sum(axis=1, keepdims=True)
    M_norm = np.divide(M, row_sum,
                        out=np.zeros_like(M), where=row_sum > 0)
    return rocks, M_norm


def transition_distance_table(transitions_by_fm: dict[str, Counter],
                              real_df: pd.DataFrame,
                              formations: list[str]) -> pd.DataFrame:
    """Per formation: Frobenius distance between sim and real transition
    matrices (aligned on the union of rocks observed in either).

    `transitions_by_fm` is pre-computed by streaming_aggregate."""
    rows = []
    for fm in formations:
        sim_pairs = transitions_by_fm.get(fm, Counter())
        if not sim_pairs:
            continue
        s_rocks, S = transition_matrix_from_counts(sim_pairs)
        r_rocks, R = transition_matrix_from_real(real_df, fm)
        if S.size == 0 or R.size == 0:
            continue
        union = sorted(set(s_rocks) | set(r_rocks))
        n = len(union)
        S_full = np.zeros((n, n))
        R_full = np.zeros((n, n))
        for i, a in enumerate(union):
            for j, b in enumerate(union):
                if a in s_rocks and b in s_rocks:
                    S_full[i, j] = S[s_rocks.index(a), s_rocks.index(b)]
                if a in r_rocks and b in r_rocks:
                    R_full[i, j] = R[r_rocks.index(a), r_rocks.index(b)]
        rows.append({
            "formation":          fm,
            "n_rocks":            n,
            "frobenius":          round(float(np.linalg.norm(S_full - R_full)),
                                         4),
            "n_sim_transitions":  int(sum(sim_pairs.values())),
            "n_real_transitions": int(R.sum() * 1),
        })
    return pd.DataFrame(rows).sort_values("frobenius")


# ─── plots ────────────────────────────────────────────────────────────────

def plot_distribution_bank_overview(bank: DistributionBank,
                                     out_path: Path,
                                     variable: str = "rhob") -> None:
    """KDE of `variable` for 8 representative cells — visualises what the
    simulator draws from."""
    # v3.1: probes are (rock, formation, bin_idx) -- 10 m bins.
    # bin = depth // 10 (e.g. 220 == 2200 m).
    # Poster: 8 representative cells in a 2x4 (wide) grid, full box width below
    # the text; big fonts so each KDE stays readable.
    key_cells = [
        ("claystone_hot",   "DC", 220),
        ("claystone_cool",  "KN", 180),
        ("sandstone_clean", "RO", 260),
        ("sandstone_shaly", "RB", 220),
        ("halite_pure",     "ZE", 260),
        ("anhydrite",       "ZE", 260),
        ("chalk",           "CK", 180),
        ("dolomite",        "ZE", 220),
    ]
    n_cols = 4
    fig, axes = plt.subplots(2, n_cols, figsize=(9.5, 4.4))
    axes = axes.flatten()
    for idx, (ax, key) in enumerate(zip(axes, key_cells)):
        cell = bank.cells.get(key)
        # backward-compat: legacy 2-tuple key for older rock-only banks
        if cell is None and len(key) == 3:
            cell = bank.cells.get((key[0], key[2]))
        if cell is None or variable not in getattr(cell, "kdes", {}):
            ax.set_visible(False)
            continue
        kde = cell.kdes[variable]
        support = cell.supports.get(variable)
        if support is None:
            arr = kde.dataset.ravel() if hasattr(kde, "dataset") else None
            support = (float(np.min(arr)), float(np.max(arr))) if arr is not None else (0, 1)
        x = np.linspace(support[0], support[1], 200)
        try:
            y = kde(x)
        except Exception:
            ax.set_visible(False)
            continue
        c = ROCK_COLOURS.get(key[0], "#2166ac")
        ax.fill_between(x, 0, y, alpha=0.45, color=c)
        ax.plot(x, y, color=c, lw=2.4)
        ax.set_title(f"{key[0]}\n{key[1]} · {cell.depth_lo:.0f}–{cell.depth_hi:.0f} m",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel(variable, fontsize=10)
        ax.tick_params(labelsize=8.5)
        if idx % n_cols == 0:
            ax.set_ylabel("density", fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(f"DistributionBank: empirical {variable} KDE, "
                 "8 representative (rock × depth) cells",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_marginal_sim_vs_real(sim: dict[str, dict[str, np.ndarray]],
                               real_df: pd.DataFrame,
                               variable: str,
                               out_path: Path,
                               target_rocks: list[str] | None = None) -> None:
    """6-panel grid: sim KDE vs real histogram per rock, for one variable."""
    if target_rocks is None:
        target_rocks = ["claystone_hot", "claystone_cool",
                         "sandstone_clean", "sandstone_shaly",
                         "halite_pure",     "chalk"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()
    for ax, rock in zip(axes, target_rocks):
        sv = sim.get(rock, {}).get(variable)
        rv = (real_df.loc[(real_df["rock_type_fine"] == rock)
                           & (real_df["measurement"] == variable),
                           "value"].dropna().to_numpy())
        if sv is None or len(sv) < 50 or len(rv) < 50:
            ax.set_visible(False)
            continue
        all_vals = np.concatenate([sv, rv])
        lo, hi = np.percentile(all_vals, [1, 99])
        bins = np.linspace(lo, hi, 60)
        ax.hist(rv, bins=bins, density=True, alpha=0.55,
                color="#b2182b", label="real",
                edgecolor="black", linewidth=0.3)
        ax.hist(sv, bins=bins, density=True, histtype="step",
                color="#2166ac", lw=2.2, ls="--", label="sim")
        ks_stat, _ = stats.ks_2samp(sv, rv)
        ax.set_title(f"{rock}  (KS {ks_stat:.2f})", fontsize=15,
                     fontweight="bold")
        ax.set_xlabel(variable, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=13)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density", fontsize=14)
    axes[3].set_ylabel("density", fontsize=14)
    fig.suptitle(f"Simulator vs real: per-rock {variable} marginals",
                 fontweight="bold", fontsize=19)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_formation_composition(sim_by_fm: dict[str, Counter],
                                real_by_fm: dict[str, Counter],
                                out_path: Path) -> None:
    """Per-formation grouped bars: real vs sim rock-fraction.

    Restricted to the 13 named formations and the 9 fine rock
    classes; the catch-all `"other"` formation and any non-fine
    rock label are dropped before plotting."""
    formations = [f for f in FORMATION_ORDER
                   if f in sim_by_fm and f in real_by_fm
                   and f in NAMED_FORMATIONS_FOR_COMPOSITION]
    # restrict to fine rock palette per formation
    sim_fine = {fm: {r: n for r, n in sim_by_fm[fm].items()
                     if r in FINE_ROCKS_FOR_COMPOSITION}
                for fm in formations}
    real_fine = {fm: {r: n for r, n in real_by_fm[fm].items()
                      if r in FINE_ROCKS_FOR_COMPOSITION}
                 for fm in formations}
    all_rocks = sorted(
        {r for fm in formations
         for r in set(sim_fine[fm]) | set(real_fine[fm])})

    # Right-side colour legend, and a hatch on the sim bars so real vs sim is
    # obvious at a glance.
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    positions = np.arange(len(formations))
    width = 0.42
    bottom_real = np.zeros(len(formations))
    bottom_sim = np.zeros(len(formations))
    for r in all_rocks:
        c = ROCK_COLOURS.get(r, "#999999")
        real_heights = []
        sim_heights = []
        for fm in formations:
            real_total = sum(real_fine[fm].values()) or 1
            sim_total = sum(sim_fine[fm].values()) or 1
            real_heights.append(real_fine[fm].get(r, 0) / real_total)
            sim_heights.append(sim_fine[fm].get(r, 0) / sim_total)
        ax.bar(positions - width/2, real_heights, width,
                bottom=bottom_real, color=c,
                edgecolor="black", linewidth=0.4)
        ax.bar(positions + width/2, sim_heights, width,
                bottom=bottom_sim, color=c, hatch="////",
                edgecolor="black", linewidth=0.4)
        bottom_real += np.array(real_heights)
        bottom_sim += np.array(sim_heights)
    ax.set_xticks(positions)
    ax.set_xticklabels(formations, fontsize=9.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("rock fraction", fontsize=11)
    ax.set_title("Rock composition per formation: left bar = real, "
                 "right bar = sim (hatched)", fontweight="bold", fontsize=12)
    # rock-type colour legend (vertical, on the right)
    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=ROCK_COLOURS.get(r, "#999999"))
                for r in all_rocks]
    ax.legend(handles, all_rocks, fontsize=8.5, loc="center left",
                bbox_to_anchor=(1.01, 0.5), handlelength=1.5,
                labelspacing=0.4, frameon=False, title="rock type",
                title_fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_transition_matrices(transitions_by_fm: dict[str, Counter],
                              real_df: pd.DataFrame,
                              out_path: Path,
                              formations: list[str] | None = None) -> None:
    """Side-by-side sim vs real transition heatmaps for 3 multi-rock
    formations. `transitions_by_fm` is pre-computed by
    streaming_aggregate."""
    if formations is None:
        # Two multi-rock, geologically distinct formations: a wide single row
        # of (real, sim) pairs keeps each heatmap big and the rock labels +
        # cell values clearly readable on the poster.
        formations = ["ZE", "RO"]
    # Build the (matrix, rock-union, title) panels: real then sim per formation.
    panels = []
    for fm in formations:
        sim_pairs = transitions_by_fm.get(fm, Counter())
        if sim_pairs:
            s_rocks, S = transition_matrix_from_counts(sim_pairs)
        else:
            s_rocks, S = [], np.zeros((0, 0))
        r_rocks, R = transition_matrix_from_real(real_df, fm)
        union = sorted(set(s_rocks) | set(r_rocks))
        if not union:
            continue
        idx_s = {r: s_rocks.index(r) if r in s_rocks else None for r in union}
        idx_r = {r: r_rocks.index(r) if r in r_rocks else None for r in union}
        m_s = np.zeros((len(union), len(union)))
        m_r = np.zeros((len(union), len(union)))
        for a in union:
            for b in union:
                ai, bi = union.index(a), union.index(b)
                if idx_r[a] is not None and idx_r[b] is not None:
                    m_r[ai, bi] = R[idx_r[a], idx_r[b]]
                if idx_s[a] is not None and idx_s[b] is not None:
                    m_s[ai, bi] = S[idx_s[a], idx_s[b]]
        panels.append((m_r, union, f"{fm}: real"))
        panels.append((m_s, union, f"{fm}: simulator"))

    n = len(panels)
    # 2x2 grid (real | simulator per formation) so each heatmap is large and
    # the rock labels and cell values are clearly readable.
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    im = None
    for k, (ax, (mat, union, title)) in enumerate(zip(axes, panels)):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(union)))
        ax.set_yticks(range(len(union)))
        ax.set_xticklabels(union, rotation=40, ha="right", fontsize=13)
        if k % 2 == 0:
            ax.set_yticklabels(union, fontsize=13)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=17, fontweight="bold")
        for (i, j2), v in np.ndenumerate(mat):
            if v >= 0.01:
                ax.text(j2, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=14, color="white" if v < 0.6 else "black")
    for ax in axes[n:]:
        ax.set_visible(False)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=12)
    fig.suptitle("Markov transition matrices: simulator vs real "
                 "(rows sum to 1)", fontweight="bold", fontsize=20)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_run_lengths(sim_runs: dict[str, list],
                      real_df: pd.DataFrame,
                      out_path: Path,
                      target_rocks: list[str] | None = None) -> None:
    """Sim vs real run-length histograms per rock (top 6 rocks).
    `sim_runs` is pre-computed by streaming_aggregate."""
    if target_rocks is None:
        target_rocks = ["claystone_hot", "sandstone_clean",
                         "halite_pure", "chalk", "anhydrite", "dolomite"]

    # real run lengths (per well, 10m-binned)
    nlog = real_df[real_df["dataset"] == "NLOG"].drop_duplicates(
        ["borehole", "depth"])
    real_runs = defaultdict(list)
    for bh, grp in nlog.groupby("borehole"):
        g = grp.sort_values("depth")
        g = g.assign(bin=(g["depth"] // 10).astype(int))
        binned = g.groupby("bin")["rock_type_fine"].first()
        seq = binned.tolist()
        if not seq:
            continue
        cur = seq[0]; run = 1
        for r in seq[1:]:
            if r == cur:
                run += 1
            else:
                if isinstance(cur, str):
                    real_runs[cur].append(run)
                cur = r; run = 1
        if isinstance(cur, str):
            real_runs[cur].append(run)

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2))
    axes = axes.flatten()
    for ax, rock in zip(axes, target_rocks):
        sr = np.array(sim_runs.get(rock, []), dtype=float)
        rr = np.array(real_runs.get(rock, []), dtype=float)
        sr = sr[sr > 0]; rr = rr[rr > 0]
        if len(sr) < 20 or len(rr) < 20:
            ax.set_visible(False)
            continue
        max_run = int(np.percentile(np.concatenate([sr, rr]), 99))
        bins = np.arange(1, max_run + 2)
        ax.hist(rr, bins=bins, density=True, alpha=0.55,
                color="#b2182b", label=f"real (med={int(np.median(rr))})",
                edgecolor="black", linewidth=0.3)
        ax.hist(sr, bins=bins, density=True, histtype="step",
                color="#2166ac", lw=1.8, ls="--",
                label=f"sim (med={int(np.median(sr))})")
        try:
            w = stats.wasserstein_distance(sr, rr)
        except Exception:
            w = float("nan")
        ax.set_title(f"{rock}\nWasserstein = {w:.2f}", fontsize=10)
        ax.set_xlabel("run length (cells, 10m each)")
        ax.set_ylabel("density")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.25)
    fig.suptitle("Run-length distributions — simulator vs real NLOG",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sample_map_cross_section(maps: list[dict],
                                    out_path: Path,
                                    ix: int = 16, iy: int = 16) -> None:
    """Single (x, y) column — rocks bar + 3 variables vs depth."""
    if not maps:
        return
    m = maps[0]
    depth_axis = m["depth_axis"]
    rocks = np.asarray(m["rock_types"][ix, iy, :]).astype(object)

    fig, axes = plt.subplots(1, 4, figsize=(14, 9),
                              gridspec_kw=dict(width_ratios=[0.6, 1, 1, 1]),
                              sharey=True)
    ax_rock = axes[0]
    for i, r in enumerate(rocks):
        col = ROCK_COLOURS.get(str(r), "#dddddd")
        ax_rock.add_patch(plt.Rectangle(
            (0, depth_axis[i]), 1, depth_axis[1] - depth_axis[0],
            color=col, edgecolor="none"))
    ax_rock.set_xlim(0, 1)
    ax_rock.set_ylim(depth_axis.max(), depth_axis.min())
    ax_rock.set_ylabel("depth (m)")
    ax_rock.set_xticks([])
    ax_rock.set_title("rock type")
    # legend
    seen = sorted(set(str(r) for r in rocks))
    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=ROCK_COLOURS.get(r, "#dddddd"))
                for r in seen]
    ax_rock.legend(handles, seen, fontsize=11,
                    loc="upper right", bbox_to_anchor=(0, 1))

    for ax, var in zip(axes[1:], ["rhob", "gr_api", "dt_us_ft"]):
        if var not in m["variables"]:
            ax.set_visible(False)
            continue
        vals = m["variables"][var][ix, iy, :]
        ax.plot(vals, depth_axis, lw=0.8, color="#2166ac")
        ax.set_xlabel(var)
        ax.grid(alpha=0.25)
        ax.set_title(var)

    fig.suptitle(f"Sample simulated column ({ix}, {iy})",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_lateral_coherence(maps: list[dict], out_path: Path,
                            y_fixed: int = 16, n_strips: int = 32) -> None:
    """Show lateral variability along x at fixed y on the first map."""
    if not maps:
        return
    m = maps[0]
    rt = np.asarray(m["rock_types"])
    depth_axis = m["depth_axis"]
    nx, ny, nz = rt.shape
    xs = np.linspace(0, nx - 1, n_strips).astype(int)
    img = np.empty((nz, n_strips, 4))
    for k, x in enumerate(xs):
        col = rt[x, y_fixed, :].astype(object)
        for d in range(nz):
            img[d, k, :] = mcolors.to_rgba(
                ROCK_COLOURS.get(str(col[d]), "#dddddd"))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img, aspect="auto",
              extent=(0, n_strips, depth_axis[-1], depth_axis[0]))
    ax.set_xlabel(f"x cell at y={y_fixed}")
    ax.set_ylabel("depth (m)")
    ax.set_title("Lateral coherence — one simulated map at fixed y",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_metric_summary(marginal: pd.DataFrame,
                         transitions: pd.DataFrame,
                         out_path: Path) -> None:
    """Heatmap-style summary of KS + Wasserstein per (rock × variable),
    plus transition Frobenius per formation as a side panel."""
    # Poster: a single wide KS heatmap (transition distances now live in their
    # own figure), with large cell annotations + axis labels. Channels are the
    # rows and rocks the columns so the grid is wide (spans the full box width
    # at a short height).
    # Smaller native size + larger cell/axis fonts so the in-cell numbers render
    # near the body-text size when the figure is displayed (~19 cm wide).
    fig, ax1 = plt.subplots(figsize=(8.0, 3.2))
    if len(marginal):
        piv = marginal.pivot(index="variable", columns="rock", values="ks_stat")
        variables = piv.index.tolist()
        rocks = piv.columns.tolist()
        im = ax1.imshow(piv.values, aspect="auto", cmap="RdYlGn_r",
                         vmin=0, vmax=0.5)
        ax1.set_xticks(range(len(rocks)))
        ax1.set_xticklabels(rocks, rotation=25, ha="right", fontsize=9)
        ax1.set_yticks(range(len(variables)))
        ax1.set_yticklabels(variables, fontsize=9.5)
        for (i, j), val in np.ndenumerate(piv.values):
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=8, fontweight="bold",
                          color="black" if val < 0.3 else "white")
        ax1.set_title("KS distance: simulator vs real "
                      "(lower = greener = closer)",
                      fontweight="bold", fontsize=11)
        cbar = plt.colorbar(im, ax=ax1, fraction=0.026, pad=0.015)
        cbar.ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── report ──────────────────────────────────────────────────────────────

def write_simulation_report(marginal: pd.DataFrame,
                             composition: pd.DataFrame,
                             transitions: pd.DataFrame,
                             n_maps: int,
                             out_path: Path) -> None:
    L = []
    a = L.append
    a("# Simulator validation report")
    a("")
    a(f"Compared **{n_maps} synthetic maps** against `data/clean/samples.parquet` "
      "across four axes: petrophysical marginals, formation composition, "
      "vertical sequence statistics, and spatial structure.")
    a("")
    a("## 1. DistributionBank — what the simulator draws from")
    a("")
    a("![Distribution bank](01_distribution_bank_overview.png)")
    a("")
    a("**Figure 1** — empirical KDE of `rhob` for 8 representative "
      "(rock × depth-bin) cells.  The simulator samples directly from "
      "these KDEs; their shapes therefore *are* the simulator's "
      "petrophysical priors.")
    a("")
    a("## 2. Per-rock marginals — sim vs real")
    a("")
    a("![rhob](02_marginal_sim_vs_real_rhob.png)")
    a("")
    a("![gr](03_marginal_sim_vs_real_gr.png)")
    a("")
    if len(marginal):
        worst = marginal.sort_values("ks_stat", ascending=False).head(5)
        a("**Worst-fit (rock × variable)** by KS statistic:")
        a("")
        a("| rock | variable | KS | Wasserstein | sim p50 | real p50 |")
        a("|---|---|---|---|---|---|")
        for _, r in worst.iterrows():
            a(f"| {r['rock']} | {r['variable']} | {r['ks_stat']} | "
              f"{r['wasserstein']} | {r['sim_p50']} | {r['real_p50']} |")
        a("")
        a("(KS < 0.10 = excellent fit, < 0.30 = good, > 0.40 = systematic "
          "mismatch worth investigating.)")
    a("")
    a("## 3. Formation composition — sim vs real")
    a("")
    a("![composition](04_formation_composition.png)")
    a("")
    if len(composition):
        tvd = composition[composition["rock"] == "__TVD__"][
            ["formation", "abs_diff"]
        ].rename(columns={"abs_diff": "TVD_pct"})
        if len(tvd):
            a("Total-variation distance per formation (lower = closer "
              "match to real corpus composition):")
            a("")
            a("| formation | TVD (%) |")
            a("|---|---|")
            for _, r in tvd.sort_values("TVD_pct").iterrows():
                a(f"| {r['formation']} | {r['TVD_pct']:.1f}% |")
            a("")
    a("## 4. Vertical sequences — Markov transition matrices")
    a("")
    a("![transitions](05_transition_matrices.png)")
    a("")
    a("![run lengths](06_run_length_distributions.png)")
    a("")
    if len(transitions):
        a("Frobenius distance between sim and real transition matrices per "
          "formation (aligned on the union of observed rocks):")
        a("")
        a("| formation | Frobenius | n rocks |")
        a("|---|---|---|")
        for _, r in transitions.sort_values("frobenius").iterrows():
            a(f"| {r['formation']} | {r['frobenius']} | {r['n_rocks']} |")
        a("")
    a("## 5. Spatial structure")
    a("")
    a("![sample column](07_sample_map_cross_section.png)")
    a("")
    a("![lateral coherence](08_lateral_coherence.png)")
    a("")
    a("Figure 7 shows one (x, y) column with its rock sequence and three "
      "variable curves.  Figure 8 shows lateral variability across the "
      "x-axis at fixed y — formation boundaries should wiggle smoothly "
      "(thanks to the anisotropic Gaussian random field used as boundary "
      "perturbation), not jump.")
    a("")
    a("## 6. Summary")
    a("")
    a("![summary](09_metric_summary.png)")
    a("")
    a("CSV tables alongside this report contain the full per-cell numbers:")
    a("")
    a("- `per_rock_marginal_distances.csv` — KS + Wasserstein per (rock × variable)")
    a("- `per_formation_composition.csv` — sim/real rock-fractions + TVD")
    a("- `per_formation_transition_distance.csv` — Frobenius per formation")
    a("")
    out_path.write_text("\n".join(L))
    print(f"  wrote {out_path}")


# ─── orchestration ────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--geometry", type=Path, default=DEFAULT_GEOM)
    ap.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                     help="if present and label-augmented, load the first "
                          "--n-maps maps from here instead of generating fresh")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-maps", type=int, default=DEFAULT_N_MAPS,
                     help=f"number of maps to analyse (default {DEFAULT_N_MAPS})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force-generate", action="store_true",
                     help="ignore saved dataset and always generate fresh maps")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if (not args.force_generate
            and args.dataset_dir is not None
            and dataset_has_labels(args.dataset_dir)):
        print(f"loading first {args.n_maps} maps from {args.dataset_dir} ...")
        maps = load_sim_maps_from_disk(args.dataset_dir, args.n_maps,
                                        TARGET_VARS)
        bank = DistributionBank.load(args.bank)
    else:
        if args.dataset_dir.exists() and not dataset_has_labels(args.dataset_dir):
            print(f"  note: {args.dataset_dir} has no labels_vocab.pkl — "
                  "saved maps don't carry rock labels.  Falling back to "
                  "on-the-fly generation.")
        print(f"loading simulator artifacts ...")
        bank  = DistributionBank.load(args.bank)
        geom  = FormationGeometry.load(args.geometry)
        prior = DiscoveryPrior.load(args.prior)
        print(f"\ngenerating {args.n_maps} synthetic maps "
              f"(single-threaded, seed={args.seed}) ...")
        maps = generate_sim_maps(bank, geom, prior, args.n_maps, args.seed)

    print(f"\nloading real corpus from {args.samples} ...")
    real_df = pd.read_parquet(args.samples)
    real_df = real_df.drop_duplicates(["borehole", "depth", "measurement"])
    print(f"  {len(real_df):,} real rows")

    print("\nstreaming aggregation over maps (single pass) ...")
    agg = streaming_aggregate(maps, TARGET_VARS)
    sim_per_rock      = {r: vd for r, vd in agg["sim_values"].items()
                         if r in FINE_ROCKS_FOR_COMPOSITION}
    sim_by_fm         = agg["sim_by_fm"]
    transitions_by_fm = agg["transitions_by_fm"]
    sim_runs          = {r: rs for r, rs in agg["run_lengths"].items()
                         if r in FINE_ROCKS_FOR_COMPOSITION}
    first_map         = agg["first_map"]
    dropped_marg = set(agg["sim_values"]) - set(sim_per_rock)
    dropped_runs = set(agg["run_lengths"]) - set(sim_runs)
    if dropped_marg or dropped_runs:
        print(f"  dropped non-fine rock labels from charts: "
              f"marginals={sorted(dropped_marg)}, "
              f"run-lengths={sorted(dropped_runs)}")
    print(f"  {len(sim_per_rock)} fine rock types; "
          f"{len(sim_by_fm)} formations seen; "
          f"{sum(len(v) for v in sim_runs.values())} run-length samples")
    del maps  # release the streaming-list reference

    print("\ncomputing per-marginal distances ...")
    marginal = marginal_distances(sim_per_rock, real_df, TARGET_VARS)
    marginal.to_csv(args.out / "per_rock_marginal_distances.csv", index=False)
    print(f"  wrote per_rock_marginal_distances.csv  ({len(marginal)} rows)")

    print("\ncomputing per-formation composition ...")
    real_by_fm = pool_real_formation_rock(real_df)
    composition = formation_composition_table(sim_by_fm, real_by_fm)
    composition.to_csv(args.out / "per_formation_composition.csv", index=False)
    print(f"  wrote per_formation_composition.csv  ({len(composition)} rows)")

    print("\ncomputing transition-matrix distances ...")
    target_fms = ["ZE", "RO", "RB", "CK", "KN", "DC", "AT", "SL", "RN"]
    transitions = transition_distance_table(transitions_by_fm, real_df,
                                              target_fms)
    transitions.to_csv(args.out
                        / "per_formation_transition_distance.csv", index=False)
    print(f"  wrote per_formation_transition_distance.csv  "
          f"({len(transitions)} rows)")

    print("\nplots ...")
    p = args.out
    plot_distribution_bank_overview(bank,        p / "01_distribution_bank_overview.png")
    plot_marginal_sim_vs_real(sim_per_rock, real_df, "rhob",
                                p / "02_marginal_sim_vs_real_rhob.png")
    plot_marginal_sim_vs_real(sim_per_rock, real_df, "gr_api",
                                p / "03_marginal_sim_vs_real_gr.png")
    plot_formation_composition(sim_by_fm, real_by_fm,
                                p / "04_formation_composition.png")
    plot_transition_matrices(transitions_by_fm, real_df,
                              p / "05_transition_matrices.png")
    plot_run_lengths(sim_runs, real_df,
                      p / "06_run_length_distributions.png")
    first_map_list = [first_map] if first_map is not None else []
    plot_sample_map_cross_section(first_map_list,
                                    p / "07_sample_map_cross_section.png")
    plot_lateral_coherence(first_map_list,
                            p / "08_lateral_coherence.png")
    plot_metric_summary(marginal, transitions,
                         p / "09_metric_summary.png")

    print("\nwriting SIMULATION_REPORT.md ...")
    write_simulation_report(marginal, composition, transitions,
                              n_maps=args.n_maps,
                              out_path=p / "SIMULATION_REPORT.md")

    print(f"\n→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()
