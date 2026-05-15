"""Quantitative validation of the simulator vs real NLOG data (Task 3).

Four checks, all per-formation, with quantitative distance measures:

  1. Run-length distributions (per formation, per rock)
     - 1-D Wasserstein distance, in cells of 10 m
     - histogram plot per formation, one panel per rock

  2. Facies fractions per formation
     - per-well composition vectors; mean + 90 % interval
     - bar plot real vs sim per rock, per formation

  3. Transition matrices vs the Task-2 fitted ones
     - Frobenius norm of (P_sim - P_real_fit)
     - KL divergence per row
     - side-by-side heatmaps per formation

  4. Petrophysical marginals per (rock, depth_bin, variable)
     - KS distance between real and sim samples for each cell
     - summary table sorted by largest KS distance

Outputs go to plots/validation/.  A one-page summary lands in
plots/validation/REPORT.md, listing the worst 10 (formation, rock) and
worst 10 (rock, bin, var) cells plus pass/fail flags against simple
empirical thresholds.

Run:
    python scripts/validate_simulator.py
    python scripts/validate_simulator.py --n-maps 10 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.distributions import DistributionBank, DiscoveryPrior, HARD_BOUNDS
from simulator.formation_geometry import (
    FormationGeometry,
    _fit_transition_matrix,
    TRANSITION_BIN_STEP_M,
    MIN_WELLS_PER_BASIN_FOR_STRATIFICATION,
)
from simulator.map_generator import MapGenerator, SimConfig


OUT_DIR = Path("plots/validation")
REPORT_PATH = OUT_DIR / "REPORT.md"

DEPTH_BINS = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200,
              3600, 4000, 4400, 4800, 5200, 5600, 6000]

# Pass/fail thresholds (empirical, tune as more reports accumulate)
WASSERSTEIN_FAIL_CELLS = 30.0      # 300 m
KS_FAIL = 0.30
MIN_SAMPLES_FOR_TEST = 30
MIN_RUNS_FOR_WS = 5


# --------------------------------------------------------------------------
# data extraction
# --------------------------------------------------------------------------
def extract_runs_from_sequence(rocks: list[str]) -> list[tuple[str, int]]:
    """Return [(rock, length_in_cells), ...] for one sequence."""
    if not rocks:
        return []
    runs: list[tuple[str, int]] = []
    cur = rocks[0]
    n = 1
    for r in rocks[1:]:
        if r == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur = r
            n = 1
    runs.append((cur, n))
    return runs


def real_per_formation_columns(
    df: pd.DataFrame,
    bin_step_m: float = TRANSITION_BIN_STEP_M,
) -> dict[str, list[list[str]]]:
    """For each formation, list of per-well rock sequences (10 m grid)."""
    out: dict[str, list[list[str]]] = defaultdict(list)
    work = df[["borehole", "formation", "depth", "rock_type_fine"]].copy()
    work["rock_type_fine"] = work["rock_type_fine"].astype(str)
    work["formation"] = work["formation"].astype(str)
    work = work.sort_values(["borehole", "depth"], kind="mergesort")
    work["_bin"] = np.floor(work["depth"].to_numpy() / bin_step_m).astype(np.int64)
    work = work.drop_duplicates(["borehole", "_bin"], keep="first")
    for (borehole, fm), wf in work.groupby(["borehole", "formation"]):
        if len(wf) < 2:
            continue
        rocks = wf["rock_type_fine"].tolist()
        if rocks:
            out[fm].append(rocks)
    return out


def sim_per_formation_columns(
    sim_maps: list[dict],
) -> dict[str, list[list[str]]]:
    """Same shape as real_per_formation_columns but extracted from sim."""
    out: dict[str, list[list[str]]] = defaultdict(list)
    for m in sim_maps:
        rt = m["rock_types"]           # (nx, ny, nz) object
        fm = m["formations"]           # (nx, ny, nz) object
        nx, ny, nz = rt.shape
        for x in range(nx):
            for y in range(ny):
                f_col = fm[x, y, :].astype(str)
                r_col = rt[x, y, :].astype(str)
                # split into per-formation contiguous slices
                if nz == 0:
                    continue
                start = 0
                cur_fm = f_col[0]
                for k in range(1, nz):
                    if f_col[k] != cur_fm:
                        if cur_fm and cur_fm != "other":
                            out[cur_fm].append(r_col[start:k].tolist())
                        start = k
                        cur_fm = f_col[k]
                if cur_fm and cur_fm != "other":
                    out[cur_fm].append(r_col[start:nz].tolist())
    return out


def sim_to_dataframe(
    sim_maps: list[dict],
    bin_step_m: float = TRANSITION_BIN_STEP_M,
) -> pd.DataFrame:
    """Flatten sim maps to a long DataFrame mimicking the real parquet
    columns we need: [borehole, depth, formation, rock_type_fine].
    Used for re-fitting the simulator's transition matrices via the
    same code path as the real fit (Task 2).
    """
    rows = []
    for m_idx, m in enumerate(sim_maps):
        rt = m["rock_types"]
        fm = m["formations"]
        depth_axis = m["depth_axis"]
        nx, ny, _ = rt.shape
        for x in range(nx):
            for y in range(ny):
                bh = f"sim_{m_idx:04d}_x{x:02d}_y{y:02d}"
                rows.append(pd.DataFrame({
                    "borehole": bh,
                    "depth": depth_axis,
                    "formation": fm[x, y, :].astype(str),
                    "rock_type_fine": rt[x, y, :].astype(str),
                }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# check 1: run-length Wasserstein
# --------------------------------------------------------------------------
def collect_runs(
    cols_by_fm: dict[str, list[list[str]]],
) -> dict[tuple[str, str], list[int]]:
    """(formation, rock) -> list of run lengths."""
    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    for fm, cols in cols_by_fm.items():
        for col in cols:
            for r, n in extract_runs_from_sequence(col):
                out[(fm, r)].append(n)
    return out


def check_run_lengths(
    real_cols: dict[str, list[list[str]]],
    sim_cols: dict[str, list[list[str]]],
    formations: list[str],
) -> pd.DataFrame:
    real_runs = collect_runs(real_cols)
    sim_runs = collect_runs(sim_cols)
    rows = []
    for (fm, rock), real_lengths in real_runs.items():
        if fm not in formations:
            continue
        sim_lengths = sim_runs.get((fm, rock), [])
        n_real = len(real_lengths)
        n_sim = len(sim_lengths)
        if n_real < MIN_RUNS_FOR_WS or n_sim < MIN_RUNS_FOR_WS:
            continue
        d = float(wasserstein_distance(real_lengths, sim_lengths))
        rows.append({
            "fm": fm, "rock": rock,
            "n_real_runs": n_real, "n_sim_runs": n_sim,
            "real_mean_cells": float(np.mean(real_lengths)),
            "sim_mean_cells": float(np.mean(sim_lengths)),
            "wasserstein_cells": d,
        })
    df = pd.DataFrame(rows).sort_values("wasserstein_cells", ascending=False)
    return df


def plot_run_length_histograms(
    real_cols: dict[str, list[list[str]]],
    sim_cols: dict[str, list[list[str]]],
    formations: list[str],
    out_dir: Path,
) -> None:
    real_runs = collect_runs(real_cols)
    sim_runs = collect_runs(sim_cols)
    for fm in formations:
        rocks = sorted({rk for (f, rk) in real_runs if f == fm})
        if not rocks:
            continue
        n = len(rocks)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows),
                                 squeeze=False)
        for i, rock in enumerate(rocks):
            ax = axes[i // cols, i % cols]
            real_lengths = real_runs.get((fm, rock), [])
            sim_lengths = sim_runs.get((fm, rock), [])
            if not real_lengths and not sim_lengths:
                continue
            hi = int(max([1] + real_lengths + sim_lengths) * 1.05)
            bins = np.linspace(0, hi, min(40, hi))
            if real_lengths:
                ax.hist(real_lengths, bins=bins, alpha=0.5, density=True,
                        label=f"real (n={len(real_lengths)})")
            if sim_lengths:
                ax.hist(sim_lengths, bins=bins, alpha=0.5, density=True,
                        label=f"sim  (n={len(sim_lengths)})")
            ax.set_title(rock)
            ax.set_xlabel("run length (cells, 10m)")
            ax.legend(fontsize=8)
        for j in range(n, rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle(f"{fm}: run-length distributions")
        fig.tight_layout()
        fig.savefig(out_dir / f"runlength_{fm}.png", dpi=110)
        plt.close(fig)


# --------------------------------------------------------------------------
# check 2: facies fractions per formation
# --------------------------------------------------------------------------
def well_fraction(rocks: list[str]) -> dict[str, float]:
    if not rocks:
        return {}
    total = len(rocks)
    out: dict[str, float] = defaultdict(int)
    for r in rocks:
        out[r] += 1
    return {r: c / total for r, c in out.items()}


def check_facies_fractions(
    real_cols: dict[str, list[list[str]]],
    sim_cols: dict[str, list[list[str]]],
    formations: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for fm in formations:
        rocks_set = set()
        real_fracs = [well_fraction(c) for c in real_cols.get(fm, [])]
        sim_fracs = [well_fraction(c) for c in sim_cols.get(fm, [])]
        if not real_fracs or not sim_fracs:
            continue
        for fr in real_fracs + sim_fracs:
            rocks_set |= fr.keys()
        rocks_list = sorted(rocks_set)
        # build (n_wells, n_rocks) arrays
        real_arr = np.zeros((len(real_fracs), len(rocks_list)))
        sim_arr = np.zeros((len(sim_fracs), len(rocks_list)))
        for j, r in enumerate(rocks_list):
            for i, fr in enumerate(real_fracs):
                real_arr[i, j] = fr.get(r, 0.0)
            for i, fr in enumerate(sim_fracs):
                sim_arr[i, j] = fr.get(r, 0.0)
        for j, r in enumerate(rocks_list):
            rows.append({
                "fm": fm, "rock": r,
                "real_mean": float(real_arr[:, j].mean()),
                "real_p5": float(np.percentile(real_arr[:, j], 5)),
                "real_p95": float(np.percentile(real_arr[:, j], 95)),
                "sim_mean": float(sim_arr[:, j].mean()),
                "sim_p5": float(np.percentile(sim_arr[:, j], 5)),
                "sim_p95": float(np.percentile(sim_arr[:, j], 95)),
                "abs_mean_diff": abs(float(real_arr[:, j].mean()
                                           - sim_arr[:, j].mean())),
            })
        # plot box plot per fm
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(rocks_list) + 2), 4))
        positions = np.arange(len(rocks_list))
        bp_real = ax.boxplot(
            [real_arr[:, j] for j in range(len(rocks_list))],
            positions=positions - 0.18, widths=0.32, patch_artist=True,
            boxprops=dict(facecolor="#3b6aa0"), medianprops=dict(color="white"),
        )
        bp_sim = ax.boxplot(
            [sim_arr[:, j] for j in range(len(rocks_list))],
            positions=positions + 0.18, widths=0.32, patch_artist=True,
            boxprops=dict(facecolor="#c97c2e"), medianprops=dict(color="white"),
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(rocks_list, rotation=45, ha="right")
        ax.set_ylabel("fraction of well")
        ax.set_title(f"{fm}: facies fractions (blue=real, orange=sim)")
        ax.set_xlim(-0.6, len(rocks_list) - 0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"facies_fractions_{fm}.png", dpi=110)
        plt.close(fig)
    return pd.DataFrame(rows).sort_values("abs_mean_diff", ascending=False)


# --------------------------------------------------------------------------
# check 3: transition matrices
# --------------------------------------------------------------------------
def _row_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def check_transition_matrices(
    geom: FormationGeometry,
    sim_df: pd.DataFrame,
    formations: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for fm in formations:
        stats = geom.formations.get(fm)
        if stats is None or stats.transition_matrix is None:
            continue
        P_real = stats.transition_matrix
        sub = sim_df[sim_df["formation"] == fm]
        if sub.empty:
            continue
        P_sim, n_sim_trans = _fit_transition_matrix(
            sub, rock_set=set(P_real.index)
        )
        # align rows/cols to real index
        P_sim = P_sim.reindex(index=P_real.index, columns=P_real.columns,
                              fill_value=1.0 / len(P_real.index))
        diff = (P_sim.values - P_real.values)
        frob = float(np.linalg.norm(diff))
        per_row_kl = np.array([_row_kl(P_real.values[i], P_sim.values[i])
                               for i in range(len(P_real.index))])
        max_kl = float(per_row_kl.max())
        rows.append({
            "fm": fm,
            "n_real_trans": stats.n_transitions,
            "n_sim_trans": n_sim_trans,
            "frobenius": frob,
            "max_row_kl": max_kl,
            "mean_row_kl": float(per_row_kl.mean()),
        })
        # heatmap real / sim / diff
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, M, t in zip(
            axes,
            [P_real.values, P_sim.values, diff],
            [f"{fm} real (n={stats.n_transitions:,})",
             f"{fm} sim  (n={n_sim_trans:,})",
             f"{fm} sim - real (Frob={frob:.2f})"],
        ):
            vmax = 1.0 if M is not diff else float(np.max(np.abs(diff)))
            vmin = 0.0 if M is not diff else -vmax
            cmap = "viridis" if M is not diff else "RdBu_r"
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(P_real.index)))
            ax.set_yticks(range(len(P_real.index)))
            ax.set_xticklabels(P_real.index, rotation=45, ha="right",
                               fontsize=8)
            ax.set_yticklabels(P_real.index, fontsize=8)
            ax.set_title(t)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"transition_compare_{fm}.png", dpi=110)
        plt.close(fig)
    return pd.DataFrame(rows).sort_values("frobenius", ascending=False)


# --------------------------------------------------------------------------
# check 4: petrophysical marginals KS per (rock, depth_bin, var)
# --------------------------------------------------------------------------
def check_petrophysical_marginals(
    real_df: pd.DataFrame,
    sim_maps: list[dict],
    variables: list[str],
    depth_bins: list[float],
) -> pd.DataFrame:
    # real parquet is long-form: one row per (well, depth, measurement),
    # value column.  Filter to variables of interest, then bin by depth.
    real = real_df[
        real_df["measurement"].isin(variables)
    ][["depth", "rock_type_fine", "measurement", "value"]].copy()
    real["rock_type_fine"] = real["rock_type_fine"].astype(str)
    real["measurement"] = real["measurement"].astype(str)
    real["bin"] = pd.cut(real["depth"], bins=depth_bins,
                         labels=range(len(depth_bins) - 1))

    # sim is wide-form (one array per variable); flatten and bin
    sim_pieces = []
    for m in sim_maps:
        rt_flat = m["rock_types"].reshape(-1).astype(str)
        depth_axis = m["depth_axis"]
        nx, ny, _ = m["rock_types"].shape
        depth_col = np.tile(depth_axis, nx * ny)
        for v in variables:
            sim_pieces.append(pd.DataFrame({
                "depth": depth_col,
                "rock_type_fine": rt_flat,
                "measurement": v,
                "value": m["variables"][v].reshape(-1),
            }))
    sim = pd.concat(sim_pieces, ignore_index=True)
    sim["bin"] = pd.cut(sim["depth"], bins=depth_bins,
                        labels=range(len(depth_bins) - 1))

    rows = []
    keys = ["rock_type_fine", "bin", "measurement"]
    for (rock, b, var), real_grp in real.groupby(keys, observed=True):
        sim_grp = sim[(sim["rock_type_fine"] == rock)
                      & (sim["bin"] == b)
                      & (sim["measurement"] == var)]
        if sim_grp.empty:
            continue
        r_vals = real_grp["value"].dropna().to_numpy()
        s_vals = sim_grp["value"].dropna().to_numpy()
        if len(r_vals) < MIN_SAMPLES_FOR_TEST or len(s_vals) < MIN_SAMPLES_FOR_TEST:
            continue
        stat, _ = ks_2samp(r_vals, s_vals)
        rows.append({
            "rock": rock, "bin": int(b), "var": var,
            "n_real": int(len(r_vals)),
            "n_sim": int(len(s_vals)),
            "ks": float(stat),
        })
    return pd.DataFrame(rows).sort_values("ks", ascending=False)


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, n: int, fmt: dict | None = None) -> str:
    """Render top-n rows of a DataFrame as a GitHub-flavoured markdown
    table.  Hand-rolled so we don't pull in `tabulate` as a dependency.
    """
    if df.empty:
        return "_(no rows)_\n"
    sub = df.head(n).copy()
    if fmt:
        for col, spec in fmt.items():
            if col in sub.columns:
                sub[col] = sub[col].map(
                    lambda v: format(v, spec)
                    if isinstance(v, (int, float)) and not pd.isna(v) else v
                )
    cols = list(sub.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row[c]) for c in cols) + " |"
        for _, row in sub.iterrows()
    )
    return "\n".join([header, sep, body]) + "\n"


def write_report(
    n_maps: int,
    ws_df: pd.DataFrame,
    frac_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    out_path: Path,
    bank: DistributionBank | None = None,
    geom: FormationGeometry | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Simulator validation report\n")
    lines.append(f"_Generated from {n_maps} simulator maps "
                 f"(32×32×440 cells each) vs the full NLOG corpus._\n")

    # pass/fail flags
    n_ws_fail = int((ws_df["wasserstein_cells"] > WASSERSTEIN_FAIL_CELLS).sum()) if not ws_df.empty else 0
    n_ks_fail = int((ks_df["ks"] > KS_FAIL).sum()) if not ks_df.empty else 0
    lines.append("## Pass / fail summary\n")
    lines.append(f"- Run-length Wasserstein > {WASSERSTEIN_FAIL_CELLS:.0f} cells "
                 f"(formation, rock) pairs: **{n_ws_fail}**\n")
    lines.append(f"- KS > {KS_FAIL:.2f} (rock, depth_bin, variable) cells: "
                 f"**{n_ks_fail}**\n")
    if not trans_df.empty:
        worst_frob = trans_df.iloc[0]
        lines.append(f"- Worst transition-matrix Frobenius: "
                     f"**{worst_frob['frobenius']:.2f}** ({worst_frob['fm']})\n")

    lines.append("\n## 1. Run-length distributions — worst 10 by Wasserstein\n")
    lines.append(_md_table(ws_df, 10, fmt={
        "real_mean_cells": ".1f", "sim_mean_cells": ".1f",
        "wasserstein_cells": ".2f",
    }))

    lines.append("\n## 2. Facies fractions — worst 10 by |mean diff|\n")
    lines.append(_md_table(frac_df, 10, fmt={
        "real_mean": ".3f", "real_p5": ".3f", "real_p95": ".3f",
        "sim_mean": ".3f", "sim_p5": ".3f", "sim_p95": ".3f",
        "abs_mean_diff": ".3f",
    }))

    lines.append("\n## 3. Transition matrices — Frobenius + KL per formation\n")
    lines.append(_md_table(trans_df, 50, fmt={
        "frobenius": ".3f", "max_row_kl": ".3f", "mean_row_kl": ".3f",
    }))

    lines.append("\n## 4. Petrophysical marginals — worst 10 by KS\n")
    lines.append(_md_table(ks_df, 10, fmt={"ks": ".3f"}))

    if bank is not None and bank.empirical_bounds:
        lines.append("\n## 5. Variable bounds — empirical vs hard-set (Task 5a)\n")
        bounds_rows = []
        for v in sorted(bank.empirical_bounds):
            elo, ehi = bank.empirical_bounds[v]
            hlo, hhi = HARD_BOUNDS.get(v, (float("nan"), float("nan")))
            bounds_rows.append({
                "var": v,
                "hard_lo": f"{hlo:.3f}", "hard_hi": f"{hhi:.3f}",
                "empirical_lo": f"{elo:.3f}", "empirical_hi": f"{ehi:.3f}",
                "tighter_lo_by": f"{elo - hlo:+.3f}",
                "tighter_hi_by": f"{ehi - hhi:+.3f}",
            })
        lines.append(_md_table(pd.DataFrame(bounds_rows), 50))

    if geom is not None and geom.combinations_by_basin:
        lines.append("\n## 6. Basin coverage of the combination pool (Task 5b)\n")
        lines.append(f"_k-means k={len(geom.basin_centers)} on (x_rd, y_rd) of "
                     f"the {len(geom.basin_labels_per_well):,} NLOG wells with "
                     f"coords. Basins below {MIN_WELLS_PER_BASIN_FOR_STRATIFICATION} "
                     "wells are dropped from stratified sampling._\n")
        n_pool = sum(sum(c for _, c in combos)
                     for combos in geom.combinations_by_basin.values())
        basin_rows = []
        for b in sorted(geom.combinations_by_basin):
            wells = sum(c for _, c in geom.combinations_by_basin[b])
            n_combos = len(geom.combinations_by_basin[b])
            basin_rows.append({
                "basin": int(b),
                "centroid_x_rd_km": f"{geom.basin_centers[b, 0]/1e3:.0f}",
                "centroid_y_rd_km": f"{geom.basin_centers[b, 1]/1e3:.0f}",
                "wells_in_pool": wells,
                "pool_pct": f"{100 * wells / max(n_pool, 1):.1f}%",
                "uniform_pct": f"{100/len(geom.combinations_by_basin):.1f}%",
                "unique_combinations": n_combos,
            })
        lines.append(_md_table(pd.DataFrame(basin_rows), 50))

    lines.append("\n## Caveats\n")
    lines.append(
        "- **Effective sample size for thin formations.** The simulator "
        "draws ONE base column per map, then produces the (32×32) grid by "
        "spatially wiggling that column's layer boundaries.  So a formation "
        "appearing in N maps' chosen combinations contributes N independent "
        "Markov-chain realisations, replicated 1024 times each.  For thin "
        "formations (AT, RN, SG, SL) that only appear in a few combinations, "
        "facies-fraction and transition-matrix metrics with `--n-maps 5–10` "
        "are dominated by which 1–3 base realisations got drawn.  Increase "
        "`--n-maps` to 30+ for stable per-formation marginals; the "
        "run-length distributions converge faster because they aggregate "
        "over all (x,y) copies of each realisation.\n"
    )
    lines.append("\n## Plots\n")
    lines.append("- `runlength_<FM>.png`         — run-length histograms\n")
    lines.append("- `facies_fractions_<FM>.png`  — facies fraction box plots\n")
    lines.append("- `transition_compare_<FM>.png` — real / sim / diff heatmaps\n")
    lines.append("- `transition_matrices/<FM>_transition.png` — Task-2 fitted matrices\n")
    lines.append("- `grf_perturbation_example.png` — Task-1 layer-boundary GRF demo\n")
    lines.append("- `variogram_check.png`         — Task-4 variogram fit (gaussian_filter vs gstools)\n")
    lines.append("- `basin_distribution.png`      — Task-5b k-means basins + pool coverage\n")

    out_path.write_text("".join(lines))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-maps", type=int, default=5,
                   help="number of simulator maps to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formations", nargs="+", default=None,
                   help="restrict to this list (default: all in geom)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading real samples.parquet ...")
    real = pd.read_parquet("data/clean/samples.parquet")
    real = real[real["dataset"] == "NLOG"].drop_duplicates(
        ["dataset", "borehole", "depth"], keep="first")

    print("loading simulator components ...")
    bank = DistributionBank.load("data/clean/distributions.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    cfg = SimConfig()
    variables = list(cfg.variables)
    formations = args.formations or [
        fm for fm in geom.formation_order
        if geom.formations.get(fm) is not None
    ]
    print(f"  formations to check: {formations}")
    print(f"  variables to check:  {variables}")

    print(f"generating {args.n_maps} simulator maps ...")
    gen = MapGenerator(bank, geom, cfg, seed=args.seed, prior=prior)
    sim_maps = [next(gen) for _ in range(args.n_maps)]
    print(f"  generated {len(sim_maps)} maps "
          f"({sim_maps[0]['rock_types'].size:,} cells each)")

    print("extracting per-formation columns (real + sim) ...")
    real_cols = real_per_formation_columns(real)
    sim_cols = sim_per_formation_columns(sim_maps)

    print("[1/4] run-length Wasserstein ...")
    ws_df = check_run_lengths(real_cols, sim_cols, formations)
    plot_run_length_histograms(real_cols, sim_cols, formations, OUT_DIR)

    print("[2/4] facies fractions ...")
    frac_df = check_facies_fractions(real_cols, sim_cols, formations, OUT_DIR)

    print("[3/4] transition matrices ...")
    sim_df = sim_to_dataframe(sim_maps)
    trans_df = check_transition_matrices(geom, sim_df, formations, OUT_DIR)

    print("[4/4] petrophysical marginals (KS) ...")
    ks_df = check_petrophysical_marginals(real, sim_maps, variables, DEPTH_BINS)

    # CSVs for downstream diffing
    ws_df.to_csv(OUT_DIR / "_runlength_wasserstein.csv", index=False)
    frac_df.to_csv(OUT_DIR / "_facies_fractions.csv", index=False)
    trans_df.to_csv(OUT_DIR / "_transition_metrics.csv", index=False)
    ks_df.to_csv(OUT_DIR / "_ks_marginals.csv", index=False)

    write_report(args.n_maps, ws_df, frac_df, trans_df, ks_df, REPORT_PATH,
                 bank=bank, geom=geom)
    print(f"\nREPORT -> {REPORT_PATH}")
    print(f"plots  -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
