"""Policy significance analysis for v2 robustness evaluation results.

Reads per-map step CSVs produced by evaluate_robustness.py (v2) and computes
additional step-level metrics and paired statistical significance tests without
re-running any simulation or loading any model checkpoint.

Outputs (written to {run_dir}/significance_analysis/):
  policy_step_metrics.csv           -- one row per (map, start, model, policy, step)
  policy_step_aggregate.csv         -- aggregated by (model, policy, start, step) + overall
  policy_significance_tests.csv     -- McNemar and Wilcoxon policy comparisons
  policy_model_comparison_tests.csv -- Wilcoxon model comparisons (greedy + uncertainty)

Usage
-----
    python -m decision_simulator.utils.evaluate_policy_significance \
        --run-dir decision_simulator/results/robustness/YYYYMMDD_HHMMSS \
        [--n-bodies 1]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import wilcoxon as _scipy_wilcoxon


# ── Constants ─────────────────────────────────────────────────────────────────

# Map model directory labels to canonical model names
_LABEL_TO_MODEL = {"cat_var": "cat_var", "only_ore": "ore_only_null"}

# Map orebody subdirectory names to n_bodies integer
_OREBODY_TO_N = {"no_orebodies": 0, "one_orebody": 1, "two_orebodies": 2}

_START_RE = re.compile(r"start_(\d+)_r(\d+)_c(\d+)")
_MAP_RE   = re.compile(r"map_(\d+)\.csv")

# Policy pairs for all pairwise comparisons
_POLICY_PAIRS = [
    ("greedy",     "uncertainty"),
    ("uncertainty","random"),
    ("greedy",     "random"),
]

# Metrics aggregated in policy_step_aggregate.csv
_STEP_AGG_METRICS = ["absolute_total_ore_error", "top_ore", "n_top_ore_found_so_far"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_step_data(run_dir: Path) -> pd.DataFrame:
    """Walk the v2 robustness run tree and load all per-map step CSVs.

    Expected tree:
        {run_dir}/{model_label}/start_XX_rXX_cXX/prediction_summaries/{orebody}/maps/map_XXX.csv
    """
    frames: list[pd.DataFrame] = []

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = _LABEL_TO_MODEL.get(model_dir.name)
        if model_name is None:
            continue

        for start_dir in sorted(model_dir.iterdir()):
            if not start_dir.is_dir():
                continue
            m = _START_RE.match(start_dir.name)
            if not m:
                continue
            start_idx = int(m.group(1))
            init_i    = int(m.group(2))
            init_j    = int(m.group(3))

            pred_dir = start_dir / "prediction_summaries"
            if not pred_dir.is_dir():
                continue

            for orebody_dir in sorted(pred_dir.iterdir()):
                if not orebody_dir.is_dir():
                    continue
                n_bodies = _OREBODY_TO_N.get(orebody_dir.name)
                if n_bodies is None:
                    continue

                maps_dir = orebody_dir / "maps"
                if not maps_dir.is_dir():
                    continue

                for csv_file in sorted(maps_dir.glob("map_*.csv")):
                    fm = _MAP_RE.match(csv_file.name)
                    if not fm:
                        continue
                    map_idx = int(fm.group(1))

                    try:
                        chunk = pd.read_csv(csv_file)
                    except Exception as exc:
                        print(f"  [warn] skipping {csv_file}: {exc}")
                        continue

                    chunk["model"]     = model_name
                    chunk["start_idx"] = start_idx
                    chunk["init_i"]    = init_i
                    chunk["init_j"]    = init_j
                    chunk["n_bodies"]  = n_bodies
                    chunk["map_idx"]   = map_idx
                    frames.append(chunk)

    if not frames:
        raise FileNotFoundError(
            f"No per-map step CSVs found under {run_dir}.\n"
            "Run evaluate_robustness.py (v2) first to generate them."
        )

    df = pd.concat(frames, ignore_index=True)

    # Normalise top_ore to bool (csv.DictWriter writes Python True/False as strings)
    df["top_ore"] = (
        df["top_ore"]
        .map({True: True, False: False, "True": True, "False": False, 1: True, 0: False})
        .astype(bool)
    )

    return df


# ── Derived metrics ───────────────────────────────────────────────────────────

def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute_total_ore_error, n_top_ore_found_so_far, top_ore_found_within_budget."""
    df = df.copy()

    # Absolute error between summed predicted map and true map total
    df["absolute_total_ore_error"] = (
        df["total_predicted_ore"] - df["total_true_ore"]
    ).abs()

    # Cumulative top-ore discoveries within each (model, start, map, policy) run
    df = df.sort_values(["model", "start_idx", "map_idx", "policy", "step"])
    df["n_top_ore_found_so_far"] = (
        df.groupby(["model", "start_idx", "map_idx", "policy"])["top_ore"]
        .cumsum()
        .astype(int)
    )

    # Binary per-run flag: did this policy find any top-ore cell within budget?
    # Propagated to all steps of the same run for convenience in policy_step_metrics.csv
    run_found = (
        df.groupby(["model", "start_idx", "map_idx", "policy"])["top_ore"]
        .any()
        .astype(int)
        .rename("top_ore_found_within_budget")
        .reset_index()
    )
    df = df.merge(run_found, on=["model", "start_idx", "map_idx", "policy"])

    return df


# ── Statistical test helpers ──────────────────────────────────────────────────

def _mcnemar(a_found: np.ndarray, b_found: np.ndarray) -> dict:
    """McNemar test with continuity correction for paired binary outcomes."""
    both    = int(((a_found == 1) & (b_found == 1)).sum())
    a_only  = int(((a_found == 1) & (b_found == 0)).sum())
    b_only  = int(((a_found == 0) & (b_found == 1)).sum())
    neither = int(((a_found == 0) & (b_found == 0)).sum())
    n_disc  = a_only + b_only

    if n_disc == 0:
        stat, p = 0.0, 1.0
    else:
        # Edwards continuity correction
        stat = float((abs(a_only - b_only) - 1) ** 2 / n_disc)
        p    = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "both_success":    both,
        "only_a_success":  a_only,
        "only_b_success":  b_only,
        "neither_success": neither,
        "statistic":       round(stat, 4),
        "p_value_raw":     round(p, 6),
    }


def _wilcoxon_pair(
    a: np.ndarray,
    b: np.ndarray,
    higher_is_better: bool,
) -> dict:
    """Paired two-sided Wilcoxon signed-rank test.

    Zero-difference pairs are excluded ('wilcox' method); n_excluded reports
    how many were dropped. proportion_a_better counts strict wins for A.
    """
    d = a - b
    n_pairs    = int(len(d))
    n_excluded = int((d == 0).sum())

    try:
        stat, p = _scipy_wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        stat, p = float(stat), float(p)
    except ValueError:
        # All differences are zero — test undefined, p=1 by convention
        stat, p = 0.0, 1.0

    if higher_is_better:
        proportion_a_better = float((a > b).mean())
    else:
        proportion_a_better = float((a < b).mean())

    return {
        "n_pairs":             n_pairs,
        "n_excluded":          n_excluded,
        "median_a":            round(float(np.median(a)), 4),
        "median_b":            round(float(np.median(b)), 4),
        "mean_difference":     round(float(np.mean(d)), 4),
        "median_difference":   round(float(np.median(d)), 4),
        "proportion_a_better": round(proportion_a_better, 4),
        "statistic":           round(stat, 4),
        "p_value_raw":         round(float(p), 6),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down multiple-comparison correction within one test family."""
    k = len(p_values)
    if k == 0:
        return []
    order      = sorted(range(k), key=lambda i: p_values[i])
    adjusted   = [0.0] * k
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj         = min(1.0, p_values[idx] * (k - rank))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [round(v, 6) for v in adjusted]


# ── Significance tests ────────────────────────────────────────────────────────

def _pivot_pair(per_run: pd.DataFrame, model: str, pol_a: str, pol_b: str, col: str):
    """Return aligned numpy arrays (a_vals, b_vals, n_common) for two policies."""
    sub = per_run[per_run["model"] == model]
    grp_a = sub[sub["policy"] == pol_a].set_index(["start_idx", "map_idx"])
    grp_b = sub[sub["policy"] == pol_b].set_index(["start_idx", "map_idx"])
    common = grp_a.index.intersection(grp_b.index)
    return (
        grp_a.loc[common, col].values.astype(float),
        grp_b.loc[common, col].values.astype(float),
        len(common),
    )


def _run_policy_tests(per_run: pd.DataFrame, model: str = "cat_var") -> pd.DataFrame:
    """McNemar (family 1) and Wilcoxon (families 2 & 3) for all policy pairs."""
    all_rows: list[dict] = []

    # ── Family 1: McNemar — top_ore_found_within_budget ──────────────────────
    mcnemar_rows: list[dict] = []
    for pol_a, pol_b in _POLICY_PAIRS:
        a, b, n = _pivot_pair(per_run, model, pol_a, pol_b, "top_ore_found_within_budget")
        result = _mcnemar(a.astype(int), b.astype(int))
        found_rate_a = float(a.mean())
        found_rate_b = float(b.mean())
        mcnemar_rows.append({
            "model":                model,
            "metric":               "top_ore_found_within_budget",
            "comparison":           f"{pol_a}_vs_{pol_b}",
            "policy_a":             pol_a,
            "policy_b":             pol_b,
            "n_pairs":              n,
            "found_rate_a":         round(found_rate_a, 4),
            "found_rate_b":         round(found_rate_b, 4),
            "found_rate_difference":round(found_rate_a - found_rate_b, 4),
            "test_name":            "mcnemar",
            **result,
        })
    for row, ph in zip(mcnemar_rows, _holm_adjust([r["p_value_raw"] for r in mcnemar_rows])):
        row["p_value_holm"] = ph
    all_rows.extend(mcnemar_rows)

    # ── Family 2: Wilcoxon — n_top_ore_found ─────────────────────────────────
    wilcoxon_n_rows: list[dict] = []
    for pol_a, pol_b in _POLICY_PAIRS:
        a, b, n = _pivot_pair(per_run, model, pol_a, pol_b, "n_top_ore_found")
        result = _wilcoxon_pair(a, b, higher_is_better=True)
        wilcoxon_n_rows.append({
            "model": model, "metric": "n_top_ore_found",
            "comparison": f"{pol_a}_vs_{pol_b}",
            "policy_a": pol_a, "policy_b": pol_b,
            "test_name": "wilcoxon", **result,
        })
    for row, ph in zip(wilcoxon_n_rows, _holm_adjust([r["p_value_raw"] for r in wilcoxon_n_rows])):
        row["p_value_holm"] = ph
    all_rows.extend(wilcoxon_n_rows)

    # ── Family 3: Wilcoxon — final_absolute_total_ore_error ──────────────────
    wilcoxon_err_rows: list[dict] = []
    for pol_a, pol_b in _POLICY_PAIRS:
        a, b, n = _pivot_pair(per_run, model, pol_a, pol_b, "final_absolute_total_ore_error")
        result = _wilcoxon_pair(a, b, higher_is_better=False)  # lower error is better
        wilcoxon_err_rows.append({
            "model": model, "metric": "final_absolute_total_ore_error",
            "comparison": f"{pol_a}_vs_{pol_b}",
            "policy_a": pol_a, "policy_b": pol_b,
            "test_name": "wilcoxon", **result,
        })
    for row, ph in zip(wilcoxon_err_rows, _holm_adjust([r["p_value_raw"] for r in wilcoxon_err_rows])):
        row["p_value_holm"] = ph
    all_rows.extend(wilcoxon_err_rows)

    return pd.DataFrame(all_rows)


def _run_model_comparison_tests(per_run: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon model comparison (cat_var vs ore_only_null) for greedy and uncertainty."""
    rows: list[dict] = []

    for policy in ["greedy", "uncertainty"]:
        for metric, higher_is_better in [
            ("n_top_ore_found",                True),
            ("final_absolute_total_ore_error", False),
        ]:
            cat  = per_run[(per_run["model"] == "cat_var")      & (per_run["policy"] == policy)
                           ].set_index(["start_idx", "map_idx"])
            null = per_run[(per_run["model"] == "ore_only_null") & (per_run["policy"] == policy)
                           ].set_index(["start_idx", "map_idx"])
            common = cat.index.intersection(null.index)

            a = cat.loc[common,  metric].values.astype(float)
            b = null.loc[common, metric].values.astype(float)
            result = _wilcoxon_pair(a, b, higher_is_better=higher_is_better)

            rows.append({
                "metric":     metric,
                "comparison": "cat_var_vs_ore_only_null",
                "policy":     policy,
                "policy_a":   f"cat_var_{policy}",
                "policy_b":   f"ore_only_null_{policy}",
                "test_name":  "wilcoxon",
                **result,
            })

    # Holm correction within this family (4 tests: 2 policies x 2 metrics)
    for row, ph in zip(rows, _holm_adjust([r["p_value_raw"] for r in rows])):
        row["p_value_holm"] = ph

    return pd.DataFrame(rows)


# ── Step-level aggregation ────────────────────────────────────────────────────

def _aggregate_by_step(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean, std, median, n of step-level metrics per group."""
    rows: list[dict] = []
    for keys, grp in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_cols, keys))
        row["n_maps"] = int(grp["map_idx"].nunique())
        for col in _STEP_AGG_METRICS:
            vals = grp[col].dropna().astype(float)
            row[f"{col}_mean"]   = round(float(vals.mean()),        4) if len(vals) > 0 else float("nan")
            row[f"{col}_std"]    = round(float(vals.std(ddof=1)),   4) if len(vals) > 1 else float("nan")
            row[f"{col}_median"] = round(float(vals.median()),      4) if len(vals) > 0 else float("nan")
            row[f"{col}_n"]      = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "robustness"

    parser = argparse.ArgumentParser(
        description="Significance analysis for v2 robustness evaluation results."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to v2 robustness run directory. Defaults to latest under results/robustness/.",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=None,
        help="Filter to maps with this many ore bodies (0 or 1). Default: all.",
    )
    args = parser.parse_args()

    if args.run_dir is None:
        candidates = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {RESULTS_DIR}.")
        args.run_dir = candidates[-1]

    out_dir = args.run_dir / "significance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and enrich data ──────────────────────────────────────────────────

    print(f"Loading step data from: {args.run_dir}")
    df = _load_step_data(args.run_dir)
    df = _add_derived_metrics(df)

    if args.n_bodies is not None:
        df = df[df["n_bodies"] == args.n_bodies].copy()
        print(f"Filtered to n_bodies={args.n_bodies}: {len(df):,} rows remaining")

    n_maps     = df["map_idx"].nunique()
    n_starts   = df["start_idx"].nunique()
    n_models   = df["model"].nunique()
    n_policies = df["policy"].nunique()
    n_steps    = df["step"].nunique()

    # ── policy_step_metrics.csv ───────────────────────────────────────────────

    step_metrics = (
        df[[
            "map_idx", "start_idx", "init_i", "init_j", "n_bodies",
            "model", "policy", "step",
            "top_ore", "n_top_ore_found_so_far", "top_ore_found_within_budget",
            "total_predicted_ore", "total_true_ore", "absolute_total_ore_error",
        ]]
        .rename(columns={
            "map_idx":   "map_id",
            "start_idx": "start_id",
            "init_i":    "start_i",
            "init_j":    "start_j",
            "step":      "drill_step",
            "top_ore":   "top_ore_hit",
        })
        .sort_values(["model", "start_id", "map_id", "policy", "drill_step"])
        .round(4)
    )

    step_metrics_path = out_dir / "policy_step_metrics.csv"
    step_metrics.to_csv(step_metrics_path, index=False)

    # ── policy_step_aggregate.csv ─────────────────────────────────────────────

    agg_by_start = _aggregate_by_step(df, ["model", "policy", "start_idx", "step"])
    agg_by_start.insert(0, "aggregation", "per_start")

    agg_overall = _aggregate_by_step(df, ["model", "policy", "step"])
    agg_overall.insert(0, "aggregation", "overall")
    agg_overall.insert(3, "start_idx", "all")

    step_agg = pd.concat([agg_by_start, agg_overall], ignore_index=True)
    step_agg_path = out_dir / "policy_step_aggregate.csv"
    step_agg.to_csv(step_agg_path, index=False)

    # ── Build per-run summary for significance tests ──────────────────────────

    # One row per (model, start, map, policy) — values at final step for error
    per_run = (
        df.sort_values("step")
        .groupby(["model", "start_idx", "map_idx", "n_bodies", "policy"])
        .agg(
            n_top_ore_found=("top_ore", "sum"),
            final_absolute_total_ore_error=("absolute_total_ore_error", "last"),
            top_ore_found_within_budget=("top_ore_found_within_budget", "first"),
        )
        .reset_index()
    )

    # ── policy_significance_tests.csv ─────────────────────────────────────────

    sig_tests = _run_policy_tests(per_run, model="cat_var")
    sig_path = out_dir / "policy_significance_tests.csv"
    sig_tests.to_csv(sig_path, index=False)

    # ── policy_model_comparison_tests.csv ─────────────────────────────────────

    model_tests = _run_model_comparison_tests(per_run)
    model_path = out_dir / "policy_model_comparison_tests.csv"
    model_tests.to_csv(model_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────

    n_body_label = f"  (filtered to n_bodies={args.n_bodies})" if args.n_bodies is not None else ""
    print(f"\nRobustness run  : {args.run_dir}")
    print(f"Maps            : {n_maps}{n_body_label}")
    print(f"Starts          : {n_starts}")
    print(f"Models          : {', '.join(sorted(df['model'].unique()))}")
    print(f"Policies        : {', '.join(sorted(df['policy'].unique()))}")
    print(f"Drill steps     : {n_steps}")
    print(f"Paired obs.     : {n_maps * n_starts:,}  (map x start)")
    print(f"Outputs         : {out_dir}/")
    print(f"  {step_metrics_path.name:<44} ({len(step_metrics):>10,} rows)")
    print(f"  {step_agg_path.name:<44} ({len(step_agg):>10,} rows)")
    print(f"  {sig_path.name:<44} ({len(sig_tests):>10,} rows)")
    print(f"  {model_path.name:<44} ({len(model_tests):>10,} rows)")


if __name__ == "__main__":
    main()
