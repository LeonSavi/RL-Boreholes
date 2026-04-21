
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.plots as plots


DEFAULT_SAMPLES = Path("data/clean/samples.parquet")
DEFAULT_OUT     = Path("plots/analysis")

DEPTH_BINS = [0, 100, 300, 800, 1500, 2500, 3500, 5000]

# Measurements where direct-unit comparison between LILY and NLOG makes sense
DIRECTLY_COMPARABLE = ["rhob", "dt_us_ft"]



def table_summary_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Median, IQR, n per (dataset, measurement, category).
    Uses formation for NLOG and rock_type for both datasets."""
    rows = []
    nlog = df[df["dataset"] == "NLOG"]
    for (meas, form), grp in nlog.groupby(["measurement", "formation"]):
        vals = grp["value"].dropna()
        if len(vals) < 30:
            continue
        rows.append({
            "dataset": "NLOG", "measurement": meas,
            "category_kind": "formation", "category": form,
            "n": len(vals), "wells": grp["borehole"].nunique(),
            "mean":  round(vals.mean(),   4),
            "std":   round(vals.std(),    4),
            "p10":   round(vals.quantile(0.10), 4),
            "p25":   round(vals.quantile(0.25), 4),
            "p50":   round(vals.median(),       4),
            "p75":   round(vals.quantile(0.75), 4),
            "p90":   round(vals.quantile(0.90), 4),
            "iqr":   round(vals.quantile(0.75) - vals.quantile(0.25), 4),
        })
    for (ds, meas, rt), grp in df.groupby(["dataset", "measurement", "rock_type"]):
        vals = grp["value"].dropna()
        if len(vals) < 30:
            continue
        rows.append({
            "dataset": ds, "measurement": meas,
            "category_kind": "rock_type", "category": rt,
            "n": len(vals), "wells": grp["borehole"].nunique(),
            "mean":  round(vals.mean(),   4),
            "std":   round(vals.std(),    4),
            "p10":   round(vals.quantile(0.10), 4),
            "p25":   round(vals.quantile(0.25), 4),
            "p50":   round(vals.median(),       4),
            "p75":   round(vals.quantile(0.75), 4),
            "p90":   round(vals.quantile(0.90), 4),
            "iqr":   round(vals.quantile(0.75) - vals.quantile(0.25), 4),
        })
    return pd.DataFrame(rows)


def table_compaction_trends(df: pd.DataFrame) -> pd.DataFrame:
    """p25/p50/p75/n per (rock_type, measurement, dataset, depth_bin).
    For showing how distributions shift with depth."""
    rows = []
    d = df.copy()
    d["depth_bin"] = pd.cut(d["depth"], bins=DEPTH_BINS, include_lowest=True)
    for (rt, meas, ds, dbin), grp in d.groupby(
            ["rock_type", "measurement", "dataset", "depth_bin"], observed=True):
        vals = grp["value"].dropna()
        if len(vals) < 30:
            continue
        rows.append({
            "rock_type":    rt,
            "measurement":  meas,
            "dataset":      ds,
            "depth_bin":    str(dbin),
            "depth_median": round(grp["depth"].median(), 1),
            "n":            len(vals),
            "wells":        grp["borehole"].nunique(),
            "p25":          round(vals.quantile(0.25), 4),
            "p50":          round(vals.median(),       4),
            "p75":          round(vals.quantile(0.75), 4),
        })
    return pd.DataFrame(rows).sort_values(
        ["rock_type", "measurement", "dataset", "depth_median"])


def table_depth_matched_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side LILY vs NLOG at matched rock_type x measurement x depth_bin.

    Outputs one row per (rock_type, measurement, depth_bin) where BOTH datasets
    have samples.  Columns include:
        lily_p50, lily_iqr, lily_n, nlog_p50, nlog_iqr, nlog_n,
        delta_p50, pooled_iqr, disagreement_flag

    The flag fires when |delta_p50| > pooled_iqr — a rough test that the
    shift is meaningful given the spread within each dataset.
    """
    d = df.copy()
    d["depth_bin"] = pd.cut(d["depth"], bins=DEPTH_BINS, include_lowest=True)

    rows = []
    for (rt, meas), grp_rm in d.groupby(["rock_type", "measurement"], observed=True):
        if meas not in DIRECTLY_COMPARABLE:
            continue
        for dbin, grp in grp_rm.groupby("depth_bin", observed=True):
            ds_groups = {ds: g for ds, g in grp.groupby("dataset")}
            if "LILY" not in ds_groups or "NLOG" not in ds_groups:
                continue
            l_vals = ds_groups["LILY"]["value"]
            n_vals = ds_groups["NLOG"]["value"]
            if len(l_vals) < 30 or len(n_vals) < 30:
                continue
            l_p50, l_p25, l_p75 = l_vals.median(), l_vals.quantile(0.25), l_vals.quantile(0.75)
            n_p50, n_p25, n_p75 = n_vals.median(), n_vals.quantile(0.25), n_vals.quantile(0.75)
            l_iqr = l_p75 - l_p25
            n_iqr = n_p75 - n_p25
            # pooled spread: average of the two IQRs
            pooled_iqr = (l_iqr + n_iqr) / 2.0
            delta = n_p50 - l_p50
            rows.append({
                "rock_type":    rt,
                "measurement":  meas,
                "depth_bin":    str(dbin),
                "depth_median_m": round(grp["depth"].median(), 1),
                "lily_n":       len(l_vals),
                "lily_wells":   ds_groups["LILY"]["borehole"].nunique(),
                "lily_p25":     round(l_p25, 4),
                "lily_p50":     round(l_p50, 4),
                "lily_p75":     round(l_p75, 4),
                "lily_iqr":     round(l_iqr, 4),
                "nlog_n":       len(n_vals),
                "nlog_wells":   ds_groups["NLOG"]["borehole"].nunique(),
                "nlog_p25":     round(n_p25, 4),
                "nlog_p50":     round(n_p50, 4),
                "nlog_p75":     round(n_p75, 4),
                "nlog_iqr":     round(n_iqr, 4),
                "delta_p50":    round(delta, 4),
                "pooled_iqr":   round(pooled_iqr, 4),
                "abs_delta_over_pooled_iqr": (
                    round(abs(delta) / pooled_iqr, 2) if pooled_iqr > 0 else None),
                "disagreement_flag":
                    "YES" if (pooled_iqr > 0 and abs(delta) > pooled_iqr) else "no",
            })
    return pd.DataFrame(rows).sort_values(
        ["rock_type", "measurement", "depth_median_m"])


def table_coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matrix of (dataset, measurement) → wells and rows."""
    agg = (df.groupby(["dataset", "measurement"])
             .agg(rows=("value", "size"),
                  wells=("borehole", "nunique"),
                  formations=("formation", lambda x: x.nunique(dropna=True)),
                  rock_types=("rock_type", "nunique"),
                  depth_p50=("depth", "median"))
             .reset_index()
             .sort_values(["dataset", "rows"], ascending=[True, False]))
    agg["depth_p50"] = agg["depth_p50"].round(1)
    return agg


def table_outlier_wells(df: pd.DataFrame,
                        mad_multiplier: float = 5.0) -> pd.DataFrame:
    """Wells whose median is more than mad_multiplier MADs from the corpus
    median, per (dataset, measurement)."""
    rows = []
    for (ds, meas), grp in df.groupby(["dataset", "measurement"]):
        wm = grp.groupby("borehole")["value"].median()
        if len(wm) < 5:
            continue
        pop = wm.median()
        mad = (wm - pop).abs().median()
        if mad == 0:
            continue
        dev = (wm - pop).abs() / mad
        outliers = wm[dev > mad_multiplier]
        for bh, v in outliers.items():
            rows.append({
                "dataset":   ds,
                "measurement": meas,
                "borehole":  bh,
                "well_median": round(float(v), 4),
                "corpus_median": round(float(pop), 4),
                "mad_distance":  round(float(dev[bh]), 1),
                "n_samples_in_well": int(grp.loc[grp["borehole"] == bh].shape[0]),
            })
    return pd.DataFrame(rows).sort_values(["dataset", "measurement", "mad_distance"],
                                          ascending=[True, True, False])



def write_overview(df: pd.DataFrame, dms: pd.DataFrame, out_path: Path) -> None:
    L = []
    add = L.append
    add("=" * 74)
    add(" DATA OVERVIEW")
    add("=" * 74)
    add(f"\nTotal rows       : {len(df):>12,}")
    add(f"Unique boreholes : {df['borehole'].nunique():>12,}")
    add(f"Measurements     : {df['measurement'].nunique():>12}")
    add(f"Rock types       : {df['rock_type'].nunique():>12}")
    add(f"Formations       : {df['formation'].dropna().nunique():>12}")

    add("\n--- By dataset ---")
    for ds, n in df["dataset"].value_counts().items():
        d = df[df["dataset"] == ds]
        add(f"  {ds:5s}  rows={n:>10,}  wells={d['borehole'].nunique():>5}  "
            f"depth_p50={d['depth'].median():>6.0f} m  "
            f"depth_max={d['depth'].max():>6.0f} m")

    add("\n--- Top measurements by row count ---")
    for meas, n in df["measurement"].value_counts().head(15).items():
        wells = df.loc[df["measurement"] == meas, "borehole"].nunique()
        add(f"  {meas:<16s}  n={n:>10,}  wells={wells:>5}")

    add("\n--- Rock type distribution ---")
    for rt, n in df["rock_type"].value_counts().items():
        wells = df.loc[df["rock_type"] == rt, "borehole"].nunique()
        add(f"  {rt:<12s}  n={n:>10,}  wells={wells:>5}")

    add("\n--- Formation distribution (NLOG) ---")
    nlog = df[df["dataset"] == "NLOG"]
    for f, n in nlog["formation"].value_counts().items():
        wells = nlog.loc[nlog["formation"] == f, "borehole"].nunique()
        add(f"  {f:<4s}  n={n:>10,}  wells={wells:>5}")

    add("\n--- Shared rock types LILY ∩ NLOG ---")
    lily_rt = set(df.loc[df["dataset"] == "LILY", "rock_type"].unique())
    nlog_rt = set(df.loc[df["dataset"] == "NLOG", "rock_type"].unique())
    shared = sorted(lily_rt & nlog_rt)
    add(f"  {shared}")

    add("\n--- Physical sanity spot-checks ---")
    checks = [
        ("NLOG ZE RHOB (halite+anhydrite bimodal)",
         nlog[(nlog["formation"] == "ZE") & (nlog["measurement"] == "rhob")]["value"]),
        ("NLOG CK RHOB (chalk)",
         nlog[(nlog["formation"] == "CK") & (nlog["measurement"] == "rhob")]["value"]),
        ("NLOG RO GR (bimodal Slochteren+Ten Boer)",
         nlog[(nlog["formation"] == "RO") & (nlog["measurement"] == "gr_api")]["value"]),
        ("LILY clay RHOB (young marine clay)",
         df[(df["dataset"] == "LILY") & (df["rock_type"] == "clay") &
            (df["measurement"] == "rhob")]["value"]),
    ]
    for label, vals in checks:
        if len(vals) < 30:
            continue
        add(f"  {label}")
        add(f"    n={len(vals):>10,}  p10={vals.quantile(0.1):>6.3f}  "
            f"p50={vals.median():>6.3f}  p90={vals.quantile(0.9):>6.3f}")

    # Depth-matched disagreement summary
    add("\n--- Depth-matched LILY-vs-NLOG disagreements ---")
    add(f"  (|delta_p50| > pooled_iqr means the two datasets disagree materially")
    add(f"   even after controlling for depth)")
    if len(dms):
        n_flag = (dms["disagreement_flag"] == "YES").sum()
        add(f"  rows with samples in both datasets  : {len(dms)}")
        add(f"  rows flagged as disagreement        : {n_flag}")
        if n_flag:
            flagged = dms[dms["disagreement_flag"] == "YES"].copy()
            add("  top disagreements (rock type / measurement / depth / delta / pooled IQR):")
            for _, r in flagged.sort_values("abs_delta_over_pooled_iqr",
                                             ascending=False).head(10).iterrows():
                add(f"    {r['rock_type']:<10s}  {r['measurement']:<10s}  "
                    f"{r['depth_bin']:<20s}  "
                    f"Δ={r['delta_p50']:>+6.3f}  IQR≈{r['pooled_iqr']:.3f}  "
                    f"({r['abs_delta_over_pooled_iqr']:.1f}×)")
    out_path.write_text("\n".join(L))
    print(f"  wrote {out_path}")



def table_support_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """For each (rock_type_fine, measurement), compute the empirical support
    bounds from the COMBINED LILY+NLOG distribution. This is the observation-
    model-ready table: 'for a given rock type, what are the physically
    reasonable bounds of the log reading?'

    """
    rows = []
    for (rt, meas), grp in df.groupby(["rock_type_fine", "measurement"]):
        vals = grp["value"].dropna()
        if len(vals) < 50:
            continue

        p05, p50, p95 = np.percentile(vals, [5, 50, 95])
        support_width = p95 - p05

        # per-dataset stats
        lily_v = grp.loc[grp["dataset"] == "LILY", "value"]
        nlog_v = grp.loc[grp["dataset"] == "NLOG", "value"]

        lily_stats = None
        nlog_stats = None
        if len(lily_v) >= 30:
            lily_stats = np.percentile(lily_v, [5, 50, 95])
        if len(nlog_v) >= 30:
            nlog_stats = np.percentile(nlog_v, [5, 50, 95])

        # which dataset extends the support?
        extends = []
        if lily_stats is not None and nlog_stats is not None:
            tol = 0.05 * support_width
            if lily_stats[0] < nlog_stats[0] - tol:
                extends.append("LILY_lo")
            if lily_stats[2] > nlog_stats[2] + tol:
                extends.append("LILY_hi")
            if nlog_stats[0] < lily_stats[0] - tol:
                extends.append("NLOG_lo")
            if nlog_stats[2] > lily_stats[2] + tol:
                extends.append("NLOG_hi")

        rows.append({
            "rock_type_fine":   rt,
            "measurement":      meas,
            "n_total":          len(vals),
            "n_lily":           len(lily_v),
            "n_nlog":           len(nlog_v),
            "combined_p05":     round(float(p05), 4),
            "combined_p50":     round(float(p50), 4),
            "combined_p95":     round(float(p95), 4),
            "support_width":    round(float(support_width), 4),
            "lily_p05":         round(float(lily_stats[0]), 4) if lily_stats is not None else None,
            "lily_p95":         round(float(lily_stats[2]), 4) if lily_stats is not None else None,
            "nlog_p05":         round(float(nlog_stats[0]), 4) if nlog_stats is not None else None,
            "nlog_p95":         round(float(nlog_stats[2]), 4) if nlog_stats is not None else None,
            "extends":          "|".join(extends) if extends else "",
        })
    return pd.DataFrame(rows).sort_values(["measurement", "rock_type_fine"])


def table_fine_vs_coarse(df: pd.DataFrame) -> pd.DataFrame:

    rows = []
    nlog = df[df["dataset"] == "NLOG"]
    for (formation, meas), grp in nlog.groupby(["formation", "measurement"]):
        if len(grp) < 1_000:
            continue
        # coarse distribution = all values under this formation, single label
        all_vals = grp["value"].dropna()
        coarse_iqr = float(all_vals.quantile(0.75) - all_vals.quantile(0.25))
        coarse_p50 = float(all_vals.median())

        # fine distribution = values weighted by component rock_type_fine
        # compute median-of-medians and pooled IQR as weighted averages
        fine_medians = []
        fine_iqrs = []
        fine_weights = []
        fine_types = []
        for rt, sub in grp.groupby("rock_type_fine"):
            if len(sub) < 200:
                continue
            vals = sub["value"].dropna()
            if len(vals) < 50:
                continue
            fine_medians.append(float(vals.median()))
            fine_iqrs.append(float(vals.quantile(0.75) - vals.quantile(0.25)))
            fine_weights.append(len(vals))
            fine_types.append(rt)

        if len(fine_types) < 2:
            continue  # no informative split

        total_w = sum(fine_weights)
        pooled_fine_iqr = sum(i * w for i, w in zip(fine_iqrs, fine_weights)) / total_w

        rows.append({
            "formation":           formation,
            "measurement":         meas,
            "n_rows":              len(grp),
            "fine_types":          ", ".join(fine_types),
            "n_fine_types":        len(fine_types),
            "coarse_p50":          round(coarse_p50, 4),
            "coarse_iqr":          round(coarse_iqr, 4),
            "pooled_fine_iqr":     round(pooled_fine_iqr, 4),
            "iqr_reduction_pct":   round(100 * (1 - pooled_fine_iqr / coarse_iqr), 1)
                                     if coarse_iqr > 0 else 0,
            "fine_medians":        ", ".join(f"{m:.3f}" for m in fine_medians),
        })
    return (pd.DataFrame(rows)
              .sort_values("iqr_reduction_pct", ascending=False))




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.samples.exists():
        raise FileNotFoundError(f"{args.samples} not found — run pull_data.py first")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.samples} ...")
    df = pd.read_parquet(args.samples)
    print(f"  {len(df):,} rows, {df['borehole'].nunique()} wells, "
          f"{df['measurement'].nunique()} measurements")

    print("\n--- Tables ---")
    t1 = table_summary_per_category(df)
    t1.to_csv(args.out / "summary_per_category.csv", index=False)
    print(f"  wrote {args.out / 'summary_per_category.csv'}  ({len(t1)} rows)")

    t2 = table_compaction_trends(df)
    t2.to_csv(args.out / "compaction_trends.csv", index=False)
    print(f"  wrote {args.out / 'compaction_trends.csv'}  ({len(t2)} rows)")

    t3 = table_depth_matched_summary(df)
    t3.to_csv(args.out / "depth_matched_summary.csv", index=False)
    print(f"  wrote {args.out / 'depth_matched_summary.csv'}  ({len(t3)} rows)")

    t4 = table_coverage_matrix(df)
    t4.to_csv(args.out / "coverage_matrix.csv", index=False)
    print(f"  wrote {args.out / 'coverage_matrix.csv'}  ({len(t4)} rows)")

    t5 = table_outlier_wells(df)
    t5.to_csv(args.out / "outlier_wells.csv", index=False)
    print(f"  wrote {args.out / 'outlier_wells.csv'}  ({len(t5)} rows)")

    write_overview(df, t3, args.out / "overview.txt")

    if "rock_type_fine" in df.columns:
        t6 = table_support_bounds(df)
        t6.to_csv(args.out / "support_bounds.csv", index=False)
        print(f"  wrote {args.out / 'support_bounds.csv'}  ({len(t6)} rows)")

        t7 = table_fine_vs_coarse(df)
        t7.to_csv(args.out / "fine_vs_coarse_comparison.csv", index=False)
        print(f"  wrote {args.out / 'fine_vs_coarse_comparison.csv'}  ({len(t7)} rows)")
    else:
        print("?????/")

    print("\n--- Plots (original analysis) ---")
    plots.plot_feature_coverage(df,        args.out / "01_feature_coverage.png")
    plots.plot_rock_type_coverage(df,      args.out / "02_rock_type_coverage.png")
    plots.plot_depth_distribution(df,      args.out / "03_depth_distribution.png")
    plots.plot_nlog_by_formation(df,       args.out / "04_nlog_by_formation.png")
    plots.plot_lily_by_lithology(df,       args.out / "05_lily_by_lithology.png")
    plots.plot_per_feature_violin(df,      args.out / "06_violin_rhob.png", "rhob")
    plots.plot_per_feature_violin(df,      args.out / "07_violin_gr.png",   "gr_api")
    plots.plot_lily_vs_nlog_native(df,     args.out / "08_lily_vs_nlog_native.png")
    plots.plot_lily_vs_nlog_zscore(df,     args.out / "09_lily_vs_nlog_zscore.png")
    plots.plot_compaction_trends(df,       args.out / "10_compaction_trends.png")
    plots.plot_depth_matched_comparison(df, args.out / "11_depth_matched_comparison.png")
    plots.plot_depth_matched_gamma(df,     args.out / "12_depth_matched_gamma.png")
    plots.plot_gardner(df,                 args.out / "13_gardner.png")
    plots.plot_rhob_vs_gr(df,              args.out / "14_rhob_vs_gr.png")
    plots.plot_resistivity_vs_porosity(df, args.out / "15_resistivity_vs_porosity.png")

    plots.plot_joint_support_by_rock_fine(df,  args.out / "16_joint_support_rhob.png",   "rhob")
    plots.plot_joint_support_by_rock_fine(df,  args.out / "17_joint_support_dt.png",     "dt_us_ft")
    plots.plot_fine_vs_coarse_distributions(df, args.out / "18_fine_vs_coarse_rhob.png",  "rhob")
    plots.plot_fine_vs_coarse_distributions(df, args.out / "19_fine_vs_coarse_gr.png",    "gr_api")
    plots.plot_nonlinear_compaction_fit(df,    args.out / "20_nonlinear_compaction.png", "rhob")
    plots.plot_nonlinear_compaction_fit(df,    args.out / "21_nonlinear_slowness.png",   "dt_us_ft")
    plots.plot_depth_hexbin_by_rock_fine(df,   args.out / "22_depth_hexbin_rhob.png",    "rhob")
    plots.plot_support_bounds_matrix(df,       args.out / "23_support_bounds_matrix.png")
    plots.plot_onshore_vs_offshore(df,         args.out / "24_onshore_vs_offshore_rhob.png", "rhob")
    plots.plot_onshore_vs_offshore(df,         args.out / "25_onshore_vs_offshore_gr.png",   "gr_api")

    print("\n--- Plots (single-dataset measurements: NPHI on NLOG, msus_si on LILY) ---")
    plots.plot_nphi_by_rock_fine(df,           args.out / "26_nphi_by_rock_fine.png")
    plots.plot_msus_by_rock_type_lily(df,      args.out / "27_msus_by_rock_lily.png")
    plots.plot_joint_support_by_rock_fine(df,  args.out / "28_support_nphi.png",  "nphi")
    plots.plot_joint_support_by_rock_fine(df,  args.out / "29_support_msus.png",  "msus_si")

    print(f"\n→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()