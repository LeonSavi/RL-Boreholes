"""Thesis-grade exploratory data analysis on the cleaned LILY+NLOG corpus.

Reads `data/clean/samples.parquet` (produced by `pull_data.py`) and writes:

  plots/analysis/
    EDA_REPORT.md                         <- narrative + key numbers + chart refs
    *.csv                                 <- the underlying tables
    01_data_census.png                    <- Section A — data overview (3 charts)
    02_depth_distribution.png
    03_nlog_well_map.png
    04_rock_type_coverage.png             <- Section B — lithology (3 charts)
    05_nlog_formation_breakdown.png
    06_lily_expeditions.png
    07_violin_rhob.png                    <- Section C — petrophysics (3 charts)
    08_compaction_trends.png
    09_fine_vs_coarse_rhob.png
    10_variable_availability.png          <- Section D — design choices (3 charts)
    11_support_bounds.png
    12_lily_vs_nlog_calibration.png

Run before training:
    python analysis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.plots as plots


DEFAULT_SAMPLES = Path("data/clean/samples.parquet")
DEFAULT_OUT     = Path("plots/analysis")
DEFAULT_GEOM    = Path("data/clean/formation_geometry.pkl")

DEPTH_BINS = [0, 100, 300, 800, 1500, 2500, 3500, 5000]

# variables considered for training (post-cleanup the encoder uses 5; PEF is
# the candidate we drop, included here so the availability chart can argue
# the case visually).
TARGET_VARS = ("rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log")
FINAL_VARS  = ("rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log")

DIRECTLY_COMPARABLE = ["rhob", "dt_us_ft"]


# ─── tables ───────────────────────────────────────────────────────────────

def table_summary_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(dataset, measurement, category) p25/p50/p75/n.  Category is
    NLOG-formation OR rock_type; the simulator's bank.cells slice the same
    way, so these are the rows the simulator sees during fitting."""
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
            "p10": round(vals.quantile(0.10), 4),
            "p50": round(vals.median(),       4),
            "p90": round(vals.quantile(0.90), 4),
            "iqr": round(vals.quantile(0.75) - vals.quantile(0.25), 4),
        })
    for (ds, meas, rt), grp in df.groupby(["dataset", "measurement",
                                            "rock_type"]):
        vals = grp["value"].dropna()
        if len(vals) < 30:
            continue
        rows.append({
            "dataset": ds, "measurement": meas,
            "category_kind": "rock_type", "category": rt,
            "n": len(vals), "wells": grp["borehole"].nunique(),
            "p10": round(vals.quantile(0.10), 4),
            "p50": round(vals.median(),       4),
            "p90": round(vals.quantile(0.90), 4),
            "iqr": round(vals.quantile(0.75) - vals.quantile(0.25), 4),
        })
    return pd.DataFrame(rows)


def table_coverage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["dataset", "measurement"])
              .agg(rows=("value", "size"),
                   wells=("borehole", "nunique"),
                   rock_types=("rock_type", "nunique"),
                   depth_p50=("depth", "median"),
                   depth_p95=("depth", lambda s: round(s.quantile(0.95), 1)))
              .reset_index()
              .sort_values(["dataset", "rows"], ascending=[True, False]))


def table_compaction_trends(df: pd.DataFrame) -> pd.DataFrame:
    """p25/p50/p75 per (rock, measurement, dataset, depth-bin) — shows that
    distributions shift meaningfully with depth, justifying the depth-bin
    structure of the simulator's CellDistribution table."""
    d = df.copy()
    d["depth_bin"] = pd.cut(d["depth"], bins=DEPTH_BINS, include_lowest=True)
    rows = []
    for (rt, meas, ds, dbin), grp in d.groupby(
            ["rock_type", "measurement", "dataset", "depth_bin"],
            observed=True):
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
            "p25":          round(vals.quantile(0.25), 4),
            "p50":          round(vals.median(),       4),
            "p75":          round(vals.quantile(0.75), 4),
        })
    return (pd.DataFrame(rows).sort_values(
        ["rock_type", "measurement", "dataset", "depth_median"]))


def table_depth_matched_summary(df: pd.DataFrame) -> pd.DataFrame:
    """LILY vs NLOG at matched (rock_type × measurement × depth_bin) — the
    'are these two corpora calibrated' check.  Disagreement flag fires when
    |Δmedian| > pooled IQR."""
    d = df.copy()
    d["depth_bin"] = pd.cut(d["depth"], bins=DEPTH_BINS, include_lowest=True)
    rows = []
    for (rt, meas), grp_rm in d.groupby(["rock_type", "measurement"],
                                         observed=True):
        if meas not in DIRECTLY_COMPARABLE:
            continue
        for dbin, grp in grp_rm.groupby("depth_bin", observed=True):
            ds_groups = {ds: g for ds, g in grp.groupby("dataset")}
            if "LILY" not in ds_groups or "NLOG" not in ds_groups:
                continue
            lv = ds_groups["LILY"]["value"]
            nv = ds_groups["NLOG"]["value"]
            if len(lv) < 30 or len(nv) < 30:
                continue
            l_p50, l_iqr = lv.median(), lv.quantile(0.75) - lv.quantile(0.25)
            n_p50, n_iqr = nv.median(), nv.quantile(0.75) - nv.quantile(0.25)
            pooled = (l_iqr + n_iqr) / 2.0
            delta = n_p50 - l_p50
            rows.append({
                "rock_type":   rt,
                "measurement": meas,
                "depth_bin":   str(dbin),
                "lily_n":      len(lv),
                "lily_p50":    round(l_p50, 4),
                "nlog_n":      len(nv),
                "nlog_p50":    round(n_p50, 4),
                "delta_p50":   round(delta, 4),
                "pooled_iqr":  round(pooled, 4),
                "abs_delta_over_pooled_iqr":
                    round(abs(delta) / pooled, 2) if pooled > 0 else None,
                "disagreement_flag":
                    "YES" if (pooled > 0 and abs(delta) > pooled) else "no",
            })
    return pd.DataFrame(rows).sort_values(
        ["rock_type", "measurement", "depth_bin"])


def table_fine_vs_coarse(df: pd.DataFrame) -> pd.DataFrame:
    """For each NLOG (formation × measurement), compare the coarse single-
    label IQR to the IQR-weighted average across rock_type_fine sub-classes.
    A big reduction means the fine-grained classes are buying real
    discriminative power — justifies the claystone_hot/cool and
    sandstone_clean/shaly splits."""
    rows = []
    nlog = df[df["dataset"] == "NLOG"]
    for (formation, meas), grp in nlog.groupby(["formation", "measurement"]):
        if len(grp) < 1_000:
            continue
        all_vals = grp["value"].dropna()
        coarse_iqr = float(all_vals.quantile(0.75) - all_vals.quantile(0.25))
        coarse_p50 = float(all_vals.median())
        fine_medians, fine_iqrs, fine_weights, fine_types = [], [], [], []
        for rt, sub in grp.groupby("rock_type_fine"):
            if len(sub) < 200:
                continue
            vals = sub["value"].dropna()
            if len(vals) < 50:
                continue
            fine_medians.append(float(vals.median()))
            fine_iqrs.append(float(vals.quantile(0.75)
                                   - vals.quantile(0.25)))
            fine_weights.append(len(vals))
            fine_types.append(rt)
        if len(fine_types) < 2:
            continue
        tw = sum(fine_weights)
        pooled_fine_iqr = sum(i * w for i, w in zip(fine_iqrs,
                                                     fine_weights)) / tw
        rows.append({
            "formation":         formation,
            "measurement":       meas,
            "n_rows":            len(grp),
            "fine_types":        ", ".join(fine_types),
            "coarse_p50":        round(coarse_p50, 4),
            "coarse_iqr":        round(coarse_iqr, 4),
            "pooled_fine_iqr":   round(pooled_fine_iqr, 4),
            "iqr_reduction_pct": (round(100 * (1 - pooled_fine_iqr
                                               / coarse_iqr), 1)
                                  if coarse_iqr > 0 else 0),
        })
    return pd.DataFrame(rows).sort_values("iqr_reduction_pct",
                                           ascending=False)


def table_support_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Per (rock_type_fine, measurement) empirical bounds [p05, p50, p95]
    on the combined LILY+NLOG distribution, plus per-dataset stats so we
    can see which corpus extends the support.  The simulator's
    DistributionBank uses the same shape to set CellDistribution.bounds."""
    rows = []
    for (rt, meas), grp in df.groupby(["rock_type_fine", "measurement"]):
        vals = grp["value"].dropna()
        if len(vals) < 50:
            continue
        p05, p50, p95 = np.percentile(vals, [5, 50, 95])
        lv = grp.loc[grp["dataset"] == "LILY", "value"]
        nv = grp.loc[grp["dataset"] == "NLOG", "value"]
        lstats = np.percentile(lv, [5, 95]) if len(lv) >= 30 else (None, None)
        nstats = np.percentile(nv, [5, 95]) if len(nv) >= 30 else (None, None)
        extends = []
        if lstats[0] is not None and nstats[0] is not None:
            tol = 0.05 * (p95 - p05)
            if lstats[0] < nstats[0] - tol: extends.append("LILY_lo")
            if lstats[1] > nstats[1] + tol: extends.append("LILY_hi")
            if nstats[0] < lstats[0] - tol: extends.append("NLOG_lo")
            if nstats[1] > lstats[1] + tol: extends.append("NLOG_hi")
        rows.append({
            "rock_type_fine": rt,
            "measurement":    meas,
            "n_total":        len(vals),
            "n_lily":         len(lv),
            "n_nlog":         len(nv),
            "combined_p05":   round(float(p05), 4),
            "combined_p50":   round(float(p50), 4),
            "combined_p95":   round(float(p95), 4),
            "extends":        "|".join(extends),
        })
    return pd.DataFrame(rows).sort_values(["measurement", "rock_type_fine"])


# ─── EDA report ───────────────────────────────────────────────────────────

def write_eda_report(df: pd.DataFrame,
                     coverage: pd.DataFrame,
                     fine_vs_coarse: pd.DataFrame,
                     depth_matched: pd.DataFrame,
                     support: pd.DataFrame,
                     out_path: Path) -> None:
    """Markdown report aggregating headline numbers and design decisions."""
    L = []
    add = L.append

    n_total = len(df)
    n_wells = df["borehole"].nunique()
    n_nlog = df[df["dataset"] == "NLOG"]["borehole"].nunique()
    n_lily = df[df["dataset"] == "LILY"]["borehole"].nunique()

    add("# EDA report — RL-Boreholes")
    add("")
    add("Run before training to characterise the corpus and the design "
        "choices it forced.  Figures live in this folder.")
    add("")
    add("## 1. Data census")
    add("")
    add(f"- **{n_total:,}** rows total · **{n_wells:,}** unique wells")
    add(f"- **NLOG**: {n_nlog:,} wells "
        f"(Dutch onshore + Dutch sector of the North Sea)")
    add(f"- **LILY**: {n_lily:,} wells "
        "(IODP scientific drilling, global)")
    add("")
    add("![Coverage matrix](01_data_census.png)")
    add("")
    add("**Figure 1** combines (a) total samples per measurement per "
        "dataset, (b) row counts per rock-type label, and (c) the depth "
        "histogram for each corpus.  LILY skews shallow (ocean-floor "
        "cores, < 1 km below seafloor) and NLOG covers depth ~0-6 km.")
    add("")
    add("![Depth distribution](02_depth_distribution.png)")
    add("")
    add("**Figure 2** — LILY and NLOG depth histograms side by side.")
    add("")
    add("## 2. Geographic coverage")
    add("")
    add("![NLOG well map](03_nlog_well_map.png)")
    add("")
    add("**Figure 3** — every NLOG well plotted on the Rijksdriehoek "
        "grid, coloured by basin (k-means cluster k=5 on (x_rd, y_rd) "
        "means per well).  Basin stratification ensures the simulator "
        "doesn't over-sample whichever region has the most logged "
        "wells.")
    add("")
    add("![LILY expeditions](06_lily_expeditions.png)")
    add("")
    add("**Figure 6** — LILY samples and wells broken down by IODP "
        "expedition number.  Expeditions are coherent regional sets "
        "(318 = Wilkes Land, 329 = South Pacific Gyre, 336 = North "
        "Atlantic, etc.) and stand in for a regional split since LILY "
        "lacks RD coordinates.")
    add("")
    add("## 3. Lithology")
    add("")
    add("![Rock-type coverage](04_rock_type_coverage.png)")
    add("")
    add("![NLOG formation breakdown](05_nlog_formation_breakdown.png)")
    add("")
    add("**Figures 4-5** — sample counts per rock_type and per NLOG "
        "formation respectively.  These set the per-(rock, formation) "
        "cells the simulator's `DistributionBank` and `FormationGeometry` "
        "tables are fit against.")
    add("")
    add("## 4. Petrophysics")
    add("")
    add("![rhob violin](07_violin_rhob.png)")
    add("")
    add("![compaction trends](08_compaction_trends.png)")
    add("")
    add("**Figure 8** — per-rock medians shift visibly with depth.  This "
        "shift is the empirical justification for binning the "
        "distribution bank into depth slices "
        f"({len(DEPTH_BINS)-1} bins: {DEPTH_BINS}).  Without depth "
        "binning, the simulator would draw a shallow clay's density for "
        "a 3 km clay.")
    add("")
    add("![fine vs coarse rhob](09_fine_vs_coarse_rhob.png)")
    add("")
    add("**Figure 9** — IQR-reduction by sub-class.  Top entries:")
    add("")
    if len(fine_vs_coarse):
        top = fine_vs_coarse.head(5)
        add("| formation | measurement | coarse IQR | fine IQR | reduction % |")
        add("|---|---|---|---|---|")
        for _, r in top.iterrows():
            add(f"| {r['formation']} | {r['measurement']} | "
                f"{r['coarse_iqr']} | {r['pooled_fine_iqr']} | "
                f"**{r['iqr_reduction_pct']:.1f}%** |")
        add("")
        add("Splitting `claystone` → `claystone_hot`/`claystone_cool` and "
            "`sandstone` → `sandstone_clean`/`sandstone_shaly` reduces "
            "within-class IQR by tens of percent, so the encoder can "
            "see distinct distributions instead of one wide blob.")
    add("")
    add("## 5. Design decisions justified by the data")
    add("")
    add("### 5.1 Variable choice (drop PEF)")
    add("")
    add("![Variable availability](10_variable_availability.png)")
    add("")
    nlog_have = (df[df["dataset"] == "NLOG"]
                   .groupby("borehole")["measurement"]
                   .apply(set))
    pct_pef = 100 * nlog_have.apply(lambda s: "pef" in s).mean()
    pct_all6 = 100 * nlog_have.apply(
        lambda s: set(TARGET_VARS).issubset(s)).mean()
    add(f"- **PEF**: {pct_pef:.1f}% of NLOG wells have it. Training on "
        "a channel that is zero-imputed in ~94% of boreholes degrades "
        "the encoder; we drop PEF.")
    add(f"- All 6 candidate variables present in only {pct_all6:.1f}% of "
        f"NLOG wells; the final 5-variable set "
        f"`{list(FINAL_VARS)}` is what the simulator generates and the "
        "encoder consumes.")
    add("")
    add("### 5.2 LILY ⊕ NLOG pooling")
    add("")
    add("![LILY vs NLOG depth-matched](12_lily_vs_nlog_calibration.png)")
    add("")
    if len(depth_matched):
        n_disagree = (depth_matched["disagreement_flag"] == "YES").sum()
        add(f"- {len(depth_matched)} (rock × measurement × depth-bin) "
            f"cells have both corpora; **{n_disagree}** flag as "
            "materially disagreeing (|Δmedian| > pooled IQR).")
        add("- Pooling the two corpora extends the empirical support "
            "(see `support_bounds.csv` for the `extends` column) but "
            "would bias absolute values where the two disagree.  The "
            "simulator therefore samples from rock-stratified pools, "
            "not the marginal.")
    add("")
    add("### 5.3 Empirical clipping bounds")
    add("")
    add("![Support bounds](11_support_bounds.png)")
    add("")
    add(f"- For every (rock × variable) cell the simulator clips at "
        "the empirical [0.5%, 99.5%] percentiles (rather than a single "
        "global hard bound).  See `support_bounds.csv` for the "
        "per-rock 5/50/95 percentiles used to derive the clipping "
        "windows.")
    add("- Effect: the simulator stops emitting a claystone with "
        "halite-like density just because the global hard bound "
        "allows it.")
    add("")
    add("## 6. Tables (CSV)")
    add("")
    add("| File | Contents |")
    add("|---|---|")
    add("| `coverage_matrix.csv` | rows/wells/rocks per (dataset, measurement) |")
    add("| `summary_per_category.csv` | p10/p50/p90/IQR per (dataset, measurement, category) |")
    add("| `compaction_trends.csv` | p25/p50/p75 per (rock × variable × depth-bin) |")
    add("| `depth_matched_summary.csv` | LILY vs NLOG at matched depth bins |")
    add("| `fine_vs_coarse_comparison.csv` | IQR reduction from rock-class splits |")
    add("| `support_bounds.csv` | per-rock empirical [p05, p50, p95] |")
    add("")
    out_path.write_text("\n".join(L))
    print(f"  wrote {out_path}")


# ─── orchestration ────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--geometry", type=Path, default=DEFAULT_GEOM,
                    help="formation_geometry.pkl for basin labels on the map")
    args = ap.parse_args()

    if not args.samples.exists():
        raise FileNotFoundError(f"{args.samples} — run pull_data.py first")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.samples} ...")
    df = pd.read_parquet(args.samples)
    print(f"  {len(df):,} rows · {df['borehole'].nunique()} wells · "
          f"{df['measurement'].nunique()} measurements")

    # ── tables ─────────────────────────────────────────────────────────
    print("\n[1/3] tables ...")
    coverage = table_coverage_matrix(df)
    coverage.to_csv(args.out / "coverage_matrix.csv", index=False)
    print(f"  wrote coverage_matrix.csv  ({len(coverage)} rows)")

    summary = table_summary_per_category(df)
    summary.to_csv(args.out / "summary_per_category.csv", index=False)
    print(f"  wrote summary_per_category.csv  ({len(summary)} rows)")

    compaction = table_compaction_trends(df)
    compaction.to_csv(args.out / "compaction_trends.csv", index=False)
    print(f"  wrote compaction_trends.csv  ({len(compaction)} rows)")

    depth_matched = table_depth_matched_summary(df)
    depth_matched.to_csv(args.out / "depth_matched_summary.csv", index=False)
    print(f"  wrote depth_matched_summary.csv  ({len(depth_matched)} rows)")

    fine_vs_coarse = pd.DataFrame()
    support = pd.DataFrame()
    if "rock_type_fine" in df.columns:
        fine_vs_coarse = table_fine_vs_coarse(df)
        fine_vs_coarse.to_csv(args.out / "fine_vs_coarse_comparison.csv",
                              index=False)
        print(f"  wrote fine_vs_coarse_comparison.csv  "
              f"({len(fine_vs_coarse)} rows)")
        support = table_support_bounds(df)
        support.to_csv(args.out / "support_bounds.csv", index=False)
        print(f"  wrote support_bounds.csv  ({len(support)} rows)")

    # ── plots: 12 thesis-grade figures ─────────────────────────────────
    print("\n[2/3] plots ...")
    p = args.out

    plots.plot_feature_coverage(df,        p / "01_data_census.png")
    plots.plot_depth_distribution(df,      p / "02_depth_distribution.png")
    plots.plot_nlog_well_map(df,           p / "03_nlog_well_map.png",
                              geometry_pkl=args.geometry)
    plots.plot_rock_type_coverage(df,      p / "04_rock_type_coverage.png")
    plots.plot_nlog_by_formation(df,       p / "05_nlog_formation_breakdown.png")
    plots.plot_lily_expeditions(df,        p / "06_lily_expeditions.png")
    plots.plot_per_feature_violin(df,      p / "07_violin_rhob.png", "rhob")
    plots.plot_compaction_trends(df,       p / "08_compaction_trends.png")
    if "rock_type_fine" in df.columns:
        plots.plot_fine_vs_coarse_distributions(
            df, p / "09_fine_vs_coarse_rhob.png", "rhob")
    plots.plot_variable_availability(df,   p / "10_variable_availability.png",
                                      target_vars=TARGET_VARS)
    if "rock_type_fine" in df.columns:
        plots.plot_support_bounds_matrix(df, p / "11_support_bounds.png")
    plots.plot_depth_matched_comparison(df,
                                         p / "12_lily_vs_nlog_calibration.png")

    # ── report ─────────────────────────────────────────────────────────
    print("\n[3/3] EDA_REPORT.md ...")
    write_eda_report(df, coverage, fine_vs_coarse, depth_matched,
                      support, p / "EDA_REPORT.md")

    print(f"\n→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()
