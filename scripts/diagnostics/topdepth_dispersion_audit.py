"""
Phase 1 -- Diagnose why the simulator's formation-top depths look
under-dispersed against real NLOG (Garzon position metric:
Wasserstein-1 137-517 m per formation).

Four checks, each producing one numeric verdict:

  1. Strip well-start contamination from real-side tops.
     Many NLOG wells START at hundreds of metres (oil/gas wells
     skip the shallow). The real "top" we compute as min(depth)
     for the formation is therefore biased upward.  Re-compute
     using only wells that truly drilled the formation's
     typical top region.  If filtered Wasserstein-1 drops by
     >= 50%, the original number was artefact, not under-dispersion.

  2. Inspect FormationGeometry's stored top-depth KDEs.
     Plot the per-formation KDE alongside its underlying real-data
     histogram + a sim resampling.  Is the KDE bandwidth narrow?

  3. Retry-loop fallback rate in sample_column.
     Instrument _sample_tops_with_constraints to count how often
     the 20-retry loop falls through to the median-collapse path.

  4. Combination diversity.
     Count unique formation combinations across 1,000 sim draws;
     compare to NLOG combinations (wells reaching >= 4000 m).

Outputs:
  plots/garcon/topdepth_check1_filtered.png  (filtered Wasserstein)
  plots/garcon/topdepth_check2_kde.png       (KDE vs real vs sim)
  plots/analysis/topdepth_dispersion_report.md (verdict A1/A2/A3/A4)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator import sample_column
from simulator.formation_geometry import FormationGeometry

TARGET_FORMATIONS = [
    "NU", "NM", "NL", "CK", "KN", "AT", "SL", "SG",
    "ZE", "RO", "RB", "RN", "DC", "SK",
]


# ----------------------------------------------------- check 1: well-start --

def real_tops_with_well_start(df: pd.DataFrame) -> pd.DataFrame:
    """Per (well, formation) top depth + well-start depth.

    Returns dataframe with columns:
        borehole, formation, top, well_start_depth
    """
    nlog = df[df.dataset == "NLOG"].copy()
    well_start = nlog.groupby("borehole").depth.min().rename("well_start_depth")
    tops = (nlog.groupby(["borehole", "formation"]).depth.min()
            .rename("top").reset_index())
    tops = tops.merge(well_start.reset_index(), on="borehole", how="left")
    return tops[tops.formation.isin(TARGET_FORMATIONS)]


def sim_tops(geom: FormationGeometry, n: int = 1000,
             max_depth: float = 4400.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        col = sample_column(rng=rng, geometry=geom, max_depth=max_depth)
        for fm, _rocks, top, _bot in col.layers:
            if fm in TARGET_FORMATIONS:
                rows.append({"borehole": f"SIM_{i:05d}",
                             "formation": fm, "top": float(top)})
    return pd.DataFrame(rows)


def check1_well_start(real_tops_df: pd.DataFrame, sim_tops_df: pd.DataFrame,
                      out_png: Path) -> tuple[pd.DataFrame, float]:
    """For each formation, compute Wasserstein-1 (sim, real) for:
      (a) all real wells
      (b) wells where well_start_depth <= median sim top - 100m
          (i.e. wells that genuinely drilled the formation's
          top region)
    """
    rows = []
    for fm in TARGET_FORMATIONS:
        sim_t = sim_tops_df[sim_tops_df.formation == fm].top.values
        if len(sim_t) < 10:
            continue
        sim_median = float(np.median(sim_t))
        real_fm = real_tops_df[real_tops_df.formation == fm]
        # all real
        r_all = real_fm.top.values
        # filtered: well started above (sim_median - 100m), so it
        # genuinely intersected the formation's typical top region
        r_filt = real_fm[real_fm.well_start_depth <= sim_median - 100].top.values
        w_all = float(sps.wasserstein_distance(r_all, sim_t)) if len(r_all) > 5 else np.nan
        w_filt = float(sps.wasserstein_distance(r_filt, sim_t)) if len(r_filt) > 5 else np.nan
        rows.append({
            "formation": fm,
            "n_real_all": len(r_all),
            "n_real_filt": len(r_filt),
            "n_sim": len(sim_t),
            "W1_all_m": w_all,
            "W1_filtered_m": w_filt,
            "drop_pct": (1 - w_filt / w_all) * 100
                if (not np.isnan(w_filt) and not np.isnan(w_all)
                    and w_all > 0) else np.nan,
            "sim_median": sim_median,
        })
    df = pd.DataFrame(rows)

    # figure
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.W1_all_m, width=0.4, color="C3",
           label="W1: all real wells")
    ax.bar(x + 0.2, df.W1_filtered_m, width=0.4, color="C2",
           label="W1: filtered (well_start <= sim_median - 100m)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.formation)
    ax.set_ylabel("Wasserstein-1 distance (m)")
    ax.set_title("Check 1: Wasserstein-1 sim vs real, "
                 "before and after filtering on well-start depth")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    for xi, drop in zip(x, df.drop_pct.fillna(0)):
        ax.annotate(f"{drop:.0f}%" if drop > 0 else "",
                    (xi, max(df.W1_all_m.iloc[xi] or 0,
                              df.W1_filtered_m.iloc[xi] or 0) + 10),
                    ha="center", fontsize=8, color="darkgreen")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")

    median_drop = float(np.nanmedian(df.drop_pct))
    return df, median_drop


# --------------------------------------------- check 2: KDE bandwidth ------

def check2_kde_inspection(geom: FormationGeometry,
                          real_tops_df: pd.DataFrame,
                          sim_tops_df: pd.DataFrame,
                          out_png: Path) -> pd.DataFrame:
    """For each formation, plot:
      - real top histogram (NLOG)
      - stored KDE evaluated densely
      - 1000 KDE-resampled draws (what sim_top_depth produces)
      - sim draws actually used (post-constraint)
    Report KDE bandwidth and the standard deviations.
    """
    rows = []
    fms_with_kde = [fm for fm in TARGET_FORMATIONS
                    if fm in geom.formations
                    and geom.formations[fm].top_depths is not None
                    and len(geom.formations[fm].top_depths) > 5]
    ncols = 3
    nrows = (len(fms_with_kde) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3 * nrows),
                              squeeze=False)
    rng = np.random.default_rng(0)
    for i, fm in enumerate(fms_with_kde):
        ax = axes[i // ncols][i % ncols]
        stats_fm = geom.formations[fm]
        real_top = real_tops_df[real_tops_df.formation == fm].top.values
        sim_post = sim_tops_df[sim_tops_df.formation == fm].top.values
        td = stats_fm.top_depths
        if stats_fm._top_kde is None:
            stats_fm._top_kde = sps.gaussian_kde(td)
        kde_resample = stats_fm._top_kde.resample(1000, seed=rng).ravel()
        kde_resample = np.clip(kde_resample, 0, None)

        lo = float(min(td.min(), sim_post.min() if len(sim_post) else td.min()))
        hi = float(max(td.max(), sim_post.max() if len(sim_post) else td.max()))
        bins = np.linspace(lo - 50, hi + 50, 40)

        ax.hist(real_top, bins=bins, density=True, alpha=0.45,
                color="C0", label=f"real wells (n={len(real_top)})")
        ax.hist(kde_resample, bins=bins, density=True, alpha=0.35,
                color="C2", label="KDE resample (1000)")
        ax.hist(sim_post, bins=bins, density=True, alpha=0.35,
                color="C1", label=f"sim post-constraint (n={len(sim_post)})")
        ax.set_title(f"{fm}  td_n={len(td)} "
                     f"std={td.std():.0f}m  bw≈{stats_fm._top_kde.factor:.3f}")
        ax.set_xlabel("top depth (m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        rows.append({
            "formation": fm,
            "n_fit": len(td),
            "td_std_m": float(td.std()),
            "kde_resample_std_m": float(np.std(kde_resample)),
            "sim_post_std_m": float(np.std(sim_post)) if len(sim_post) else np.nan,
            "real_std_m": float(np.std(real_top)),
            "kde_bw_factor": float(stats_fm._top_kde.factor),
        })
    for j in range(len(fms_with_kde), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Check 2: Top-depth KDE inspection per formation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    return pd.DataFrame(rows)


# --------------------------- check 3: retry-loop fallback firing rate -----

def check3_retry_loop(geom: FormationGeometry, n: int = 1000,
                      max_depth: float = 4400.0) -> tuple[float, int, int]:
    """Re-implement the retry loop with a fallback counter.

    Returns (fallback_rate, n_total, n_fallback).
    """
    from simulator.formation_geometry import MIN_LAYER_THICKNESS_DEFAULT
    min_t = MIN_LAYER_THICKNESS_DEFAULT
    max_retries = 20
    rng = np.random.default_rng(7)
    n_fallback = 0
    for _ in range(n):
        # pick a combination same way as sample_column
        if geom.combinations_by_basin:
            basin_keys = sorted(geom.combinations_by_basin.keys())
            basin = basin_keys[int(rng.integers(0, len(basin_keys)))]
            combos, counts = zip(*geom.combinations_by_basin[basin])
        else:
            combos, counts = zip(*geom.combinations)
        w = np.array(counts, dtype=np.float64); w /= w.sum()
        combination = combos[int(rng.choice(len(combos), p=w))]
        n_fm = len(combination)
        ok_attempt = -1
        for attempt in range(max_retries):
            sampled = []
            for fm in combination:
                s = geom.formations.get(fm)
                if s is None:
                    sampled.append(None)
                else:
                    sampled.append(s.sample_top_depth(rng))
            for i, t in enumerate(sampled):
                if t is None:
                    if i == 0:
                        sampled[i] = 0.0
                    else:
                        sampled[i] = sampled[i - 1] + min_t
            if sampled[-1] > max_depth - min_t:
                sampled[-1] = max_depth - min_t
            if sampled[0] < 0:
                sampled[0] = 0.0
            ok = True
            for i in range(1, n_fm):
                if sampled[i] < sampled[i - 1] + min_t:
                    ok = False
                    break
            if ok:
                ok_attempt = attempt
                break
        if ok_attempt == -1:
            n_fallback += 1
    rate = n_fallback / n
    return rate, n, n_fallback


# --------------------------- check 4: combination diversity ---------------

def check4_combination_diversity(geom: FormationGeometry,
                                  df: pd.DataFrame,
                                  n_sim: int = 1000) -> dict:
    rng = np.random.default_rng(13)
    sim_combos: list[tuple[str, ...]] = []
    for _ in range(n_sim):
        col = sample_column(rng=rng, geometry=geom, max_depth=4400.0)
        sim_combos.append(tuple(fm for fm, _, _, _ in col.layers
                                 if fm in TARGET_FORMATIONS))
    sim_unique = len(set(sim_combos))
    sim_counts = pd.Series(sim_combos).value_counts()
    sim_top3_frac = float(sim_counts.head(3).sum() / len(sim_combos))

    # real combinations: per well, ordered list of formations by min(depth)
    nlog = df[df.dataset == "NLOG"].copy()
    well_depth = nlog.groupby("borehole").depth.max()
    deep_wells = well_depth[well_depth >= 4000.0].index
    real = nlog[nlog.borehole.isin(deep_wells)]
    real_combos = []
    for _, g in real.groupby("borehole"):
        order = (g.groupby("formation").depth.min().sort_values().index.tolist())
        order = tuple(o for o in order if o in TARGET_FORMATIONS)
        if order:
            real_combos.append(order)
    real_unique = len(set(real_combos))
    real_counts = pd.Series(real_combos).value_counts()
    real_top3_frac = float(real_counts.head(3).sum() / len(real_combos)) if real_combos else np.nan

    return {
        "n_sim_draws": n_sim,
        "sim_unique_combos": sim_unique,
        "sim_top3_frac": sim_top3_frac,
        "n_real_wells_>=4000m": len(real_combos),
        "real_unique_combos": real_unique,
        "real_top3_frac": real_top3_frac,
    }


# ----------------------------------------------------------- report writer --

def write_report(out_path: Path,
                 c1: pd.DataFrame, c1_median_drop: float,
                 c2: pd.DataFrame,
                 c3_rate: float, c3_total: int, c3_fb: int,
                 c4: dict) -> None:
    lines = [
        "# Top-depth dispersion audit -- is the simulator really "
        "under-dispersed?",
        "",
        "*Generated by `scripts/diagnostics/topdepth_dispersion_audit.py`.*",
        "",
        "## Check 1 -- well-start contamination",
        "",
        "For each real well we recorded its actual start depth "
        "(`min(depth)`).  Oil/gas wells often start hundreds of "
        "metres below the shallow Cenozoic; their 'top' for a "
        "shallow formation is then the well's start depth, not the "
        "formation's geological top.  We filter to wells whose "
        "start depth is more than 100 m above the simulator's "
        "median sim top for that formation, then recompute "
        "Wasserstein-1.",
        "",
        "![check 1](../garcon/topdepth_check1_filtered.png)",
        "",
        "| formation | W1 all (m) | W1 filtered (m) | drop |",
        "|---|---|---|---|",
    ]
    for _, r in c1.iterrows():
        d = ("--" if np.isnan(r.drop_pct)
             else f"{r.drop_pct:.0f}%")
        wa = ("--" if np.isnan(r.W1_all_m) else f"{r.W1_all_m:.0f}")
        wf = ("--" if np.isnan(r.W1_filtered_m) else f"{r.W1_filtered_m:.0f}")
        lines.append(f"| {r.formation} | {wa} | {wf} | {d} |")
    lines += [
        "",
        f"**Median drop after filtering: {c1_median_drop:.0f}%.**",
        "",
        "## Check 2 -- top-depth KDE bandwidth",
        "",
        "Per formation, compare the real-data standard deviation "
        "of the fitted tops, the KDE-resample standard deviation, "
        "and the sim-post-constraint standard deviation.",
        "",
        "![check 2](../garcon/topdepth_check2_kde.png)",
        "",
        "| formation | n_fit | real std (m) | KDE-resample std (m) | "
        "sim post std (m) | bw factor |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in c2.iterrows():
        lines.append(
            f"| {r.formation} | {int(r.n_fit)} | {r.td_std_m:.0f} | "
            f"{r.kde_resample_std_m:.0f} | "
            f"{'--' if np.isnan(r.sim_post_std_m) else f'{r.sim_post_std_m:.0f}'}"
            f" | {r.kde_bw_factor:.3f} |"
        )
    # is the sim-post std systematically < real std?
    valid = c2.dropna(subset=["sim_post_std_m"])
    if len(valid):
        sim_to_real_std_ratio = float(np.median(
            valid.sim_post_std_m / valid.real_std_m.replace(0, 1)))
    else:
        sim_to_real_std_ratio = np.nan
    lines += [
        "",
        f"Median ratio of sim-post-std / real-std across formations: "
        f"**{sim_to_real_std_ratio:.2f}** "
        f"(1.0 = simulator matches real spread; "
        f"<<1.0 = simulator under-disperses).",
        "",
        "## Check 3 -- retry-loop fallback rate",
        "",
        f"Out of {c3_total} simulated columns, "
        f"**{c3_fb} ({c3_rate*100:.1f}%) fell through to the median-collapse "
        f"path** after exhausting the 20-retry loop.",
        "",
        "## Check 4 -- combination diversity",
        "",
        f"In {c4['n_sim_draws']} sim draws: "
        f"**{c4['sim_unique_combos']} unique formation combinations**, "
        f"top-3 combinations cover **{c4['sim_top3_frac']*100:.0f}%** of draws.",
        "",
        f"In the {c4['n_real_wells_>=4000m']} real NLOG wells reaching "
        f">=4000m: **{c4['real_unique_combos']} unique combinations**, "
        f"top-3 cover **{c4['real_top3_frac']*100:.0f}%**.",
        "",
        "## Verdict",
        "",
    ]
    # auto-classify
    a1 = c1_median_drop >= 50
    a2 = (not np.isnan(sim_to_real_std_ratio)) and sim_to_real_std_ratio < 0.6
    a3 = c3_rate >= 0.30
    a4 = c4["sim_top3_frac"] - c4["real_top3_frac"] >= 0.20

    parts = []
    if a1:
        parts.append("**A1 fires** -- well-start contamination explains "
                     ">=50% of the apparent under-dispersion. Treat the "
                     "original Garzón position number as an artefact and "
                     "rebaseline it using the filtered comparison.")
    if a2:
        parts.append("**A2 fires** -- the simulator's post-constraint top "
                     "spread is < 60% of real spread. KDE bandwidth or "
                     "constraint enforcement is collapsing variance. "
                     "Suggested fix: widen the KDE bandwidth via the "
                     "`bw_method` argument to `gaussian_kde`, or replace "
                     "the resample call with `resample-with-noise`.")
    if a3:
        parts.append("**A3 fires** -- the 20-retry loop falls through to "
                     "the median path more than 30% of the time. "
                     "Suggested fix: raise `MAX_RESAMPLES`, relax "
                     "`MIN_LAYER_THICKNESS_DEFAULT`, or use a random "
                     "fallback instead of the median.")
    if a4:
        parts.append("**A4 fires** -- the top-3 sim combinations dominate "
                     "by 20+ percentage points more than real. Basin-"
                     "stratified sampling is collapsing diversity. "
                     "Suggested fix: cap the per-combination weight in "
                     "the basin sampler.")
    if not parts:
        parts.append("**None of A1-A4 fire above their thresholds.** "
                     "The under-dispersion is real but not explained by "
                     "any single mechanism; a more careful look is "
                     "warranted before changing anything.")
    lines += parts + [""]

    # Phase 2 implication
    lines += [
        "## Implication for Phase 2 (RVG integration)",
        "",
    ]
    if a1 and not (a2 or a3 or a4):
        lines.append(
            "The position-metric finding was an artefact. RVG "
            "integration still stands on its own merits "
            "(NM/NL shallow coverage, transition-matrix shift) "
            "but should NOT be framed as 'fixing under-dispersion'."
        )
    elif a2 or a3 or a4:
        lines.append(
            "There is a genuine simulator defect (A2/A3/A4). "
            "Apply the small fix BEFORE RVG integration. After "
            "the fix, Phase 2's value-add is cleaner to measure."
        )
    else:
        lines.append(
            "Inconclusive. Bring this report back to the user "
            "for a decision before continuing to Phase 2."
        )
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# -------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path,
                   default=Path("data/clean/samples.parquet"))
    p.add_argument("--geom", type=Path,
                   default=Path("data/clean/formation_geometry.pkl"))
    p.add_argument("--n-sim", type=int, default=1000)
    p.add_argument("--out-dir", type=Path, default=Path("plots/garcon"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path("plots/analysis").mkdir(parents=True, exist_ok=True)

    print("loading NLOG samples...")
    df = pd.read_parquet(args.samples)
    real = real_tops_with_well_start(df)
    print(f"  real: {real.borehole.nunique()} wells, {len(real)} (well,fm) tops")

    print("loading FormationGeometry + sampling sim columns...")
    geom = FormationGeometry.load(args.geom)
    sim = sim_tops(geom, n=args.n_sim)

    print("\n[1] well-start contamination filter...")
    c1, c1_drop = check1_well_start(real, sim,
                                     args.out_dir / "topdepth_check1_filtered.png")

    print("[2] KDE inspection...")
    c2 = check2_kde_inspection(geom, real, sim,
                                args.out_dir / "topdepth_check2_kde.png")

    print("[3] retry-loop fallback rate...")
    c3_rate, c3_total, c3_fb = check3_retry_loop(geom, n=args.n_sim)
    print(f"  fallback fired {c3_fb}/{c3_total} = {c3_rate*100:.1f}%")

    print("[4] combination diversity...")
    c4 = check4_combination_diversity(geom, df, n_sim=args.n_sim)
    print(f"  sim unique={c4['sim_unique_combos']}, "
          f"top3 frac={c4['sim_top3_frac']:.2f}; "
          f"real unique={c4['real_unique_combos']}, "
          f"top3 frac={c4['real_top3_frac']:.2f}")

    write_report(Path("plots/analysis/topdepth_dispersion_report.md"),
                 c1, c1_drop, c2, c3_rate, c3_total, c3_fb, c4)


if __name__ == "__main__":
    main()
