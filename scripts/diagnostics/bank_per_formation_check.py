"""
Per-(formation, rock) consistency check at 10 m bins.

Aggregate KS already shows the bank reproduces real data with median
KS ~0.05. But the bank is fit per (rock, depth_bin) WITHOUT a formation
key. At map generation, the simulator first picks a formation, then
rocks inside that formation, then calls the bank. If `claystone_hot`
inside RO has different petrophysics than `claystone_hot` inside DC,
the bank would average them -- and synthetic data inside each formation
would be biased even though the aggregate is fine.

This script tests for that bias.

Probe panel: every (formation in {CK, KN, ZE, RO, DC, RB, SL, SG},
top-3 rocks within that formation by row count, depth in
{1000, 1500, 2000, 2500, 3000, 3500} m). For each cell:

  - truth = real samples filtered by formation==fm AND rock==rock
            AND depth in [d-5, d+5]
  - synth = bank.sample(rock, d, n=5000)  -- bank has no formation key
  - KS per variable

Outputs:
    plots/analysis/bank_per_formation_ks.csv
    plots/bank_per_formation_ks.png
    plots/bank_per_formation_overlays/<fm>_<rock>_<depth>.png
        -- only for cells with KS >= PROBLEM_KS in any variable
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from simulator.distributions import DistributionBank

VARIABLES   = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
N_DRAWS     = 5000
TRUTH_W     = 10.0
MIN_TRUTH_N = 60          # need this many real samples to bother
PROBLEM_KS  = 0.15

# (formation, depth_m) probe ladder. For each formation we only test
# the depths that have real data; the per-cell loop skips empty cells.
FORMATIONS = ["CK", "KN", "ZE", "RO", "DC", "RB", "SL", "SG"]
DEPTHS_M   = [1000, 1500, 2000, 2500, 3000, 3500]


def real_window(df, formation, rock, lo, hi, depth_col):
    sub = df[
        (df["formation"] == formation)
        & (df["rock_type_fine"] == rock)
        & (df[depth_col] >= lo)
        & (df[depth_col] < hi)
    ]
    if not len(sub):
        return pd.DataFrame()
    return sub.pivot_table(
        index=["dataset", "borehole", depth_col],
        columns="measurement", values="value", aggfunc="mean",
    ).reset_index()


def ks(a, b):
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def dominant_rocks(df: pd.DataFrame, formation: str, top: int = 3) -> list[str]:
    sub = (df[(df["formation"] == formation)
             & (df["measurement"] == "rhob")]
              .groupby("rock_type_fine").size()
              .sort_values(ascending=False))
    rocks = [r for r in sub.head(top).index
             if isinstance(r, str) and r and r != "other"]
    return rocks


def make_overlay(formation, rock, depth, truth, synth, out: Path,
                 ks_per_var: dict):
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(15, 3))
    fig.suptitle(f"{formation} / {rock} @ {int(depth)} m  "
                 f"(real n={len(truth)})", fontsize=11, y=1.02)
    for ax, v in zip(axes, VARIABLES):
        r = truth.get(v, pd.Series(dtype=float)).dropna().values
        s = np.asarray(synth.get(v, np.array([])))
        s = s[np.isfinite(s)]
        if len(r) >= 30 and len(s) >= 30:
            lo = float(min(np.percentile(r, 1), np.percentile(s, 1)))
            hi = float(max(np.percentile(r, 99), np.percentile(s, 99)))
            bins = np.linspace(lo, hi, 40)
            ax.hist(r, bins=bins, alpha=0.55, color="#4C72B0",
                    label=f"real (n={len(r)})", density=True)
            ax.hist(s, bins=bins, alpha=0.55, color="#d6604d",
                    label="synth", density=True)
        k = ks_per_var.get(v, float("nan"))
        title = v
        if not np.isnan(k):
            title += f"  KS={k:.2f}"
            if k >= PROBLEM_KS:
                title += " *"
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        if v == VARIABLES[0]:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path,
                   default=Path("data/clean/distributions.pkl"))
    p.add_argument("--parquet", type=Path,
                   default=Path("data/clean/samples.parquet"))
    p.add_argument("--use-burial", action="store_true",
                   help="filter real data by `burial_depth_m` instead "
                        "of `depth`. Use this for the post-burial run.")
    p.add_argument("--out-suffix", default="",
                   help="suffix appended to output filenames so two runs "
                        "(pre-burial baseline + post-burial corrected) "
                        "can write distinct files. Example: '_pre_burial'.")
    args = p.parse_args()

    depth_col = "burial_depth_m" if args.use_burial else "depth"
    print(f"using depth column: {depth_col}")
    print(f"bank: {args.bank}")
    print(f"parquet: {args.parquet}")

    out_csv = Path(f"plots/analysis/bank_per_formation_ks{args.out_suffix}.csv")
    out_heatmap = Path(f"plots/bank_per_formation_ks{args.out_suffix}.png")

    bank = DistributionBank.load(args.bank)
    df = pd.read_parquet(args.parquet)
    if depth_col not in df.columns:
        raise ValueError(
            f"parquet {args.parquet} missing column {depth_col!r}. "
            f"Available: {[c for c in df.columns if 'depth' in c.lower()]}"
        )
    df = df[df["measurement"].isin(VARIABLES)]
    print(f"bank: {len(bank.cells)} populated cells")

    rng = np.random.default_rng(0)
    rows = []
    problem_cells = []

    for fm in FORMATIONS:
        rocks = dominant_rocks(df, fm)
        if not rocks:
            print(f"{fm}: no dominant rock found (skip)")
            continue
        print(f"{fm}: dominant rocks -> {rocks}")
        for rock in rocks:
            for d in DEPTHS_M:
                truth = real_window(df, fm, rock,
                                    d - TRUTH_W / 2, d + TRUTH_W / 2,
                                    depth_col=depth_col)
                if len(truth) < MIN_TRUTH_N:
                    continue
                synth = bank.sample(rock, float(d), n=N_DRAWS, rng=rng,
                                    interpolate=True, formation=fm)
                ks_per_var = {}
                worst = 0.0
                for v in VARIABLES:
                    tv = truth.get(v, pd.Series(dtype=float)).dropna().values
                    sv = np.asarray(synth.get(v, np.array([])))
                    k = ks(tv, sv)
                    ks_per_var[v] = k
                    rows.append({
                        "formation": fm, "rock": rock, "depth": d,
                        "var": v, "n_truth": len(tv),
                        "ks": k,
                    })
                    if not np.isnan(k):
                        worst = max(worst, k)
                tag = " *" if worst >= PROBLEM_KS else "  "
                print(f"  {tag} {rock:18s} @ {d:>5d} m  n={len(truth):>5d}  "
                      f"worst KS={worst:.2f}")
                if worst >= PROBLEM_KS:
                    problem_cells.append((fm, rock, d, truth, synth, ks_per_var))

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ---- heatmap: rows=(formation, rock, depth), cols=variables -------
    pivot = (out.dropna(subset=["ks"])
                .assign(cell=lambda d: d.apply(
                    lambda r: f"{r['formation']}/{r['rock']}@{int(r['depth'])}m",
                    axis=1))
                .pivot_table(index="cell", columns="var", values="ks"))
    pivot = pivot[VARIABLES]  # consistent column order

    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.3,
                   aspect="auto")
    cb = fig.colorbar(im, ax=ax, label="KS (bank synth vs real)")
    ax.set_xticks(range(len(VARIABLES))); ax.set_xticklabels(VARIABLES,
                                                              fontsize=8)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index,
                                                          fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if v > 0.2 else "black")
    ax.set_title(f"Per-(formation, rock, depth) bank-vs-real KS "
                 f"(10 m bins, TVD)\nRed cells (KS>={PROBLEM_KS}) "
                 f"are 'problem cells'.")
    fig.tight_layout()
    out_heatmap.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_heatmap, dpi=140, bbox_inches="tight")
    print(f"wrote {out_heatmap}")

    # ---- overlay panels for problem cells -------------------------------
    overlay_dir = Path(f"plots/bank_per_formation_overlays{args.out_suffix}")
    print(f"\n{len(problem_cells)} problem cells (KS >= {PROBLEM_KS} in any "
          f"variable). Rendering overlays...")
    if problem_cells:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for fm, rock, d, truth, synth, ks_per_var in problem_cells:
            out_png = overlay_dir / f"{fm}_{rock}_{int(d)}m.png"
            make_overlay(fm, rock, d, truth, synth, out_png, ks_per_var)
            print(f"  wrote {out_png}")
    else:
        print("  none. Bank is formation-consistent at the probe panel.")

    # ---- aggregate summary for cross-run comparison ----
    finite = out.dropna(subset=["ks"])
    if len(finite):
        n_cells = len(finite)
        med = float(finite["ks"].median())
        p75 = float(finite["ks"].quantile(0.75))
        p95 = float(finite["ks"].quantile(0.95))
        n_prob = int((finite["ks"] >= PROBLEM_KS).sum())
        print(f"\n=== summary{args.out_suffix} ===")
        print(f"  cells:        {n_cells}")
        print(f"  median KS:    {med:.4f}")
        print(f"  P75 KS:       {p75:.4f}")
        print(f"  P95 KS:       {p95:.4f}")
        print(f"  problem cells (KS>={PROBLEM_KS}): {n_prob} "
              f"({100*n_prob/n_cells:.1f}%)")


if __name__ == "__main__":
    main()
