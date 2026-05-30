"""
TVD-aligned depth coverage check.

After the TVD fix in 2_pull_data.py, every NLOG well's output `depth`
is its true vertical depth instead of its measured depth. Inclined
wells therefore reach SHALLOWER TVD than the previous MD-aligned data
had them. The simulator's `max_depth = 4400 m` was chosen because
it covered ~90% of NLOG hc-positive wells in MD; in TVD it may
cover fewer.

This script:
  - computes per-well max depth in the TVD-aligned parquet,
    separately for hc-positive and hc-negative wells (LILY excluded);
  - plots a CDF of max-depth per well;
  - reports the depth that covers 85% / 90% / 95% of hc-positive
    wells, so we can defend (or revise) the 4400 m simulator window.

Outputs:
    plots/tvd_depth_coverage.png
    plots/analysis/tvd_depth_coverage_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARQUET = Path("data/clean/samples.parquet")
OUT_PNG = Path("plots/tvd_depth_coverage.png")
OUT_CSV = Path("plots/analysis/tvd_depth_coverage_summary.csv")

SIM_MAX_DEPTH = 4400.0


def main() -> None:
    df = pd.read_parquet(PARQUET)
    nlog = df[df["dataset"] == "NLOG"].copy()
    print(f"NLOG rows: {len(nlog):,}")

    # one row per well with its max-TVD-sample and hc flag
    per_well = (nlog.groupby("borehole")
                    .agg(max_tvd=("depth", "max"),
                         hc=("hc_discovery", "first"))
                    .reset_index())
    print(f"NLOG wells in parquet: {len(per_well):,}")
    print(f"  hc_discovery = True : {int((per_well['hc'] == True).sum()):,}")
    print(f"  hc_discovery = False: {int((per_well['hc'] == False).sum()):,}")
    print(f"  hc_discovery = NA  : {int(per_well['hc'].isna().sum()):,}")

    pos = per_well[per_well["hc"] == True]["max_tvd"].values
    neg = per_well[per_well["hc"] == False]["max_tvd"].values
    allw = per_well["max_tvd"].values

    def pct_below(x: np.ndarray, depth: float) -> float:
        if len(x) == 0:
            return float("nan")
        return float((x < depth).mean() * 100)

    def depth_at_pct(x: np.ndarray, pct: float) -> float:
        if len(x) == 0:
            return float("nan")
        return float(np.percentile(x, pct))

    print(f"\n@ {SIM_MAX_DEPTH:.0f} m TVD (the simulator's max_depth):")
    print(f"  hc-positive wells reaching or exceeding it: "
          f"{100 - pct_below(pos, SIM_MAX_DEPTH):.1f}%")
    print(f"  hc-negative wells reaching or exceeding it: "
          f"{100 - pct_below(neg, SIM_MAX_DEPTH):.1f}%")
    print(f"  all NLOG wells reaching or exceeding it   : "
          f"{100 - pct_below(allw, SIM_MAX_DEPTH):.1f}%")

    print(f"\nmax-TVD percentiles (hc-positive wells):")
    for p in [50, 70, 80, 85, 90, 95]:
        print(f"  P{p:>2d}  {depth_at_pct(pos, p):>7.0f} m")

    rows = []
    for label, x in [("hc_positive", pos), ("hc_negative", neg), ("all", allw)]:
        for p in [50, 70, 80, 85, 90, 95]:
            rows.append({"label": label, "pct": p,
                         "depth_m": depth_at_pct(x, p)})
        rows.append({"label": label, "pct": None,
                     "depth_m": SIM_MAX_DEPTH,
                     "frac_reaches_max_depth_pct":
                         100 - pct_below(x, SIM_MAX_DEPTH)})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # CDF plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, x, c in [("hc-positive", pos, "#d6604d"),
                        ("hc-negative", neg, "#4C72B0"),
                        ("all NLOG",   allw, "#888")]:
        if len(x) == 0:
            continue
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        ax.plot(xs, ys, color=c, linewidth=1.6, label=f"{label}  (n={len(x):,})")

    ax.axvline(SIM_MAX_DEPTH, color="black", linestyle="--", linewidth=1)
    ax.text(SIM_MAX_DEPTH + 50, 5,
            f"simulator max_depth\n= {SIM_MAX_DEPTH:.0f} m",
            fontsize=9)

    # annotate fraction of hc-positive at the simulator threshold
    if len(pos):
        below = pct_below(pos, SIM_MAX_DEPTH)
        ax.scatter([SIM_MAX_DEPTH], [below],
                   color="#d6604d", zorder=5, s=50)
        ax.text(SIM_MAX_DEPTH - 200, below + 3,
                f"{below:.0f}% of hc-positive wells stop\n"
                f"shallower than {SIM_MAX_DEPTH:.0f} m TVD",
                ha="right", fontsize=9, color="#d6604d")

    ax.set_xlabel("max TVD depth reached by the well [m]")
    ax.set_ylabel("% wells with max depth below x")
    ax.set_title("TVD depth coverage of NLOG wells\n"
                 "(after TVD fix in 2_pull_data.py)")
    ax.set_xlim(0, max(6000, float(np.max(allw)) if len(allw) else 6000))
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
