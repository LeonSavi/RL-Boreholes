"""
Lateral-drift diagnostic for the NLOG well corpus.

Companion to nlog_inclination.py. For each well's deviation survey,
computes the maximum horizontal offset along the path:

    drift(p) = sqrt(dx_p^2 + dy_p^2)

where (dx_p, dy_p) is the cumulative horizontal displacement of survey
point p from the surface spud point (provided directly by NLOG in
metres).

Why this matters: inclination tells you the path's angle; lateral drift
tells you the actual distance between the bit and the surface (x, y).
The simulator generates vertical columns at one (x, y) per well, so a
well that drifts 1 km laterally is sampling rock that lives in a
different simulator cell than the surface coordinate suggests.

Outputs:
    plots/nlog_lateral_drift.png
    plots/analysis/nlog_lateral_drift_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NLOG_DIR = Path("data/nlog/nlog_scrape")
OUT_PNG  = Path("plots/nlog_lateral_drift.png")
OUT_CSV  = Path("plots/analysis/nlog_lateral_drift_summary.csv")


def well_drift(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(obj, dict):
        return None
    surveys = obj.get("dirSurveys") or []
    best_pts: list = []
    for s in surveys:
        pts = s.get("dirSurveyPoints") or []
        if len(pts) > len(best_pts):
            best_pts = pts
    if not best_pts:
        return None
    drifts, mds = [], []
    for p in best_pts:
        dx = p.get("dx")
        dy = p.get("dy")
        md = p.get("ahDepth")
        if dx is None or dy is None or md is None:
            continue
        drifts.append(float(np.hypot(dx, dy)))
        mds.append(float(md))
    if not drifts:
        return None
    drifts = np.asarray(drifts)
    mds = np.asarray(mds)
    return {
        "borehole":      obj.get("boreholeName") or path.parent.name,
        "n_points":      len(drifts),
        "md_max_m":      float(mds.max()),
        "drift_at_td_m": float(drifts[-1]),
        "max_drift_m":   float(drifts.max()),
    }


def main() -> None:
    files = list(NLOG_DIR.glob("*/dirsurvey.json"))
    print(f"found {len(files):,} dirsurvey.json files")

    rows = []
    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i + 1:,}")
        r = well_drift(f)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"\n{len(df):,} wells with usable (dx, dy) drift data")

    bands = [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 5000)]
    print("\nmax horizontal drift bands (% of wells):")
    for lo, hi in bands:
        m = (df["max_drift_m"] >= lo) & (df["max_drift_m"] < hi)
        print(f"  [{lo:>4d}, {hi:>4d})  n={m.sum():>5d}  ({m.mean()*100:5.1f}%)")

    median = df["max_drift_m"].median()
    p95 = df["max_drift_m"].quantile(0.95)
    p99 = df["max_drift_m"].quantile(0.99)
    print(f"\nmax drift  median={median:.0f} m  P95={p95:.0f} m  P99={p99:.0f} m")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # log-spaced bins so we can see the long tail
    bins = np.concatenate([[0], np.logspace(0, 4, 41)])
    ax1.hist(df["max_drift_m"].clip(upper=5000),
             bins=bins, color="#4C72B0",
             edgecolor="white", linewidth=0.4)
    for ref, c, label in [(100, "#888", "100 m"),
                          (500, "#d6604d", "500 m"),
                          (1000, "#222", "1 km")]:
        ax1.axvline(ref, color=c, linestyle="--", linewidth=1, alpha=0.7)
        ax1.text(ref * 1.05, ax1.get_ylim()[1] * 0.92, label,
                 color=c, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("max horizontal drift (bit vs spud) [m]  --- log scale")
    ax1.set_ylabel("number of wells")
    ax1.set_title(f"NLOG lateral drift ({len(df):,} wells)")
    ax1.set_xlim(1, 5000)

    cumulative = np.sort(df["max_drift_m"].values)
    yvals = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
    ax2.plot(cumulative, yvals, color="#4C72B0", linewidth=1.6)
    for ref, c, label in [(100, "#888", "100 m"),
                          (500, "#d6604d", "500 m"),
                          (1000, "#222", "1 km")]:
        ax2.axvline(ref, color=c, linestyle="--", linewidth=1, alpha=0.7)
        frac = (df["max_drift_m"] < ref).mean() * 100
        ax2.text(ref * 1.05, frac - 4, f"{label}: {frac:.0f}%",
                 color=c, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("max horizontal drift [m]  --- log scale")
    ax2.set_ylabel("% wells with drift below x")
    ax2.set_title("Cumulative distribution")
    ax2.set_xlim(1, 5000)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(
        "How far do NLOG wells drift from their surface spud point?  "
        "(simulator currently samples a single vertical column per well)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
