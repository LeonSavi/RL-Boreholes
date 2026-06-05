"""
Inclination diagnostic for the NLOG well corpus.

Parses every `dirsurvey.json` under data/nlog/nlog_scrape/, extracts the
maximum `devAngle` (degrees from vertical) along each well's path, and
plots the distribution.

The encoder treats borehole depth as if every well were vertical, which
is true for Lily (IODP) cores but only approximately true for NLOG. This
script quantifies how approximate.

Outputs:
    plots/nlog_inclination.png
    plots/analysis/nlog_inclination_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NLOG_DIR = Path("data/nlog/nlog_scrape")
OUT_PNG  = Path("plots/nlog_inclination.png")
OUT_CSV  = Path("plots/analysis/nlog_inclination_summary.csv")


def well_inclination(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(obj, dict):
        return None
    surveys = obj.get("dirSurveys") or []
    angles = []
    md_max = 0.0
    tvd_max = 0.0
    for s in surveys:
        for p in s.get("dirSurveyPoints") or []:
            a = p.get("devAngle")
            if a is None:
                continue
            angles.append(float(a))
            md_max = max(md_max, float(p.get("ahDepth") or 0))
            tvd_max = max(tvd_max, float(p.get("tvDepth") or 0))
    if not angles:
        return None
    return {
        "borehole": obj.get("boreholeName") or path.parent.name,
        "n_points": len(angles),
        "max_dev_deg": float(max(angles)),
        "mean_dev_deg": float(np.mean(angles)),
        "p95_dev_deg": float(np.percentile(angles, 95)),
        "md_max_m": md_max,
        "tvd_max_m": tvd_max,
        "md_tvd_gap_m": md_max - tvd_max,
    }


def main() -> None:
    files = list(NLOG_DIR.glob("*/dirsurvey.json"))
    print(f"found {len(files):,} dirsurvey.json files")

    rows = []
    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  parsed {i + 1:,}")
        r = well_inclination(f)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"\n{len(df):,} wells with at least one devAngle reading")

    bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 90)]
    print("\nmax devAngle bands (% of wells):")
    for lo, hi in bands:
        m = (df["max_dev_deg"] >= lo) & (df["max_dev_deg"] < hi)
        print(f"  [{lo:>3d}, {hi:>3d})  n={m.sum():>5d}  ({m.mean()*100:5.1f}%)")

    p95 = df["max_dev_deg"].quantile(0.95)
    p99 = df["max_dev_deg"].quantile(0.99)
    median = df["max_dev_deg"].median()
    print(f"\nmax devAngle  median={median:.2f}°  P95={p95:.2f}°  P99={p99:.2f}°")

    df_gap = df[df["md_max_m"] > 100]
    rel_gap = df_gap["md_tvd_gap_m"] / df_gap["md_max_m"].clip(lower=1)
    print(f"\nMD-TVD gap (relative, wells >100 m deep):")
    print(f"  median {rel_gap.median()*100:.2f}%   P95 {rel_gap.quantile(0.95)*100:.2f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.concatenate([np.arange(0, 30, 1), [30, 45, 60, 90]])
    ax1.hist(df["max_dev_deg"].clip(upper=90), bins=bins,
             color="#4C72B0", edgecolor="white", linewidth=0.4)
    for ref, c, label in [(5, "#888", "5°"), (10, "#d6604d", "10°"),
                          (30, "#222", "30°")]:
        ax1.axvline(ref, color=c, linestyle="--", linewidth=1, alpha=0.7)
        ax1.text(ref + 0.5, ax1.get_ylim()[1] * 0.9, label,
                 color=c, fontsize=9)
    ax1.set_xlabel("max deviation angle along well [°]")
    ax1.set_ylabel("number of wells")
    ax1.set_title(f"NLOG well inclination ({len(df):,} wells)")
    ax1.set_xlim(0, 90)

    cumulative = np.sort(df["max_dev_deg"].values)
    yvals = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
    ax2.plot(cumulative, yvals, color="#4C72B0", linewidth=1.6)
    for ref, c, label in [(5, "#888", "5°"), (10, "#d6604d", "10°"),
                          (30, "#222", "30°")]:
        ax2.axvline(ref, color=c, linestyle="--", linewidth=1, alpha=0.7)
        frac = (df["max_dev_deg"] < ref).mean() * 100
        ax2.text(ref + 0.5, frac - 4, f"{label}: {frac:.0f}%",
                 color=c, fontsize=9)
    ax2.set_xlabel("max deviation angle [°]")
    ax2.set_ylabel("% wells with max devAngle below x")
    ax2.set_title("Cumulative distribution")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "How vertical are NLOG wells?  "
        "(encoder currently treats them all as fully vertical)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
