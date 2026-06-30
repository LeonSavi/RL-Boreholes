"""
Per-well depth-datum offset audit.

NLOG LAS depths are measured from the Rotary Table (RT) at the drill
rig. `details.json` stores `drpHeightInMeters` (KB elevation above the
local datum) and `drpDatumCode` ("NAP" for onshore Dutch wells, "MSL"
for offshore platforms).

The simulator bins samples by RT-relative depth, so two wells at
"depth=2000 m" can sit at different sea-floor-relative depths
because their KB heights differ. This script quantifies the spread.

Onshore (NAP): offset = drpHeightInMeters (KB height above NAP ≈ MSL
ground level). Typically 5-30 m.

Offshore (MSL): offset = drpHeightInMeters (KB above sea surface).
Real distance to sea-floor = drpHeightInMeters + water_depth, but
water_depth isn't in details.json — drpHeightInMeters is a LOWER
bound for offshore wells. Typically 25-60 m.

Outputs:
    plots/depth_datum_audit.png
    plots/analysis/depth_datum_audit.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

NLOG_DIR = Path("data/nlog/nlog_scrape")
OUT_PNG  = Path("plots/depth_datum_audit.png")
OUT_CSV  = Path("plots/analysis/depth_datum_audit.csv")


def main() -> None:
    rows = []
    files = list(NLOG_DIR.glob("*/details.json"))
    print(f"scanning {len(files):,} NLOG details.json files...")
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        rows.append({
            "borehole":   f.parent.name,
            "drp_height": d.get("drpHeightInMeters"),
            "datum":      d.get("drpDatumCode"),
            "onoffshore": d.get("onOffshore"),
            "depth_ref":  d.get("depthRefPointDescription"),
        })

    df = pd.DataFrame(rows)
    print(f"\nfound {len(df):,} wells with details.json")
    print(f"  drp_height present: {df['drp_height'].notna().sum():,}")
    print(f"  drp_height MISSING: {df['drp_height'].isna().sum():,}")

    # only those with usable values
    valid = df.dropna(subset=["drp_height"]).copy()
    valid["drp_height"] = valid["drp_height"].astype(float)
    valid["onoffshore"] = valid["onoffshore"].fillna("UNK")

    print("\nby onshore/offshore split:")
    for s, sub in valid.groupby("onoffshore"):
        h = sub["drp_height"]
        print(f"  {s:5s}  n={len(sub):>5d}  "
              f"median={h.median():>6.1f}  "
              f"P95={h.quantile(0.95):>6.1f}  "
              f"max={h.max():>6.1f}  m")

    print("\noverall offset distribution (drpHeightInMeters):")
    h_all = valid["drp_height"]
    for q in [50, 75, 90, 95, 99]:
        print(f"  P{q:>2d}  {h_all.quantile(q/100):>6.1f}  m")
    print(f"\n  count > 10 m: {(h_all > 10).sum():,} "
          f"({100*(h_all > 10).mean():.1f}%)")
    print(f"  count > 30 m: {(h_all > 30).sum():,} "
          f"({100*(h_all > 30).mean():.1f}%)")
    print(f"  count > 100 m: {(h_all > 100).sum():,} "
          f"({100*(h_all > 100).mean():.1f}%)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    on = valid[valid["onoffshore"] == "ON"]["drp_height"]
    off = valid[valid["onoffshore"] == "OFF"]["drp_height"]
    unk = valid[valid["onoffshore"] == "UNK"]["drp_height"]
    bins = np.arange(0, 110, 2)
    if len(on):
        ax.hist(on.clip(upper=100), bins=bins, alpha=0.6,
                color="#4C72B0", label=f"onshore (n={len(on)})",
                edgecolor="white", linewidth=0.3)
    if len(off):
        ax.hist(off.clip(upper=100), bins=bins, alpha=0.6,
                color="#d6604d", label=f"offshore (n={len(off)})",
                edgecolor="white", linewidth=0.3)
    if len(unk):
        ax.hist(unk.clip(upper=100), bins=bins, alpha=0.4,
                color="#888888", label=f"unknown (n={len(unk)})",
                edgecolor="white", linewidth=0.3)
    ax.axvline(10, color="#888", linestyle=":", linewidth=1, label="10 m (bin width)")
    ax.set_xlabel("drpHeightInMeters [m]")
    ax.set_ylabel("number of wells")
    ax.set_title("KB elevation above local datum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_all = np.sort(h_all.values)
    y_all = np.arange(1, len(x_all) + 1) / len(x_all) * 100
    ax.plot(x_all, y_all, color="black", linewidth=1.6, label="all")
    if len(on):
        x = np.sort(on.values); y = np.arange(1, len(x)+1)/len(x)*100
        ax.plot(x, y, color="#4C72B0", linewidth=1.3, label="onshore")
    if len(off):
        x = np.sort(off.values); y = np.arange(1, len(x)+1)/len(x)*100
        ax.plot(x, y, color="#d6604d", linewidth=1.3, label="offshore")
    for ref, c, lbl in [(10, "#888", "10 m"),
                        (30, "#d6604d", "30 m"),
                        (100, "black", "100 m")]:
        ax.axvline(ref, color=c, linestyle=":", linewidth=1, alpha=0.7)
        frac = (h_all < ref).mean() * 100
        ax.text(ref + 1, frac - 4, f"{lbl}: {frac:.0f}%", color=c, fontsize=8)
    ax.set_xlabel("drpHeightInMeters [m]")
    ax.set_ylabel("% wells with KB height below x")
    ax.set_title("Cumulative")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"NLOG depth-datum offset audit "
        f"(KB elevation = lower bound for offshore; "
        f"sim bin width = 10 m)",
        fontsize=18, y=1.02,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
