"""
Per-rock empirical gas-response shift table.

Water saturation (sw) is measured in ~5,030 of 8.5M parquet rows --
not enough to fit a per-(rock, formation, depth) bank from it, but
enough to compute aggregate shifts:

    shift_v(rock) = mean(v | sw < 0.5, rock) - mean(v | sw >= 0.5, rock)

for v in {rhob, nphi, res_deep_log}. These shifts get applied
parametrically to ore-body cells in map_generator after the normal
bank draw.

Outputs:
    data/clean/gas_shift_table.json
    plots/gas_shift_table.png
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

PARQUET = Path("data/clean/samples.parquet")
OUT_JSON = Path("data/clean/gas_shift_table.json")
OUT_PNG  = Path("plots/gas_shift_table.png")

VARIABLES = ["rhob", "nphi", "res_deep_log"]
GAS_THRESHOLD = 0.5   # sw < 0.5 -> gas-bearing
MIN_PER_BUCKET = 30


def main() -> None:
    print("loading samples.parquet ...")
    df = pd.read_parquet(PARQUET)
    # we need sw + any of the gas-sensitive variables, per (well, depth)
    keep_measurements = ["sw"] + VARIABLES
    df = df[df["measurement"].isin(keep_measurements)]
    wide = df.pivot_table(
        index=["dataset", "borehole", "depth", "rock_type_fine"],
        columns="measurement", values="value", aggfunc="mean",
    ).reset_index()
    wide = wide.dropna(subset=["sw"])
    print(f"{len(wide):,} (well, depth) rows have an sw measurement")

    shifts = {}
    rows = []
    for rock in sorted(wide["rock_type_fine"].dropna().unique()):
        sub = wide[wide["rock_type_fine"] == rock]
        gas = sub[sub["sw"] < GAS_THRESHOLD]
        wet = sub[sub["sw"] >= GAS_THRESHOLD]
        if len(gas) < MIN_PER_BUCKET or len(wet) < MIN_PER_BUCKET:
            print(f"  skip {rock:18s}  gas={len(gas)}, wet={len(wet)}  (sparse)")
            continue
        per_var = {}
        for v in VARIABLES:
            g = gas[v].dropna().values
            w = wet[v].dropna().values
            if len(g) < MIN_PER_BUCKET or len(w) < MIN_PER_BUCKET:
                per_var[v] = 0.0
                continue
            per_var[v] = float(np.mean(g) - np.mean(w))
        shifts[rock] = per_var
        print(f"  {rock:18s}  gas={len(gas):>4d} wet={len(wet):>4d}   "
              f"drhob={per_var['rhob']:+.3f}  "
              f"dnphi={per_var['nphi']:+.3f}  "
              f"dres={per_var['res_deep_log']:+.2f}")
        rows.append({"rock": rock, "n_gas": len(gas), "n_wet": len(wet),
                     **{f"d_{v}": per_var[v] for v in VARIABLES}})

    # global default for unseen rocks
    if shifts:
        default = {}
        for v in VARIABLES:
            vals = [s[v] for s in shifts.values()]
            default[v] = float(np.mean(vals))
        shifts["default"] = default
        print(f"\ndefault (mean across rocks):  "
              f"drhob={default['rhob']:+.3f}  "
              f"dnphi={default['nphi']:+.3f}  "
              f"dres={default['res_deep_log']:+.2f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(shifts, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # plot ----------------------------------------------------------------
    tbl = pd.DataFrame(rows)
    if len(tbl):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, v in zip(axes, VARIABLES):
            col = f"d_{v}"
            tbl_sorted = tbl.sort_values(col)
            colors = ["#d6604d" if x < 0 else "#4C72B0"
                      for x in tbl_sorted[col]]
            ax.barh(tbl_sorted["rock"], tbl_sorted[col], color=colors,
                    edgecolor="black", linewidth=0.4)
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(f"d {v}\n(gas - wet)", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Empirical gas-response shifts per rock "
                     f"(sw<{GAS_THRESHOLD} vs sw>={GAS_THRESHOLD})",
                     fontsize=18, y=1.02)
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
        print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
