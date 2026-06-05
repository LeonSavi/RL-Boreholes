"""Recompute the per-cell Spearman correlation table shown on the
"Does the bank carry the right rock physics?" slide.

Reads `data/clean/samples.parquet`, picks the seven (rock, depth_bin)
cells used in the slide's table (same cells as `bank_overlay_panel`),
and computes Spearman rho across the 5 wireline channels.

Output:
  plots/analysis/rock_physics_corr.csv
  printed table in slide order
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PARQUET = Path("data/clean/samples.parquet")
OUT_CSV = Path("plots/analysis/rock_physics_corr.csv")
DEPTH_BIN_WIDTH_M = 10.0

# (rock_type_fine, representative depth in metres) — matches the slide
CELLS = [
    ("claystone_hot",   2200),
    ("claystone_cool",  1800),
    ("sandstone_clean", 2600),
    ("sandstone_shaly", 2200),
    ("halite_pure",     2600),
    ("dolomite",        2200),
    ("chalk",           1800),
    ("anhydrite",       2600),
]

VARS = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
# pairs reported in the slide
PAIRS = [
    ("rhob", "dt_us_ft", r"$\rho \times \Delta t$"),
    ("rhob", "nphi",     r"$\rho \times \phi_n$"),
    ("rhob", "gr_api",   r"$\rho \times \mathrm{GR}$"),
    ("nphi", "dt_us_ft", r"$\phi_n \times \Delta t$"),
]


def cell_dataframe(samples: pd.DataFrame, rock: str, depth_m: float) -> pd.DataFrame:
    """Return a (wireline-wide) frame for one (rock, 10 m bin) cell."""
    lo = depth_m - DEPTH_BIN_WIDTH_M / 2.0
    hi = depth_m + DEPTH_BIN_WIDTH_M / 2.0
    cell = samples[
        (samples["dataset"] == "NLOG")
        & (samples["rock_type_fine"] == rock)
        & (samples["depth"] >= lo)
        & (samples["depth"] < hi)
    ]
    wide = (
        cell[cell["measurement"].isin(VARS)]
            .pivot_table(index=["borehole", "depth"],
                          columns="measurement",
                          values="value",
                          aggfunc="mean")
            .reset_index()
    )
    for v in VARS:
        if v not in wide.columns:
            wide[v] = np.nan
    return wide


def main() -> None:
    samples = pd.read_parquet(PARQUET)
    rows = []

    header_pairs = "  ".join(f"{a}x{b}" for a, b, _ in PAIRS)
    print(f"\n{'rock':<17} {'depth':>6}  {'n':>6}   {header_pairs}")
    print("-" * 70)

    for rock, depth_m in CELLS:
        df = cell_dataframe(samples, rock, depth_m)
        n_rows = len(df)
        corrs = {}
        for a, b, _ in PAIRS:
            xs = df[[a, b]].dropna()
            if len(xs) < 30:
                corrs[(a, b)] = float("nan")
            else:
                rho, _ = spearmanr(xs[a], xs[b])
                corrs[(a, b)] = float(rho)
        row = {
            "rock": rock,
            "depth_m": depth_m,
            "n": n_rows,
        }
        for a, b, _ in PAIRS:
            row[f"{a}_x_{b}"] = round(corrs[(a, b)], 3)
        rows.append(row)

        pretty = "  ".join(f"{corrs[(a, b)]:+.3f}" for a, b, _ in PAIRS)
        print(f"{rock:<17} {depth_m:>6}  {n_rows:>6}   {pretty}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
