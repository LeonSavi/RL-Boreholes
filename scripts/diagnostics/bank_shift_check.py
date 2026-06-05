"""
H1 check: how much do the per-cell distributions move between the old
(MD-aligned, 100 m bank) and the new (TVD-aligned, 10 m bank)?

For a small panel of representative (rock, depth) cells, draws N samples
from each bank version and compares mean / P50 / P90 per variable. Any
shift > 0.5 sigma in any variable counts as material -- if so, the 10k
synthetic maps should be regenerated and the encoder retrained.

Outputs:
    plots/analysis/bank_shift_check.csv
    plots/bank_shift_check.png  (delta-vs-cell heatmap)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulator.distributions import DistributionBank

OLD = Path("data/clean/distributions.pkl.bak_400m")
NEW = Path("data/clean/distributions.pkl")
OUT_CSV = Path("plots/analysis/bank_shift_check.csv")
OUT_PNG = Path("plots/bank_shift_check.png")

N_DRAWS = 2000
VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]

# (rock, depth_m) cells -- pick rocks with dense coverage at typical
# Dutch reservoir depths.
PROBE_CELLS = [
    ("claystone_hot",    2200.0),
    ("claystone_cool",   1800.0),
    ("sandstone_clean",  2600.0),
    ("sandstone_shaly",  2200.0),
    ("halite_pure",      2600.0),
    ("dolomite",         2200.0),
    ("chalk",            1500.0),
    ("anhydrite",        2600.0),
]


def main() -> None:
    old_bank = DistributionBank.load(OLD)
    new_bank = DistributionBank.load(NEW)
    rng_old = np.random.default_rng(0)
    rng_new = np.random.default_rng(1)

    rows = []
    for rock, depth in PROBE_CELLS:
        s_old = old_bank.sample(rock, depth, n=N_DRAWS, rng=rng_old)
        s_new = new_bank.sample(rock, depth, n=N_DRAWS, rng=rng_new)
        for v in VARIABLES:
            o = np.asarray(s_old.get(v, np.full(N_DRAWS, np.nan)))
            n = np.asarray(s_new.get(v, np.full(N_DRAWS, np.nan)))
            o = o[np.isfinite(o)]
            n = n[np.isfinite(n)]
            if len(o) < 50 or len(n) < 50:
                rows.append({"rock": rock, "depth": depth, "var": v,
                             "delta_mean_sigma": np.nan,
                             "delta_p50_sigma": np.nan,
                             "delta_p90_sigma": np.nan})
                continue
            sigma = float(o.std() + 1e-9)
            rows.append({
                "rock": rock,
                "depth": depth,
                "var": v,
                "old_mean": float(o.mean()),
                "new_mean": float(n.mean()),
                "delta_mean_sigma": float((n.mean() - o.mean()) / sigma),
                "delta_p50_sigma":  float((np.median(n) - np.median(o)) / sigma),
                "delta_p90_sigma":  float((np.quantile(n, 0.9)
                                           - np.quantile(o, 0.9)) / sigma),
            })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))

    big = df["delta_mean_sigma"].abs().max(skipna=True)
    print(f"\nmax |delta_mean| across all (rock, var) = {big:.3f} sigma")
    print("MATERIAL shift" if big >= 0.5 else "shift WITHIN noise (<0.5 sigma)")

    # heatmap
    pivot = df.pivot_table(index=["rock", "depth"], columns="var",
                           values="delta_mean_sigma")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([f"{r} @ {int(d)}m" for r, d in pivot.index],
                       fontsize=8)
    fig.colorbar(im, ax=ax, label="delta mean [sigma_old]")
    ax.set_title("Per-cell mean shift, new (TVD, 10 m) vs old (MD, 100 m)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
