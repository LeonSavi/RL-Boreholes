"""
Visual sanity check: overlay bank samples on real samples at a panel
of (rock, depth) cells. Companion to bank_fit_per_width.py -- this is
the eyeball test.

For each probe, draws 5000 synthetic samples from the 10 m bank,
pulls the real samples in the same 10 m window from samples.parquet,
and plots them as overlapping histograms per variable.

Output:
    plots/bank_overlay_panel.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from simulator.distributions import DistributionBank

BANK_PATH    = Path("data/clean/distributions.pkl")
PARQUET_PATH = Path("data/clean/samples.parquet")
OUT_PNG      = Path("plots/bank_overlay_panel.png")

VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
N_DRAWS   = 5000
TRUTH_W   = 10.0

PROBES = [
    ("chalk",            1500),
    ("claystone_hot",    2200),
    ("halite_pure",      2700),
    ("sandstone_clean",  3000),
    ("sandstone_shaly",  2200),
    ("dolomite",         2200),
]


def real_window(df, rock, lo, hi):
    sub = df[(df["rock_type_fine"] == rock)
             & (df["depth"] >= lo) & (df["depth"] < hi)]
    if not len(sub):
        return pd.DataFrame()
    return sub.pivot_table(
        index=["dataset", "borehole", "depth"],
        columns="measurement", values="value", aggfunc="mean",
    ).reset_index()


def main() -> None:
    bank = DistributionBank.load(BANK_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["measurement"].isin(VARIABLES)]

    rng = np.random.default_rng(0)
    n_rows = len(PROBES); n_cols = len(VARIABLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.4 * n_rows))

    for i, (rock, depth) in enumerate(PROBES):
        real = real_window(df, rock,
                           depth - TRUTH_W / 2, depth + TRUTH_W / 2)
        synth = bank.sample(rock, float(depth), n=N_DRAWS, rng=rng,
                            interpolate=True)
        for j, v in enumerate(VARIABLES):
            ax = axes[i, j]
            r = real.get(v, pd.Series(dtype=float)).dropna().values
            s = np.asarray(synth.get(v, np.array([])))
            s = s[np.isfinite(s)]
            if len(r) >= 30 and len(s) >= 30:
                lo = float(min(np.percentile(r, 1), np.percentile(s, 1)))
                hi = float(max(np.percentile(r, 99), np.percentile(s, 99)))
                bins = np.linspace(lo, hi, 40)
                ax.hist(r, bins=bins, alpha=0.55, color="#4C72B0",
                        label="real", density=True)
                ax.hist(s, bins=bins, alpha=0.55, color="#d6604d",
                        label="synth", density=True)
                ks = float(ks_2samp(r, s).statistic)
                ax.set_title(f"{v}  KS={ks:.2f}", fontsize=8)
            elif len(r) >= 30:
                ax.hist(r, bins=40, alpha=0.55, color="#4C72B0",
                        label="real", density=True)
                ax.set_title(f"{v} (no synth)", fontsize=8)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f"{rock}\n@ {int(depth)} m", fontsize=8)
            if i == 0 and j == n_cols - 1:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Bank-vs-real per-cell overlays (10 m bin, "
                 f"{N_DRAWS} synth samples vs real)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
