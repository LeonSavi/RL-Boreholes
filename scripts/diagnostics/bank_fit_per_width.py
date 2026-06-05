"""
Compare bank-vs-real KS at multiple bin widths.

Refits a small DistributionBank at each candidate width on the SAME
parquet, then for each probe (rock, depth) draws synthetic samples and
measures KS vs the narrow (10 m) real-data ground truth.

If KS gets worse at narrower widths, the KDE is being starved of
samples (noise dominates). If KS gets worse at wider widths, depth
smearing is the bigger error. The crossover is the "smallest
consistent" width.

Outputs:
    plots/bank_fit_per_width.png
    plots/analysis/bank_fit_per_width.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from simulator.distributions import DistributionBank

PARQUET_PATH = Path("data/clean/samples.parquet")
OUT_PNG      = Path("plots/bank_fit_per_width.png")
OUT_CSV      = Path("plots/analysis/bank_fit_per_width.csv")

VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
WIDTHS_M  = [10, 20, 25, 50, 100]
TRUTH_W   = 10.0
N_DRAWS   = 5000

PROBE_CELLS = [
    ("chalk",            1500),
    ("chalk",            1800),
    ("claystone_hot",    2200),
    ("claystone_cool",   1800),
    ("halite_pure",      2700),
    ("sandstone_clean",  3000),
    ("sandstone_shaly",  2200),
    ("dolomite",         2200),
    ("anhydrite",        2700),
]


def real_window(df: pd.DataFrame, rock: str, lo: float, hi: float) -> pd.DataFrame:
    sub = df[(df["rock_type_fine"] == rock)
             & (df["depth"] >= lo)
             & (df["depth"] < hi)]
    if not len(sub):
        return pd.DataFrame()
    return sub.pivot_table(
        index=["dataset", "borehole", "depth"],
        columns="measurement", values="value", aggfunc="mean",
    ).reset_index()


def ks(a, b):
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []

    for W in WIDTHS_M:
        edges = list(range(0, 6001, W))
        print(f"\nfitting bank at {W} m bins ({len(edges) - 1} bins)...")
        bank = DistributionBank.fit(
            PARQUET_PATH, variables=VARIABLES, depth_bins=edges,
        )
        df = pd.read_parquet(PARQUET_PATH)
        df = df[df["measurement"].isin(VARIABLES)]

        for rock, depth in PROBE_CELLS:
            truth = real_window(df, rock,
                                depth - TRUTH_W / 2, depth + TRUTH_W / 2)
            if len(truth) < 30:
                continue
            synth = bank.sample(rock, float(depth), n=N_DRAWS, rng=rng,
                                interpolate=True)
            # find the cell used (for fallback diagnostic)
            bin_idx = int(np.clip(depth // W, 0, len(edges) - 2))
            status = "direct" if (rock, bin_idx) in bank.cells else "fallback"

            for v in VARIABLES:
                tv = truth.get(v, pd.Series(dtype=float)).dropna().values
                sv = np.asarray(synth.get(v, np.array([])))
                rows.append({
                    "W": W, "rock": rock, "depth": depth,
                    "var": v, "n_truth": len(tv),
                    "ks_bank": ks(tv, sv),
                    "status": status,
                })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    agg = (out.dropna(subset=["ks_bank"])
              .groupby("W")["ks_bank"]
              .agg(["median", "mean", "max", "count"])
              .reset_index())
    print("\nbank-vs-truth KS, aggregated across probes and variables:")
    print(agg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = plt.cm.tab10.colors
    for i, v in enumerate(VARIABLES):
        sub = (out[out["var"] == v]
               .dropna(subset=["ks_bank"])
               .groupby("W")["ks_bank"].median())
        ax.plot(sub.index, sub.values, marker="o", linewidth=1.6,
                color=palette[i], label=v)
    overall = (out.dropna(subset=["ks_bank"])
                  .groupby("W")["ks_bank"].median())
    ax.plot(overall.index, overall.values, marker="s", linewidth=2.2,
            color="black", label="overall median")
    ax.axhline(0.10, color="#888", linestyle=":", linewidth=1,
               label="KS = 0.10")
    ax.set_xlabel("bin width [m]")
    ax.set_ylabel("KS distance (bank synth vs 10 m real truth)")
    ax.set_title("How well does the fitted bank reproduce real data\n"
                 "at each candidate bin width? (median across {n} probes)".format(
                     n=len(PROBE_CELLS)))
    ax.set_xscale("log")
    ax.set_xticks(WIDTHS_M); ax.set_xticklabels([str(w) for w in WIDTHS_M])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_ylim(0, max(0.2, out["ks_bank"].quantile(0.95) * 1.1))

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
