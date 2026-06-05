"""
"Smallest consistent bin width" empirical check.

Question we want to answer: at what bin width does the distribution
bank start producing samples that look meaningfully DIFFERENT from
the real underlying data at the same (rock, depth) cell?

Method, per (rock, depth) probe and per candidate width W in
{10, 20, 25, 50, 100} m:

  1. Real samples in the WIDE window [d - W/2, d + W/2]: these are
     what a bank fit at width W would have used as KDE input.
     This isolates the "depth-smearing" question -- if W is so wide
     that compaction trends inside the bin are no longer constant,
     the wide-window real distribution will already drift away from
     the narrow-window ("truth") distribution.

  2. Synthetic samples from the actual fitted bank at the same
     (rock, depth). This isolates "KDE fit quality" -- once we are
     happy that W's wide window is consistent, does the KDE on top
     also stay consistent?

Two metrics per (variable, width):
  - KS distance between real-wide(W) and real-narrow(10 m). Tests
    whether widening the bin changes WHAT data is being fit.
  - KS distance between bank-samples and real-narrow(10 m). Tests
    whether the KDE + copula pipeline preserves that distribution.

The smallest "consistent" bin width is the smallest W where both
KS distances stay below a reasonable threshold (~0.10) for the
densely-populated cells we probe.

Outputs:
    plots/bank_consistency_summary.png
    plots/analysis/bank_consistency_summary.csv
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
OUT_PNG      = Path("plots/bank_consistency_summary.png")
OUT_CSV      = Path("plots/analysis/bank_consistency_summary.csv")

VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
WIDTHS_M  = [10, 20, 25, 50, 100]
TRUTH_W   = 10.0    # narrow window we treat as "ground truth"
N_DRAWS   = 5000

# Probes: rock + centre depth. Picked from densely-populated cells so
# the narrow window has enough samples to be a meaningful reference.
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


def real_window(df: pd.DataFrame, rock: str, lo: float, hi: float
                ) -> pd.DataFrame:
    sub = df[(df["rock_type_fine"] == rock)
             & (df["depth"] >= lo)
             & (df["depth"] < hi)]
    if not len(sub):
        return pd.DataFrame()
    wide = sub.pivot_table(
        index=["dataset", "borehole", "depth"],
        columns="measurement",
        values="value",
        aggfunc="mean",
    ).reset_index()
    return wide


def ks(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30:
        return float("nan")
    s, _ = ks_2samp(a, b)
    return float(s)


def main() -> None:
    bank = DistributionBank.load(BANK_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["measurement"].isin(VARIABLES)]

    print(f"bank: {len(bank.cells)} populated cells, "
          f"{len(bank.depth_bins) - 1} depth bins of "
          f"{bank.depth_bins[1] - bank.depth_bins[0]:.0f} m\n")

    rng = np.random.default_rng(0)
    rows = []

    for rock, depth in PROBE_CELLS:
        truth = real_window(df, rock,
                            depth - TRUTH_W / 2, depth + TRUTH_W / 2)
        if len(truth) < 30:
            print(f"  skip {rock} @ {depth} m (only {len(truth)} truth samples)")
            continue

        synth = bank.sample(rock, float(depth), n=N_DRAWS,
                            rng=rng, interpolate=True)

        for v in VARIABLES:
            tv = truth.get(v, pd.Series(dtype=float)).dropna().values
            sv = np.asarray(synth.get(v, np.array([])))
            ks_bank = ks(tv, sv)

            for W in WIDTHS_M:
                wide_window = real_window(df, rock,
                                          depth - W / 2, depth + W / 2)
                wv = wide_window.get(v, pd.Series(dtype=float)).dropna().values
                ks_data = ks(tv, wv) if W > TRUTH_W else 0.0
                rows.append({
                    "rock": rock, "depth": depth, "var": v, "W": W,
                    "n_truth": len(tv), "n_wide": len(wv),
                    "ks_widen": ks_data,
                    "ks_bank":  ks_bank,   # constant across W (10m bank)
                })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}\n")

    # Aggregate: median KS-widen per width, across all (rock, var) probes
    # with enough data. Bank KS reported separately (it's the same number
    # at every W since the bank IS the 10 m bank).
    agg = (out.dropna(subset=["ks_widen"])
              .groupby("W")["ks_widen"]
              .agg(["median", "mean", "max", "count"])
              .reset_index())
    print("aggregated KS(narrow truth, wide-real) per bin width:")
    print(agg.to_string(index=False))

    bank_med = (out["ks_bank"].dropna().median())
    bank_mean = (out["ks_bank"].dropna().mean())
    bank_max = (out["ks_bank"].dropna().max())
    print(f"\n10 m bank KS(narrow truth, bank-synth):  "
          f"median={bank_med:.3f}  mean={bank_mean:.3f}  max={bank_max:.3f}")

    # Plot: KS-widen as a function of bin width, one line per variable
    # (median across rocks).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = plt.cm.tab10.colors
    for i, v in enumerate(VARIABLES):
        sub = (out[out["var"] == v]
               .dropna(subset=["ks_widen"])
               .groupby("W")["ks_widen"].median())
        ax.plot(sub.index, sub.values, marker="o", linewidth=1.6,
                color=palette[i], label=v)
    ax.axhline(0.10, color="#888", linestyle=":", linewidth=1,
               label="KS = 0.10 (rough \"consistent\" line)")
    ax.set_xlabel("bin width [m]")
    ax.set_ylabel("KS distance vs 10 m ground truth")
    ax.set_title("Depth-smearing as a function of bin width\n"
                 "(median across {n} probe cells, per variable)".format(
                     n=len(PROBE_CELLS)))
    ax.set_xscale("log")
    ax.set_xticks(WIDTHS_M); ax.set_xticklabels([str(w) for w in WIDTHS_M])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(0.2, out["ks_widen"].quantile(0.9) * 1.1))

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
