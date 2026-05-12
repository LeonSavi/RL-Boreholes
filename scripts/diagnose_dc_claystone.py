"""Diagnose why DC claystone run lengths in the simulator are ~5× too long
vs the validation report's real-NLOG baseline.

The validation pipeline resamples each well to a 10 m grid before fitting
the transition matrix.  If the raw NLOG sequence has thin alternating
beds (≪ 10 m), the resampling collapses them into whichever rock
dominates each 10 m cell — inflating the fitted run lengths.

This script overlays three run-length distributions for DC claystone:
  1. Raw NLOG (native ~1 m sample step) — ground truth
  2. 10 m-resampled NLOG (what the fitter actually sees) — what shapes
     the matrix that drives the simulator
  3. The mean run length implied by the fitted matrix's diagonal
     (1 / (1 − P[i, i])) — what the simulator produces in steady state

Saves plots/validation/dc_claystone_resampling_diagnostic.png and prints
the headline numbers.

Run:
    python scripts/diagnose_dc_claystone.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.formation_geometry import (
    FormationGeometry,
    TRANSITION_BIN_STEP_M,
)


OUT_PATH = Path("plots/validation/dc_claystone_resampling_diagnostic.png")
TARGET_FORMATION = "DC"
TARGET_ROCK = "claystone"


def _runs(rocks: list[str], target: str) -> list[int]:
    runs: list[int] = []
    n = 0
    for r in rocks:
        if r == target:
            n += 1
        elif n > 0:
            runs.append(n)
            n = 0
    if n > 0:
        runs.append(n)
    return runs


def _native_step_summary(df: pd.DataFrame) -> tuple[float, float]:
    """Return (median, p95) of intra-well consecutive depth differences."""
    diffs = (df.sort_values(["borehole", "depth"])
               .groupby("borehole")["depth"].diff().dropna().to_numpy())
    diffs = diffs[(diffs > 0) & (diffs < 100)]
    return float(np.median(diffs)), float(np.percentile(diffs, 95))


def main() -> None:
    print(f"loading samples.parquet for formation={TARGET_FORMATION!r}, "
          f"rock={TARGET_ROCK!r} ...")
    df = pd.read_parquet("data/clean/samples.parquet")
    df = df[(df["dataset"] == "NLOG")
            & (df["formation"] == TARGET_FORMATION)]
    df = df.drop_duplicates(["dataset", "borehole", "depth"], keep="first")
    df["rock_type_fine"] = df["rock_type_fine"].astype(str)

    n_wells = df["borehole"].nunique()
    median_step, p95_step = _native_step_summary(df)
    print(f"  {len(df):,} rows, {n_wells} wells")
    print(f"  native depth step: median={median_step:.2f} m, "
          f"p95={p95_step:.2f} m")

    # 1. raw run lengths (native step)
    raw_lengths_cells: list[int] = []
    raw_lengths_m: list[float] = []
    for bh, well_df in df.groupby("borehole"):
        well_df = well_df.sort_values("depth")
        rocks = well_df["rock_type_fine"].tolist()
        depths = well_df["depth"].to_numpy()
        # walk runs, record both length-in-rows and length-in-metres
        if not rocks:
            continue
        cur = rocks[0]; i0 = 0
        for k in range(1, len(rocks)):
            if rocks[k] != cur:
                if cur == TARGET_ROCK:
                    raw_lengths_cells.append(k - i0)
                    raw_lengths_m.append(float(depths[k - 1] - depths[i0]))
                cur = rocks[k]; i0 = k
        if cur == TARGET_ROCK:
            raw_lengths_cells.append(len(rocks) - i0)
            raw_lengths_m.append(float(depths[-1] - depths[i0]))

    # 2. 10m-resampled run lengths (this is what the fitter sees)
    resampled_lengths_cells: list[int] = []
    work = df[["borehole", "depth", "rock_type_fine"]].copy()
    work = work.sort_values(["borehole", "depth"], kind="mergesort")
    work["_bin"] = np.floor(
        work["depth"].to_numpy() / TRANSITION_BIN_STEP_M
    ).astype(np.int64)
    work = work.drop_duplicates(["borehole", "_bin"], keep="first")
    for bh, well_df in work.groupby("borehole"):
        rocks = well_df["rock_type_fine"].tolist()
        resampled_lengths_cells.extend(_runs(rocks, TARGET_ROCK))

    # 3. fitted matrix → theoretical mean run length
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    stats = geom.formations[TARGET_FORMATION]
    P = stats.transition_matrix
    p_self = float(P.loc[TARGET_ROCK, TARGET_ROCK]) if TARGET_ROCK in P.index else float("nan")
    theory_mean_cells = 1.0 / max(1.0 - p_self, 1e-9)

    print()
    print(f"  raw runs  : n={len(raw_lengths_cells):>4d}  "
          f"mean_cells={np.mean(raw_lengths_cells):>5.1f}  "
          f"mean_m={np.mean(raw_lengths_m):>6.1f}  "
          f"median_cells={np.median(raw_lengths_cells):>4.0f}  "
          f"p95_m={np.percentile(raw_lengths_m, 95):>6.1f}")
    print(f"  10m runs  : n={len(resampled_lengths_cells):>4d}  "
          f"mean_cells={np.mean(resampled_lengths_cells):>5.1f}  "
          f"(= mean_m {np.mean(resampled_lengths_cells)*TRANSITION_BIN_STEP_M:>6.1f})  "
          f"median_cells={np.median(resampled_lengths_cells):>4.0f}")
    print(f"  matrix    : P({TARGET_ROCK}|{TARGET_ROCK})={p_self:.4f}  "
          f"theory mean run = 1/(1-p) = {theory_mean_cells:.1f} cells "
          f"(= {theory_mean_cells * TRANSITION_BIN_STEP_M:.0f} m)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: distributions in cells of NATIVE step (incommensurate, but
    # plot in metres to be apples-to-apples)
    bins_m = np.linspace(0, max(np.max(raw_lengths_m),
                                 max(resampled_lengths_cells, default=0)
                                 * TRANSITION_BIN_STEP_M) * 1.05, 50)
    axes[0].hist(raw_lengths_m, bins=bins_m, alpha=0.55,
                 label=f"raw native step ({median_step:.1f} m)  "
                       f"n={len(raw_lengths_m)}", color="tab:blue")
    axes[0].hist(np.array(resampled_lengths_cells) * TRANSITION_BIN_STEP_M,
                 bins=bins_m, alpha=0.55,
                 label=f"resampled to {TRANSITION_BIN_STEP_M:.0f} m  "
                       f"n={len(resampled_lengths_cells)}", color="tab:orange")
    axes[0].axvline(theory_mean_cells * TRANSITION_BIN_STEP_M,
                    color="tab:red", linestyle="--",
                    label=f"matrix theory mean = "
                          f"{theory_mean_cells * TRANSITION_BIN_STEP_M:.0f} m")
    axes[0].set_xlabel("run length (m)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"{TARGET_FORMATION} {TARGET_ROCK}: raw vs resampled run lengths")
    axes[0].legend(fontsize=8)

    # right: log-scaled to show tail behaviour
    axes[1].hist(raw_lengths_m, bins=bins_m, alpha=0.55,
                 label="raw", color="tab:blue", density=True)
    axes[1].hist(np.array(resampled_lengths_cells) * TRANSITION_BIN_STEP_M,
                 bins=bins_m, alpha=0.55,
                 label="resampled 10m", color="tab:orange", density=True)
    axes[1].axvline(theory_mean_cells * TRANSITION_BIN_STEP_M,
                    color="tab:red", linestyle="--", label="matrix theory")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("run length (m)")
    axes[1].set_ylabel("density (log)")
    axes[1].set_title("same, log-scaled")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"DC {TARGET_ROCK} run-length resampling diagnostic — "
        f"does 10m resampling explain the simulator's overlong runs?"
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
