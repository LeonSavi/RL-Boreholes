"""
Per-formation depth statistics from samples.parquet.

Prints two tables suitable for copy-paste:

  Table 1: per-formation depth distribution (top, median, bottom, P5, P95)
           plus prevalence (% of wells where the formation appears) and
           thickness statistics.

  Table 2: per-formation rock_type_fine prevalence (the empirical
           equivalent of DUTCH_COLUMN's facies dict).

Use this to derive realistic depth ranges and facies probabilities for
the rewritten data-driven stratigraphy module.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = "data/clean/samples.parquet"

# Formation order — youngest at top to oldest at bottom (the stratigraphic
# order; matches DUTCH_COLUMN). Formations not in this list won't appear
# in the table.
FORMATION_ORDER = [
    "NU", "NM", "NL", "CK", "KN", "SL", "SG", "AT",
    "RN", "RB", "ZE", "RO", "DC",
]


def _percentile(s, p):
    """Safe percentile that handles empty series."""
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, p))


def main():
    print(f"loading {DATA} ...")
    df = pd.read_parquet(DATA)

    # NLOG only — LILY has no formation labels
    df = df[df["dataset"] == "NLOG"]
    # one row per (well, depth) cell — drop multi-variable duplicates
    df = df.drop_duplicates(subset=["dataset", "borehole", "depth"],
                             keep="first")
    print(f"  {len(df):,} unique (well, depth) cells")
    print(f"  {df['borehole'].nunique():,} unique wells")

    n_wells_total = df["borehole"].nunique()

    # =========================================================================
    # Table 1: per-formation depth + thickness statistics
    # =========================================================================
    rows = []
    for fm in FORMATION_ORDER:
        sub = df[df["formation"] == fm]
        if len(sub) == 0:
            rows.append({
                "fm":        fm,
                "n_wells":   0,
                "prevalence": 0.0,
                "n_cells":   0,
                "P5_top":    np.nan,
                "median_top": np.nan,
                "P95_top":   np.nan,
                "P5_bot":    np.nan,
                "median_bot": np.nan,
                "P95_bot":   np.nan,
                "median_thick": np.nan,
                "mean_thick": np.nan,
            })
            continue

        # per-well: this formation's depth range in that well
        per_well = sub.groupby("borehole").agg(
            top=("depth", "min"),
            bot=("depth", "max"),
            n_cells=("depth", "size"),
        )
        per_well["thickness"] = per_well["bot"] - per_well["top"]

        rows.append({
            "fm":          fm,
            "n_wells":     len(per_well),
            "prevalence":  100.0 * len(per_well) / n_wells_total,
            "n_cells":     int(sub.shape[0]),
            "P5_top":      _percentile(per_well["top"], 5),
            "median_top":  _percentile(per_well["top"], 50),
            "P95_top":     _percentile(per_well["top"], 95),
            "P5_bot":      _percentile(per_well["bot"], 5),
            "median_bot":  _percentile(per_well["bot"], 50),
            "P95_bot":     _percentile(per_well["bot"], 95),
            "median_thick": _percentile(per_well["thickness"], 50),
            "mean_thick":  float(per_well["thickness"].mean()),
        })

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("TABLE 1: per-formation depth + thickness (NLOG, per-well aggregated)")
    print("=" * 100)
    print(f"{'fm':<4} {'wells':>6} {'prev%':>6} "
          f"{'P5_top':>7} {'med_top':>7} {'P95_top':>7} "
          f"{'P5_bot':>7} {'med_bot':>7} {'P95_bot':>7} "
          f"{'med_thk':>7} {'mn_thk':>7}")
    print("-" * 100)
    for _, r in summary.iterrows():
        if r["n_wells"] == 0:
            print(f"{r['fm']:<4} {r['n_wells']:>6} {r['prevalence']:>6.1f} "
                  f"{'-':>7} {'-':>7} {'-':>7} "
                  f"{'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}")
            continue
        print(f"{r['fm']:<4} {r['n_wells']:>6.0f} {r['prevalence']:>6.1f} "
              f"{r['P5_top']:>7.0f} {r['median_top']:>7.0f} {r['P95_top']:>7.0f} "
              f"{r['P5_bot']:>7.0f} {r['median_bot']:>7.0f} {r['P95_bot']:>7.0f} "
              f"{r['median_thick']:>7.0f} {r['mean_thick']:>7.0f}")

    # =========================================================================
    # Table 2: per-formation rock_type_fine prevalence
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 2: rock_type_fine prevalence per formation")
    print("(rows that fall back to the coarse formation label; "
          "share of cells in NLOG)")
    print("=" * 100)

    facies = (
        df.groupby(["formation", "rock_type_fine"], observed=True)
          .size().rename("n").reset_index()
    )
    fm_totals = facies.groupby("formation")["n"].sum().rename("total")
    facies = facies.merge(fm_totals, on="formation")
    facies["pct"] = 100.0 * facies["n"] / facies["total"]

    for fm in FORMATION_ORDER:
        sub = facies[facies["formation"] == fm].sort_values("pct", ascending=False)
        if len(sub) == 0:
            continue
        # only show rocks contributing >= 1%
        sub = sub[sub["pct"] >= 1.0]
        print(f"\n{fm}  (total {int(fm_totals.get(fm, 0)):>10,d} cells):")
        for _, r in sub.iterrows():
            print(f"    {r['rock_type_fine']:<18s} {r['pct']:>5.1f}%   "
                  f"({int(r['n']):>10,d} cells)")

    # =========================================================================
    # Table 3: depth distribution within each formation, by rock_type_fine
    # =========================================================================
    print("\n" + "=" * 100)
    print("TABLE 3: depth-conditioned facies — for each formation, where in")
    print("the column does each refined rock type live? (median depth)")
    print("=" * 100)
    print("(useful for sanity-checking that ROCL is found below ROSL etc.)")

    for fm in FORMATION_ORDER:
        sub = df[df["formation"] == fm]
        if len(sub) == 0:
            continue
        rock_depths = sub.groupby("rock_type_fine")["depth"].agg(
            n_cells="size",
            P5=lambda x: _percentile(x, 5),
            median="median",
            P95=lambda x: _percentile(x, 95),
        ).sort_values("median")
        rock_depths = rock_depths[rock_depths["n_cells"] >= 100]
        if len(rock_depths) == 0:
            continue
        print(f"\n{fm}:")
        print(f"    {'rock':<18s} {'n_cells':>10s} {'P5':>6s} {'med':>6s} {'P95':>6s}")
        for r, row in rock_depths.iterrows():
            print(f"    {r:<18s} {int(row['n_cells']):>10,d} "
                  f"{row['P5']:>6.0f} {row['median']:>6.0f} {row['P95']:>6.0f}")


if __name__ == "__main__":
    main()