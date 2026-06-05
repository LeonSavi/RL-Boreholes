"""
Audit which DistributionBank cells were fitted, partially fitted, or
not fitted at all.

The bank admits a (rock, formation, depth_bin) cell when it has
$\\geq 30$ rows. Inside an admitted cell, each of the 5 variables
gets its own KDE if $\\geq 20$ rows survive bounds + winsorising;
otherwise that variable's KDE is silently skipped. So a cell can
be admitted with only 2 of its 5 KDEs present.

This script reconstructs, for every (rock, formation, bin) in the
universe, which class it falls into:

  no_data            -- 0 rows in the parquet for that triple.
                        Most of these are geologically impossible
                        combinations.
  too_few            -- 1..29 rows; cell rejected at admission.
                        Cells the bank wishes it had but the real
                        data isn't dense enough.
  fitted_full        -- admitted; all 5 variables have a KDE.
  fitted_partial_kde -- admitted; one or more variables missing
                        a KDE because they had < 20 clean samples.

Read-only on data/clean/{samples.parquet, distributions.pkl}.

Outputs:
  plots/analysis/bank_fit_audit.csv             one row per triple
  plots/analysis/bank_fit_audit_summary.md      headline + top-10 gaps
  plots/garcon/bank_fit_audit_classes.png       per-formation panel
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.distributions import DistributionBank


VARIABLES = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
DEPTH_MAX = 4400.0
BIN_WIDTH = 10.0
MIN_SAMPLES = 30   # the bank's admission threshold


CLASS_ORDER = ["no_data", "too_few",
               "fitted_partial_kde", "fitted_full"]
CLASS_COLOR = {
    "no_data":            "#f0f0f0",
    "too_few":            "#fdae61",   # orange = gap we wish we had
    "fitted_partial_kde": "#fee08b",   # yellow = partial
    "fitted_full":        "#1a9850",   # green = good
}


def main() -> None:
    bank = DistributionBank.load("data/clean/distributions.pkl")
    print(f"bank: {len(bank.cells)} cells, "
          f"{len(bank.rock_types)} rocks x {len(bank.formations)} "
          f"formations x ~{int(DEPTH_MAX / BIN_WIDTH)} bins")

    df = pd.read_parquet("data/clean/samples.parquet")
    df = df[df.measurement.isin(VARIABLES)]
    df = df[df.depth.between(0, DEPTH_MAX)]
    print(f"parquet: {len(df):,} rows after filtering")

    # pivot to (well, depth, rock, formation) wide once — matches the bank.fit logic
    wide = df.pivot_table(
        index=["dataset", "borehole", "depth", "rock_type_fine", "formation"],
        columns="measurement", values="value", aggfunc="mean",
    ).reset_index()
    wide = wide[wide.rock_type_fine.isin(bank.rock_types)]
    wide = wide[wide.formation.isin(bank.formations)]
    wide["bin"] = (wide.depth // BIN_WIDTH).astype(int)
    print(f"wide: {len(wide):,} (well, depth) rows")

    # row counts per (rock, formation, bin)
    counts = (wide.groupby(["rock_type_fine", "formation", "bin"])
              .size().rename("n_rows").reset_index())
    print(f"non-empty triples in real data: {len(counts):,}")

    # build universe
    n_bins = int(DEPTH_MAX // BIN_WIDTH)
    rows = []
    counts_map = {(r, f, b): int(n) for r, f, b, n
                  in counts[["rock_type_fine", "formation", "bin", "n_rows"]].itertuples(index=False)}

    for rock in bank.rock_types:
        for fm in bank.formations:
            for b in range(n_bins):
                n = counts_map.get((rock, fm, b), 0)
                # check if admitted (in bank.cells)
                cell = bank.cells.get((rock, fm, b))
                if cell is None:
                    cls = "no_data" if n == 0 else "too_few"
                    n_kdes = 0
                else:
                    n_kdes = len(cell.kdes)
                    cls = "fitted_full" if n_kdes == 5 else "fitted_partial_kde"
                rows.append({
                    "rock": rock, "formation": fm, "bin": b,
                    "depth_lo": b * BIN_WIDTH,
                    "n_rows": n,
                    "n_kdes": n_kdes,
                    "class": cls,
                })
    audit = pd.DataFrame(rows)
    print(f"\nuniverse: {len(audit):,} (rock, formation, bin) triples")

    # ---- headline counts ----
    class_counts = audit["class"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    print("\nclass counts:")
    for c in CLASS_ORDER:
        n = int(class_counts.get(c, 0))
        print(f"  {c:22s}  {n:>8,}  ({100*n/len(audit):.1f}%)")

    # sanity check: bank may have cells beyond the encoder's 0--4400m
    # window (the fitter looks at bins 0..600 since depth_bins go up to
    # 6000m). Those are not used by the encoder but count toward
    # len(bank.cells). We surface that as an informational note.
    n_admitted = (audit["class"].isin(["fitted_full", "fitted_partial_kde"])).sum()
    n_beyond_window = len(bank.cells) - n_admitted
    print(f"\nadmitted in audit window (0--{DEPTH_MAX:.0f}m): {n_admitted}")
    if n_beyond_window > 0:
        print(f"  ({n_beyond_window} extra cells in bank.cells beyond "
              f"{DEPTH_MAX:.0f}m — not used by the encoder)")

    # ---- per-formation breakdown ----
    by_fm = (audit.groupby(["formation", "class"]).size()
             .unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0))
    print("\nper-formation:")
    print(by_fm)

    # ---- top-10 "real data exists but didn't make it" gaps ----
    # rock present in the formation (>= 100 total rows) but >= 30% of
    # attested bins (with > 0 rows) fall in too_few
    gaps = []
    for (rock, fm), g in audit.groupby(["rock", "formation"]):
        total_rows = g.n_rows.sum()
        if total_rows < 100:
            continue
        attested = g[g.n_rows > 0]
        if len(attested) == 0:
            continue
        too_few_attested = (attested["class"] == "too_few").sum()
        pct_too_few = 100.0 * too_few_attested / len(attested)
        if pct_too_few < 30:
            continue
        gaps.append({
            "rock": rock, "formation": fm,
            "total_rows": int(total_rows),
            "attested_bins": int(len(attested)),
            "too_few_bins": int(too_few_attested),
            "pct_too_few": pct_too_few,
            "fitted_bins": int((attested["class"].str.startswith("fitted")).sum()),
        })
    gaps_df = pd.DataFrame(gaps).sort_values("pct_too_few", ascending=False)
    print(f"\ngap candidates (rock present, >=30% bins below 30-sample bar): {len(gaps_df)}")
    if len(gaps_df):
        print(gaps_df.head(10).to_string(index=False))

    # ---- write CSV ----
    audit_path = Path("plots/analysis/bank_fit_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(f"\nwrote {audit_path}  ({len(audit):,} rows)")

    # ---- write markdown summary ----
    lines = [
        "# DistributionBank fit audit",
        "",
        f"Generated by `scripts/diagnostics/bank_fit_audit.py`.",
        "",
        f"Universe: {len(audit):,} (rock, formation, bin) triples across",
        f"{len(bank.rock_types)} rock types, {len(bank.formations)} formations, "
        f"{n_bins} depth bins of {BIN_WIDTH:.0f}\\,m.",
        "",
        "## Headline counts",
        "",
        "| Class | Count | % of universe |",
        "|---|---:|---:|",
    ]
    for c in CLASS_ORDER:
        n = int(class_counts.get(c, 0))
        lines.append(f"| `{c}` | {n:,} | {100*n/len(audit):.1f}% |")
    fitted_total = class_counts.get("fitted_full", 0) + class_counts.get("fitted_partial_kde", 0)
    lines += [
        f"| **fitted total** | **{int(fitted_total):,}** | **{100*fitted_total/len(audit):.1f}%** |",
        "",
        "Of the {:,} admitted cells, {:.1f}% have a full 5-variable KDE; the rest "
        "have one or more variables silently dropped because their per-variable sample "
        "count fell below 20 after bounds clipping + winsorising.".format(
            int(fitted_total),
            100 * class_counts.get("fitted_full", 0) / max(int(fitted_total), 1),
        ),
        "",
        "## Per-formation breakdown",
        "",
        "| Formation | no_data | too_few | partial_kde | full |",
        "|---|---:|---:|---:|---:|",
    ]
    for fm, row in by_fm.iterrows():
        lines.append(f"| {fm} | {int(row['no_data']):,} | {int(row['too_few']):,} | "
                     f"{int(row['fitted_partial_kde']):,} | {int(row['fitted_full']):,} |")
    lines += [
        "",
        "## Top-10 \"present but undersampled\" rock-formation pairs",
        "",
        "Rock types attested in real data (>=100 rows in the formation) where "
        "30%+ of the attested bins fall below the 30-sample admission bar. "
        "These are the gaps the bank wishes it could fill.",
        "",
    ]
    if len(gaps_df) == 0:
        lines.append("_(none — every well-attested rock has dense enough sampling)_")
    else:
        lines += [
            "| rock | formation | total rows | attested bins | too_few bins | % too_few |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for _, r in gaps_df.head(10).iterrows():
            lines.append(f"| {r.rock} | {r.formation} | {r.total_rows:,} | "
                         f"{r.attested_bins} | {r.too_few_bins} | {r.pct_too_few:.0f}% |")

    md_path = Path("plots/analysis/bank_fit_audit_summary.md")
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")

    # ---- write heatmap ----
    formations_to_plot = [fm for fm in bank.formations
                          if (audit[(audit.formation == fm)
                                    & audit["class"].isin(["fitted_full",
                                                            "fitted_partial_kde",
                                                            "too_few"])])
                              .shape[0] > 30]
    if not formations_to_plot:
        formations_to_plot = bank.formations
    ncols = 3
    nrows = (len(formations_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows),
                              squeeze=False)
    class_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    cmap = ListedColormap([CLASS_COLOR[c] for c in CLASS_ORDER])
    norm = BoundaryNorm(boundaries=np.arange(-0.5, len(CLASS_ORDER), 1), ncolors=len(CLASS_ORDER))
    for k, fm in enumerate(formations_to_plot):
        ax = axes[k // ncols][k % ncols]
        sub = audit[audit.formation == fm].copy()
        sub["class_i"] = sub["class"].map(class_idx)
        grid = (sub.pivot_table(index="rock", columns="bin",
                                 values="class_i", aggfunc="first")
                .reindex(index=bank.rock_types))
        ax.imshow(grid.values, cmap=cmap, norm=norm,
                   aspect="auto", interpolation="nearest",
                   extent=[0, n_bins * BIN_WIDTH, len(bank.rock_types), 0])
        ax.set_title(fm, fontsize=10)
        ax.set_yticks(np.arange(len(bank.rock_types)) + 0.5)
        ax.set_yticklabels(bank.rock_types, fontsize=6)
        if k // ncols == nrows - 1:
            ax.set_xlabel("Depth (m)")
    # hide unused axes
    for k in range(len(formations_to_plot), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASS_ORDER]
    fig.legend(handles, CLASS_ORDER, loc="lower center", ncol=len(CLASS_ORDER),
                bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle("Bank-fit audit: cell class per (rock, formation, depth bin)",
                  fontsize=11, y=1.0)
    fig.tight_layout()
    out_png = Path("plots/garcon/bank_fit_audit_classes.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
