"""
Option C -- reclassify the coarse rock_type_fine labels using
each row's wireline signature.

Two classification methods are supported:

  STRICT  - hardcoded literature thresholds (Asquith & Krygowski 2004):
            sandstone     GR<60  -> sandstone_clean
                          GR>=60 -> sandstone_shaly
            claystone     GR>80  -> claystone_hot
                          GR<=80 -> claystone_cool
            clay          -> claystone_cool (Quaternary)
            other (ZE)    rhob<2.20 & nphi<0.05 -> halite_pure
                          rhob>2.85 & nphi<0.10 -> anhydrite
                          rhob in [2.80,2.95] & nphi in [0.02,0.15] -> dolomite

  DATA    - data-driven nearest fine class with a per-formation
            adaptive depth window (v7). For each
            (formation, fine_class), aggregate per-10m-bin running
            sums of the 5 wireline channels in samples.parquet.
            For each coarse row at depth d in formation fm, expand
            the window around d's bin until >= min_rows are
            collected for the candidate class, then compute the
            mean per-channel z-score. Assign to argmin if
            min(z) <= max_distance AND at least min_channels
            wireline values are present. No off-formation or
            geology fallback -- unmatched rows are dropped.

  COMPARE - runs both methods on every coarse row, produces an
            agreement matrix + disagreement examples. No apply
            mode here -- pick one method first.

Run examples:

  python scripts/reclassify_coarse_rocks.py
      report-only, method=strict (default)

  python scripts/reclassify_coarse_rocks.py --method data
      report-only, method=data-driven

  python scripts/reclassify_coarse_rocks.py --method compare
      compares both, no parquet write possible from this mode

  python scripts/reclassify_coarse_rocks.py --method data --apply
      writes the reclassified samples.parquet (after backup)

LILY rows are passed through unchanged (different lithology
naming, no coarse classes anyway).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COARSE_CLASSES = {"sandstone", "claystone", "clay", "other"}

# `clay` is its own valid rock-physics class (Quaternary
# unconsolidated; distinct from lithified claystones). It is
# NEVER reassigned by the data-driven matcher; it just passes
# through unchanged.
PROTECTED_COARSE = {"clay"}

# All five wireline channels are used by the data-driven matcher.
ALL_WIRELINE = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]

# Strict-mode subset (only the channels each rule needs).
WIRELINE_CHANNELS = ["gr_api", "rhob", "nphi"]

# Bin width for the adaptive depth window (matches the bank's
# native resolution).
DATA_BIN_WIDTH_M = 10.0

# Candidate fine classes per coarse class (used by data-driven matcher).
CANDIDATE_FINE = {
    "sandstone": ["sandstone_clean", "sandstone_shaly"],
    "claystone": ["claystone_cool", "claystone_hot"],
    # clay is protected — never reassigned.
    "other":     ["halite_pure", "anhydrite", "dolomite",
                  "claystone_cool", "claystone_hot"],
}

# v7: no geology fallback. The per-formation adaptive depth window
# is the only path; rows that cannot be matched against their own
# formation's fine-class templates are dropped.
GEOLOGICAL_DEFAULT: dict[tuple[str, str], str] = {}


def classify_strict(rt: str, fm: str | None,
                     gr: float | None, rhob: float | None,
                     nphi: float | None) -> str:
    """Return new fine label, the literal "DROP", or rt unchanged."""
    if rt not in COARSE_CLASSES:
        return rt

    if rt == "clay":
        # Quaternary clays (NU/NM/NL) — geologically uniform, no
        # wireline needed
        return "claystone_cool"

    if rt == "sandstone":
        if gr is None or not np.isfinite(gr):
            return "DROP"
        return "sandstone_clean" if gr < 60.0 else "sandstone_shaly"

    if rt == "claystone":
        if gr is None or not np.isfinite(gr):
            return "DROP"
        return "claystone_hot" if gr > 80.0 else "claystone_cool"

    if rt == "other":
        # ZE-only triage
        if fm != "ZE":
            return "DROP"
        if rhob is None or not np.isfinite(rhob):
            return "DROP"
        if nphi is None or not np.isfinite(nphi):
            return "DROP"
        if rhob < 2.20 and nphi < 0.05:
            return "halite_pure"
        if rhob > 2.85 and nphi < 0.10:
            return "anhydrite"
        if 2.80 <= rhob <= 2.95 and 0.02 <= nphi <= 0.15:
            return "dolomite"
        return "DROP"

    return rt


# ------------------------------------------------------------- data-driven --

def build_template_grids(df: pd.DataFrame
                          ) -> dict[tuple[str, str], pd.DataFrame]:
    """Per-(formation, fine_class) per-bin row counts and running
    sum / sum-of-squares for each wireline channel.

    Returns a dict keyed by (formation, fine_class) -> DataFrame
    with columns:
        bin              -- depth bin index (10 m bins)
        n                -- row count in that bin (rows w/ ANY of the 5 channels)
        n_<var>          -- non-null row count for each variable
        sum_<var>        -- sum of values
        sum2_<var>       -- sum of squared values

    The grid lets us aggregate over an expanding depth window
    cheaply at match time -- mean = sum / n, var = sum2 / n - mean^2.
    """
    nlog = df[df.dataset == "NLOG"]
    wide = (
        nlog[nlog.measurement.isin(ALL_WIRELINE)]
            .pivot_table(
                index=["borehole", "depth", "rock_type_fine", "formation"],
                columns="measurement",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
    )
    for ch in ALL_WIRELINE:
        if ch not in wide.columns:
            wide[ch] = np.nan
    # only fine-class rows feed the templates
    fine_only = wide[
        ~wide.rock_type_fine.isin(COARSE_CLASSES) & wide.rock_type_fine.notna()
    ].copy()
    fine_only["bin"] = (fine_only.depth // DATA_BIN_WIDTH_M).astype(int)

    grids: dict[tuple[str, str], pd.DataFrame] = {}
    for (fm, fcls), g in fine_only.groupby(["formation", "rock_type_fine"]):
        rows = []
        for b, gb in g.groupby("bin"):
            row = {"bin": int(b), "n": len(gb)}
            for v in ALL_WIRELINE:
                vals = gb[v].dropna().values
                row[f"n_{v}"] = int(len(vals))
                row[f"sum_{v}"] = float(vals.sum())
                row[f"sum2_{v}"] = float((vals ** 2).sum())
            rows.append(row)
        gdf = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
        if not gdf.empty:
            grids[(fm, fcls)] = gdf
    return grids


def build_global_grids(grids: dict[tuple[str, str], pd.DataFrame]
                        ) -> dict[str, pd.DataFrame]:
    """Aggregate the per-(formation, fine_class) grids across all
    formations, keyed by fine_class only. Used as the
    any-formation fallback.
    """
    by_fine: dict[str, list[pd.DataFrame]] = {}
    for (fm, fcls), gdf in grids.items():
        by_fine.setdefault(fcls, []).append(gdf)
    out: dict[str, pd.DataFrame] = {}
    for fcls, dfs in by_fine.items():
        merged = pd.concat(dfs, ignore_index=True)
        # sum per bin across formations
        agg = (merged.groupby("bin").sum(numeric_only=True)
               .reset_index().sort_values("bin").reset_index(drop=True))
        out[fcls] = agg
    return out


def adaptive_template(grid: pd.DataFrame, query_bin: int,
                       min_rows: int, max_half_width: int
                       ) -> dict[str, tuple[float, float]] | None:
    """Aggregate grid rows in an expanding window centred on
    query_bin until n >= min_rows or the half-width reaches
    max_half_width.

    Returns {var: (mean, std)} per variable that has at least
    min_rows samples in the window; or None if the window cap
    is reached with insufficient data.
    """
    if grid.empty:
        return None
    # set index by bin for O(1) lookup
    g = grid.set_index("bin")
    bins_present = g.index
    half = 0
    while half <= max_half_width:
        window = g.loc[(bins_present >= query_bin - half)
                        & (bins_present <= query_bin + half)]
        if window.n.sum() >= min_rows or half == max_half_width:
            break
        half += 1
    if window.n.sum() < min_rows:
        return None
    stats: dict[str, tuple[float, float]] = {}
    for v in ALL_WIRELINE:
        nv = window[f"n_{v}"].sum()
        if nv < min_rows:
            continue
        mu = window[f"sum_{v}"].sum() / nv
        ex2 = window[f"sum2_{v}"].sum() / nv
        var = max(ex2 - mu * mu, 1e-12)
        std = float(np.sqrt(var))
        if std < 1e-6:
            continue
        stats[v] = (float(mu), std)
    return stats if len(stats) >= 2 else None


def _score_against_grids(candidates: list[str],
                          query_bin: int,
                          row_values: dict[str, float],
                          available: list[str],
                          lookup_grid,                 # callable: cls -> grid|None
                          min_rows: int, max_half_width: int,
                          ) -> list[tuple[float, str]]:
    """Score the row against each candidate fine class. lookup_grid
    is a callable that returns the grid for a given fine_class
    (either formation-restricted or global)."""
    scored = []
    for cls in candidates:
        grid = lookup_grid(cls)
        if grid is None:
            continue
        tmpl = adaptive_template(grid, query_bin,
                                  min_rows=min_rows,
                                  max_half_width=max_half_width)
        if tmpl is None:
            continue
        zs = []
        for v in available:
            if v in tmpl:
                m, s = tmpl[v]
                zs.append(abs(row_values[v] - m) / s)
        if not zs:
            continue
        scored.append((float(np.mean(zs)), cls))
    scored.sort()
    return scored


def classify_data(rt: str, fm: str | None, depth: float,
                   row_values: dict[str, float],
                   grids: dict[tuple[str, str], pd.DataFrame],
                   global_grids: dict[str, pd.DataFrame],
                   min_channels: int = 2,
                   max_distance: float = 2.0,
                   min_rows: int = 50,
                   max_half_width: int = 50,
                  ) -> tuple[str, float, str | None, str]:
    """Data-driven nearest-fine-class match with a per-formation
    adaptive depth window. No off-formation or geology fallback.

    Returns (decision, distance, runner_up, reason).
    `reason` is one of:
      - "passthrough"       row is not coarse, returned unchanged
      - "protected"         rt == "clay", returned unchanged
      - "data_formation"    matched a formation-restricted template
      - "drop"              no usable template in this formation
    """
    del global_grids  # v7: only formation-restricted templates are used
    if rt not in COARSE_CLASSES:
        return rt, 0.0, None, "passthrough"
    if rt in PROTECTED_COARSE:                # clay -> pass through
        return rt, 0.0, None, "protected"
    if fm is None or not np.isfinite(depth):
        return "DROP", float("inf"), None, "drop"

    available = [v for v in ALL_WIRELINE
                 if v in row_values and np.isfinite(row_values[v])]
    if len(available) < min_channels:
        return "DROP", float("inf"), None, "drop"

    query_bin = int(depth // DATA_BIN_WIDTH_M)
    candidates = CANDIDATE_FINE.get(rt, [])
    scored = _score_against_grids(
        candidates, query_bin, row_values, available,
        lookup_grid=lambda cls: grids.get((fm, cls)),
        min_rows=min_rows, max_half_width=max_half_width)
    if scored and scored[0][0] <= max_distance:
        best_d, best_cls = scored[0]
        runner = scored[1][1] if len(scored) > 1 else None
        return best_cls, best_d, runner, "data_formation"

    return "DROP", float("inf"), None, "drop"


# -------------------------------------------------------------------- strict --

def reclassify(df: pd.DataFrame
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the rules.

    Returns (decision_df, summary_df) where:
      decision_df has one row per (dataset, borehole, depth,
        rock_type_fine, formation) with columns:
          old_rock_type_fine, new_rock_type_fine, gr, rhob, nphi
      summary_df has per-(old_rock_type_fine, new_rock_type_fine)
        counts.

    decision_df only contains coarse-class rows (the ones we
    are deciding on); fine-class rows are unchanged elsewhere.
    """
    nlog = df[df.dataset == "NLOG"].copy()

    # build a per-(well, depth) wireline lookup
    wireline = (
        nlog[nlog.measurement.isin(WIRELINE_CHANNELS)]
            .pivot_table(
                index=["borehole", "depth"],
                columns="measurement",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
    )
    for ch in WIRELINE_CHANNELS:
        if ch not in wireline.columns:
            wireline[ch] = np.nan

    # one decision per (borehole, depth, rock_type_fine, formation)
    coarse_rows = (
        nlog[nlog.rock_type_fine.isin(COARSE_CLASSES)]
            [["borehole", "depth", "rock_type_fine", "formation"]]
            .drop_duplicates()
            .reset_index(drop=True)
    )
    coarse_rows = coarse_rows.merge(
        wireline[["borehole", "depth"] + WIRELINE_CHANNELS],
        on=["borehole", "depth"], how="left")

    new_labels = []
    for _, r in coarse_rows.iterrows():
        new_labels.append(classify_strict(
            r.rock_type_fine, r.formation,
            r.gr_api, r.rhob, r.nphi))
    coarse_rows["new_rock_type_fine"] = new_labels

    summary = (coarse_rows.groupby(["rock_type_fine", "new_rock_type_fine"])
               .size().reset_index(name="n_rows"))
    summary = summary.sort_values(["rock_type_fine", "n_rows"],
                                    ascending=[True, False])
    return coarse_rows, summary


def apply_reclassification(df: pd.DataFrame,
                            decisions: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-(well, depth, formation, rock_type_fine)
    decisions back to the long-format samples dataframe.

    Drops every NLOG row at a (well, depth) pair that the decision
    table marked DROP.  Renames the rock_type_fine value of the
    remaining coarse-class rows to the new label.
    """
    drop_keys = set(map(
        tuple,
        decisions[decisions.new_rock_type_fine == "DROP"]
            [["borehole", "depth", "rock_type_fine"]].values))
    rename_map = {
        (b, d, old): new for b, d, old, new in decisions[
            decisions.new_rock_type_fine != "DROP"
        ][["borehole", "depth", "rock_type_fine",
            "new_rock_type_fine"]].itertuples(index=False)
    }
    is_nlog = df.dataset == "NLOG"
    is_coarse = df.rock_type_fine.isin(COARSE_CLASSES)
    nlog_coarse = df[is_nlog & is_coarse]
    keep_mask = ~nlog_coarse.apply(
        lambda r: (r.borehole, r.depth, r.rock_type_fine) in drop_keys,
        axis=1)
    survivors = nlog_coarse[keep_mask].copy()
    survivors["rock_type_fine"] = survivors.apply(
        lambda r: rename_map.get(
            (r.borehole, r.depth, r.rock_type_fine), r.rock_type_fine),
        axis=1)

    # rebuild
    not_changed = df[~(is_nlog & is_coarse)]
    new_df = pd.concat([not_changed, survivors], ignore_index=True)
    return new_df


def write_report(summary: pd.DataFrame, decisions: pd.DataFrame,
                  out_md: Path, out_csv: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out_csv, index=False)

    total_coarse = int(summary.n_rows.sum())
    total_drops = int(summary[summary.new_rock_type_fine == "DROP"].n_rows.sum())
    total_kept = total_coarse - total_drops

    has_reason = "reason" in summary.columns

    lines = [
        "# Option C reclassification report (REPORT MODE -- nothing written yet)",
        "",
        f"Total coarse-class (well, depth, fm) rows considered: "
        f"**{total_coarse:,}**",
        f"Would be kept and renamed: **{total_kept:,}** "
        f"({100 * total_kept / max(total_coarse, 1):.1f} %)",
        f"Would be **dropped**: **{total_drops:,}** "
        f"({100 * total_drops / max(total_coarse, 1):.1f} %)",
        "",
    ]
    if has_reason:
        lines += [
            "v7 rule: per-formation adaptive depth window only. "
            "If the row has fewer than 2 wireline channels, or no "
            "candidate template can be built within the window for "
            "this row's formation, the row is dropped. No "
            "off-formation or geology fallback.",
            "",
            "## Per-class outcome (with reason)",
            "",
            "| coarse class | reason | new label | rows | % of class |",
            "|---|---|---|---:|---:|",
        ]
        for coarse_cls in sorted(summary.rock_type_fine.unique()):
            cls_total = summary[summary.rock_type_fine == coarse_cls].n_rows.sum()
            sub = summary[summary.rock_type_fine == coarse_cls]
            for _, row in sub.iterrows():
                tag = ("**DROP**" if row.new_rock_type_fine == "DROP"
                       else f"`{row.new_rock_type_fine}`")
                lines.append(
                    f"| `{coarse_cls}` | {row.reason} | {tag} "
                    f"| {int(row.n_rows):,} "
                    f"| {100 * row.n_rows / cls_total:.1f} % |")
        # also a reason-only summary
        lines += [
            "",
            "## Reason summary",
            "",
            "| reason | rows | description |",
            "|---|---:|---|",
        ]
        reason_text = {
            "passthrough":    "not a coarse row (unreachable here)",
            "protected":      "`clay` -- left unchanged",
            "data_formation": "matched a per-(formation, fine class) template",
            "drop":           "no usable per-formation template (or <2 wireline channels)",
        }
        reason_counts = summary.groupby("reason").n_rows.sum().sort_values(
            ascending=False)
        for reason, n in reason_counts.items():
            lines.append(f"| `{reason}` | {int(n):,} | {reason_text.get(reason, '')} |")
    else:
        lines += [
            "## Per-class outcome",
            "",
            "| coarse class | new label | rows | % of class |",
            "|---|---|---:|---:|",
        ]
        for coarse_cls in sorted(summary.rock_type_fine.unique()):
            cls_total = summary[summary.rock_type_fine == coarse_cls].n_rows.sum()
            for _, row in summary[summary.rock_type_fine == coarse_cls].iterrows():
                tag = ("**DROP**" if row.new_rock_type_fine == "DROP"
                       else f"`{row.new_rock_type_fine}`")
                lines.append(
                    f"| `{coarse_cls}` | {tag} | {int(row.n_rows):,} "
                    f"| {100 * row.n_rows / cls_total:.1f} % |")

    # per-formation drop breakdown for the drops
    drops = decisions[decisions.new_rock_type_fine == "DROP"]
    if len(drops):
        by_fm = (drops.groupby(["rock_type_fine", "formation"])
                 .size().reset_index(name="n_rows"))
        lines += [
            "",
            "## Drop breakdown by (coarse class, formation)",
            "",
            "| coarse class | formation | rows dropped |",
            "|---|---|---:|",
        ]
        for _, r in by_fm.sort_values("n_rows", ascending=False).iterrows():
            lines.append(f"| `{r.rock_type_fine}` | {r.formation} "
                          f"| {int(r.n_rows):,} |")

    lines += [
        "",
        "## Next step",
        "",
        f"If these numbers look right, re-run with `--apply` to write "
        f"the reclassified `samples.parquet`. The old parquet is "
        f"backed up to `samples.parquet.bak_pre_optionC` before the "
        f"overwrite.",
        "",
        f"Per-row decisions: `{out_csv}` "
        f"({len(decisions):,} rows).",
    ]
    out_md.write_text("\n".join(lines))


def reclassify_data_driven(df: pd.DataFrame,
                            grids,
                            global_grids,
                            min_channels: int = 2,
                            max_distance: float = 2.0,
                            min_rows: int = 50,
                            max_half_width: int = 50,
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same shape as `reclassify` but uses the data-driven matcher
    with a per-formation adaptive depth window only. Rows that
    cannot be matched against their own formation's fine-class
    templates are dropped."""
    nlog = df[df.dataset == "NLOG"].copy()
    wide = (
        nlog[nlog.measurement.isin(ALL_WIRELINE)]
            .pivot_table(
                index=["borehole", "depth"],
                columns="measurement",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
    )
    for ch in ALL_WIRELINE:
        if ch not in wide.columns:
            wide[ch] = np.nan

    coarse_rows = (
        nlog[nlog.rock_type_fine.isin(COARSE_CLASSES)]
            [["borehole", "depth", "rock_type_fine", "formation"]]
            .drop_duplicates()
            .reset_index(drop=True)
    )
    coarse_rows = coarse_rows.merge(
        wide[["borehole", "depth"] + ALL_WIRELINE],
        on=["borehole", "depth"], how="left")

    decisions, dists, runners, reasons = [], [], [], []
    for _, r in coarse_rows.iterrows():
        rv = {v: r[v] for v in ALL_WIRELINE}
        d, dist, runner, reason = classify_data(
            r.rock_type_fine, r.formation, r.depth, rv,
            grids, global_grids,
            min_channels=min_channels, max_distance=max_distance,
            min_rows=min_rows, max_half_width=max_half_width)
        decisions.append(d)
        dists.append(dist)
        runners.append(runner)
        reasons.append(reason)
    coarse_rows["new_rock_type_fine"] = decisions
    coarse_rows["distance"] = dists
    coarse_rows["runner_up"] = runners
    coarse_rows["reason"] = reasons

    summary = (coarse_rows.groupby(["rock_type_fine",
                                      "new_rock_type_fine", "reason"])
               .size().reset_index(name="n_rows"))
    summary = summary.sort_values(
        ["rock_type_fine", "n_rows"], ascending=[True, False])
    return coarse_rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path,
                   default=Path("data/clean/samples.parquet"))
    p.add_argument("--method", choices=["strict", "data", "compare"],
                   default="data",
                   help="strict = hardcoded GR/rhob thresholds; "
                        "data = data-driven nearest-fine-class with "
                        "adaptive depth window (default); "
                        "compare = run both and report agreement.")
    p.add_argument("--min-channels", type=int, default=2,
                   help="data mode: minimum wireline channels for a "
                        "confident decision (default 2).")
    p.add_argument("--max-distance", type=float, default=2.0,
                   help="data mode: max mean per-channel z-score "
                        "before dropping (default 2.0).")
    p.add_argument("--min-rows", type=int, default=50,
                   help="data mode: minimum rows aggregated by the "
                        "adaptive window before computing a template "
                        "(default 50).")
    p.add_argument("--max-half-width-bins", type=int, default=50,
                   help="data mode: cap on the adaptive window half-"
                        "width in 10m bins (default 50 = +/-500m).")
    p.add_argument("--apply", action="store_true",
                   help="Actually write the reclassified parquet "
                        "(otherwise: report-only). Not allowed in "
                        "compare mode.")
    p.add_argument("--report-md", type=Path,
                   default=Path("plots/analysis/reclassify_report.md"))
    p.add_argument("--report-csv", type=Path,
                   default=Path("plots/analysis/reclassify_decisions.csv"))
    args = p.parse_args()

    print(f"loading {args.samples} ...")
    df = pd.read_parquet(args.samples)
    print(f"  {len(df):,} rows")

    # ---------- compare-mode: run both, write side-by-side report ----------
    if args.method == "compare":
        if args.apply:
            print("--apply is not allowed in --method compare.")
            return
        print("classifying via STRICT thresholds ...")
        dec_strict, _ = reclassify(df)
        print("building per-(formation, fine_class) bin grids ...")
        grids = build_template_grids(df)
        global_grids = build_global_grids(grids)
        print(f"  {len(grids)} (formation, fine_class) grids; "
              f"{len(global_grids)} global fine_class grids")
        print("classifying via DATA-DRIVEN nearest fine class ...")
        dec_data, _ = reclassify_data_driven(
            df, grids, global_grids,
            min_channels=args.min_channels,
            max_distance=args.max_distance,
            min_rows=args.min_rows,
            max_half_width=args.max_half_width_bins)

        # join on (borehole, depth, rock_type_fine, formation)
        join_keys = ["borehole", "depth", "rock_type_fine", "formation"]
        merged = dec_strict[join_keys + ["new_rock_type_fine"]].rename(
            columns={"new_rock_type_fine": "decision_strict"}).merge(
                dec_data[join_keys + ["new_rock_type_fine", "distance",
                                       "runner_up"]].rename(
                    columns={"new_rock_type_fine": "decision_data"}),
                on=join_keys, how="outer")

        merged["agree"] = merged.decision_strict == merged.decision_data
        n_total = len(merged)
        n_agree = int(merged.agree.sum())
        print(f"\nagreement: {n_agree:,} / {n_total:,} "
              f"({100 * n_agree / max(n_total, 1):.1f} %)")

        print("\nper-coarse-class agreement / disagreement:")
        for coarse_cls in sorted(merged.rock_type_fine.unique()):
            sub = merged[merged.rock_type_fine == coarse_cls]
            n = len(sub)
            agree = int(sub.agree.sum())
            both_drop = int(((sub.decision_strict == "DROP")
                              & (sub.decision_data == "DROP")).sum())
            strict_only_drop = int(((sub.decision_strict == "DROP")
                                     & (sub.decision_data != "DROP")).sum())
            data_only_drop = int(((sub.decision_strict != "DROP")
                                   & (sub.decision_data == "DROP")).sum())
            disagree_assigned = int(((sub.decision_strict != "DROP")
                                      & (sub.decision_data != "DROP")
                                      & ~sub.agree).sum())
            print(f"  {coarse_cls:10s}  n={n:>7,}  "
                  f"agree={agree:>6,} ({100*agree/n:.0f}%)  "
                  f"both_drop={both_drop:>6,}  "
                  f"strict_only_drop={strict_only_drop:>6,}  "
                  f"data_only_drop={data_only_drop:>6,}  "
                  f"disagree_assigned={disagree_assigned:>6,}")

        print("\ndisagreement cross-table (strict_decision -> data_decision):")
        cross = (merged[~merged.agree]
                 .groupby(["rock_type_fine", "decision_strict", "decision_data"])
                 .size().reset_index(name="n").sort_values("n", ascending=False))
        for _, r in cross.head(30).iterrows():
            print(f"  {r.rock_type_fine:10s}  "
                  f"strict={r.decision_strict:>20s}  "
                  f"data={r.decision_data:>20s}  n={int(r.n):>6,}")

        out_csv = args.report_csv.with_name("reclassify_compare_decisions.csv")
        merged.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")
        return

    # ---------- single-method path ----------
    print(f"classifying coarse rows via {args.method.upper()} method ...")
    if args.method == "strict":
        decisions, summary = reclassify(df)
    else:                               # data-driven (adaptive window)
        print("building per-(formation, fine_class) bin grids ...")
        grids = build_template_grids(df)
        global_grids = build_global_grids(grids)
        print(f"  {len(grids)} (formation, fine_class) grids; "
              f"{len(global_grids)} global fine_class grids")
        decisions, summary = reclassify_data_driven(
            df, grids, global_grids,
            min_channels=args.min_channels,
            max_distance=args.max_distance,
            min_rows=args.min_rows,
            max_half_width=args.max_half_width_bins)
    print(f"  {len(decisions):,} (well, depth, fm) coarse decisions")

    write_report(summary, decisions, args.report_md, args.report_csv)
    print(f"\nwrote {args.report_md}")
    print(f"wrote {args.report_csv}")

    # Print summary to console
    print("\n" + "=" * 60)
    total = int(summary.n_rows.sum())
    drops = int(summary[summary.new_rock_type_fine == "DROP"].n_rows.sum())
    print(f"total coarse rows : {total:,}")
    print(f"would be kept     : {total - drops:,} "
          f"({100 * (total - drops) / max(total, 1):.1f} %)")
    print(f"would be dropped  : {drops:,} "
          f"({100 * drops / max(total, 1):.1f} %)")
    print("=" * 60)
    print("\nper-class breakdown:")
    has_reason = "reason" in summary.columns
    for coarse_cls in sorted(summary.rock_type_fine.unique()):
        cls_total = summary[summary.rock_type_fine == coarse_cls].n_rows.sum()
        print(f"\n  {coarse_cls}: {cls_total:,} rows")
        for _, row in summary[summary.rock_type_fine == coarse_cls].iterrows():
            tag = ("DROP" if row.new_rock_type_fine == "DROP"
                   else row.new_rock_type_fine)
            reason = f"[{row.reason}]" if has_reason else ""
            print(f"    -> {tag:18s} {reason:18s} "
                  f"{int(row.n_rows):>10,}  "
                  f"({100 * row.n_rows / cls_total:.1f}%)")

    if not args.apply:
        print("\n[report-only mode]  nothing was written to samples.parquet.")
        print("Review the report; re-run with `--apply` to commit.")
        return

    # ---- apply path ----
    print("\napplying reclassification ...")
    backup = args.samples.with_suffix(args.samples.suffix + ".bak_pre_optionC")
    print(f"  backing up {args.samples} -> {backup}")
    import shutil
    shutil.copyfile(args.samples, backup)

    new_df = apply_reclassification(df, decisions)
    print(f"  new dataframe: {len(new_df):,} rows "
          f"(delta vs original: {len(new_df) - len(df):+,})")
    new_df.to_parquet(args.samples, index=False)
    print(f"  wrote {args.samples}")

    # post-apply sanity check
    after = pd.read_parquet(args.samples)
    rocks = sorted(after[after.dataset == "NLOG"].rock_type_fine.dropna().unique())
    print(f"\nrock_type_fine values in NLOG after apply: {rocks}")
    coarse_still = [r for r in rocks if r in COARSE_CLASSES]
    if coarse_still:
        print(f"WARNING: coarse classes still present: {coarse_still}")
    else:
        print("OK: no coarse classes remain in NLOG.")


if __name__ == "__main__":
    main()
