"""
RVG / Garcon gap analysis.

Question we need to answer:
    Would adding the Roer Valley Graben (RVG) dataset from Garzon
    et al. (2026) improve the formation-geometry side of our
    simulator?

This script does NOT change the simulator. It only measures.

Inputs (read-only):
    data/clean/samples.parquet                            -- our NLOG fit
    data/garcon/DGMplus_extractie/*.csv                   -- RVG strat
    data/garcon/basisdata/basisdata_gef_*.csv             -- RVG lithology
    data/garcon/strat_class_map.csv                       -- sub-unit -> unit
    data/garcon/categorieen_hoofdlitho.csv                -- litho codes

Outputs:
    plots/garcon/coverage_uplift.png        -- per-formation per-depth bin
    plots/garcon/top_depth_shift.png        -- KDEs with/without RVG
    plots/garcon/transition_delta.png       -- markov chain delta
    plots/analysis/rvg_gap_report.md        -- recommendation
    plots/analysis/rvg_gap_metrics.csv      -- raw numbers

Depth convention:
    Our pipeline already uses depth-below-NAP (positive going down)
    after the v3 fix.  Convert:
      - NLOG samples.parquet: `depth` column, already NAP-relative.
      - DGM-plus: depth_below_NAP = TOP - MV  (TOP is below surface,
        MV is surface elevation in NAP; so true depth below NAP
        is TOP minus the surface elevation).
      - Basisdata: depth_below_NAP = -top  (top is NAP elevation,
        negative means below sea).

Mapping RVG -> NLOG formations:
    RVG covers the Cenozoic of the southern Netherlands.  NLOG
    groups its Cenozoic into NU/NM/NL.  The mapping below comes
    from the Dutch stratigraphic nomenclature (TNO-GDN):

      AAOP, NA, EC, NI, BX, BE, KR  -> NU  (Upper North Sea Group)
      ST, SY, PZWA, MS              -> NM  (Middle North Sea Group)
      OO, KI, IE, BRVI              -> NL  (Lower North Sea Group)

    The strat_class_map.csv handles sub-unit to leaf-unit; we then
    apply the table above to lift leaf-unit to NLOG formation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------- mappings ---

# RVG leaf unit -> NLOG formation (Upper / Middle / Lower North Sea Group)
RVG_TO_NLOG = {
    "AAOP": "NU",
    "NA":   "NU", "NUNA": "NU",
    "EC":   "NU",
    "NI":   "NU",
    "BX":   "NU",
    "BE":   "NU",
    "KR":   "NU",
    "UR":   "NU",
    "ST":   "NM",
    "SY":   "NM",
    "PZWA": "NM",
    "MS":   "NM",
    "OO":   "NL",
    "KI":   "NL",
    "IE":   "NL",
    "BRVI": "NL",
}

# RVG lithology code -> our rock_type_fine vocabulary
# (best-effort; some codes have no clean mapping)
RVG_LITHO_TO_FINE = {
    "Z":   "sandstone_clean",   # zand
    "K":   "claystone_cool",    # klei
    "L":   "claystone_cool",    # leem (loam)
    "G":   "sandstone_clean",   # grind (gravel)
    "V":   "claystone_hot",     # veen (peat) - organic-rich
    "BRK": "claystone_hot",     # bruinkool (brown coal) - organic
    "STK": "claystone_hot",     # steenkool (coal)
    "GY":  "claystone_hot",     # gyttja (organic mud)
    "SHE": "chalk",             # schelpen (shells, carbonate)
    "ZNS": "sandstone_clean",   # zandsteen (sandstone)
    "GCZ": "sandstone_shaly",   # glauconietzand (glauconitic sand)
    "KAS": "chalk",             # kalksteen (limestone)
    "LEI": "claystone_cool",    # leisteen (slate)
    "KLS": "claystone_cool",    # kleisteen (claystone)
    "MER": "chalk",             # mergel (marl)
    "SHA": "claystone_cool",    # schalie (shale)
    "DOL": "dolomite",          # dolomiet
    "STN": "sandstone_clean",   # stenen (stones)
    "SIS": "sandstone_shaly",   # ...
}


def load_rvg(garcon_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rvg_strat_intervals, rvg_litho_intervals).

    Both dataframes have columns
        borehole, x_rd, y_rd, surface_elev_nap,
        top_depth_below_nap, bottom_depth_below_nap, label
    where `label` is the formation (NU/NM/NL) for strat and the
    rock_type_fine (or NaN) for litho.
    """
    strat_path = (garcon_dir / "DGMplus_extractie"
                  / "DGMplus_extractie_DGM_v02r2_en_H3OdeKempen_NL_public_domain_RVG.csv")
    lith_path = (garcon_dir / "basisdata"
                 / "basisdata_gef_bovennaaronder_public_domain_RVG.csv")
    smap_path = garcon_dir / "strat_class_map.csv"

    smap = pd.read_csv(smap_path).dropna(subset=["key"])
    smap_dict = dict(zip(smap["key"], smap["value"]))

    def to_leaf_unit(s: str) -> str:
        # if already a leaf (not in map), return as-is
        return smap_dict.get(s, s)

    # ---- strat (DGMplus)
    dgm = pd.read_csv(strat_path)
    dgm = dgm.rename(columns={
        "NR": "borehole", "X": "x_rd", "Y": "y_rd",
        "MV": "surface_elev_nap",
        "TOP": "_top_below_surf", "BASIS": "_bot_below_surf",
        "STRAT": "_strat_raw",
    })
    dgm["top_depth_below_nap"] = dgm["_top_below_surf"] - dgm["surface_elev_nap"]
    dgm["bottom_depth_below_nap"] = dgm["_bot_below_surf"] - dgm["surface_elev_nap"]
    dgm["_strat_leaf"] = dgm["_strat_raw"].astype(str).map(to_leaf_unit)
    dgm["label"] = dgm["_strat_leaf"].map(RVG_TO_NLOG)
    # unmapped strat codes -> drop (e.g. NN, RU, LA, HO, DIEP, VA, VE)
    dgm = dgm.dropna(subset=["label"])
    strat_df = dgm[["borehole", "x_rd", "y_rd", "surface_elev_nap",
                    "top_depth_below_nap", "bottom_depth_below_nap", "label",
                    "_strat_leaf", "_strat_raw"]].copy()

    # ---- litho (basisdata)
    lit = pd.read_csv(lith_path)
    lit = lit.rename(columns={
        "nr": "borehole", "x": "x_rd", "y": "y_rd",
        "mv": "surface_elev_nap",
        "top": "_top_nap", "bottom": "_bot_nap",
        "lith": "_lith_raw",
    })
    # in basisdata, top/bottom are NAP-elevation (negative = below sea).
    # depth_below_nap = -elevation
    lit["top_depth_below_nap"] = -lit["_top_nap"]
    lit["bottom_depth_below_nap"] = -lit["_bot_nap"]
    lit["label"] = lit["_lith_raw"].astype(str).map(RVG_LITHO_TO_FINE)
    litho_df = lit[["borehole", "x_rd", "y_rd", "surface_elev_nap",
                    "top_depth_below_nap", "bottom_depth_below_nap", "label",
                    "_lith_raw"]].copy()

    return strat_df, litho_df


# ----------------------------------------------------------- coverage uplift --

def coverage_uplift(nlog_df: pd.DataFrame, rvg_strat: pd.DataFrame,
                    bin_width: float = 10.0,
                    max_depth: float = 200.0) -> pd.DataFrame:
    """How many wells does each formation have per 10 m bin, with
    and without RVG, in the upper `max_depth` metres?

    Returns a long dataframe with columns
        formation, depth_bin_centre, source ('nlog' | 'nlog+rvg'),
        n_wells, n_intervals
    """
    bins = np.arange(0, max_depth + bin_width, bin_width)
    centres = (bins[:-1] + bins[1:]) / 2

    rows = []
    formations = ["NU", "NM", "NL"]

    for fm in formations:
        # --- NLOG: per-(well, depth) intervals, one row each
        nlog_fm = (nlog_df[(nlog_df.formation == fm) & nlog_df.depth.between(0, max_depth)]
                   .drop_duplicates(["borehole", "depth"])[["borehole", "depth"]])
        for c, (lo, hi) in zip(centres, zip(bins[:-1], bins[1:])):
            sub = nlog_fm[(nlog_fm.depth >= lo) & (nlog_fm.depth < hi)]
            rows.append({
                "formation": fm, "depth_bin_centre": c, "source": "nlog",
                "n_wells": sub.borehole.nunique(), "n_intervals": len(sub),
            })

        # --- RVG: per-(well, top..bot) interval; expand to bin counts
        rvg_fm = rvg_strat[rvg_strat.label == fm]
        for c, (lo, hi) in zip(centres, zip(bins[:-1], bins[1:])):
            # interval overlaps bin if top<hi and bot>lo
            sub = rvg_fm[(rvg_fm.top_depth_below_nap < hi)
                         & (rvg_fm.bottom_depth_below_nap > lo)]
            rows.append({
                "formation": fm, "depth_bin_centre": c, "source": "rvg",
                "n_wells": sub.borehole.nunique(), "n_intervals": len(sub),
            })

    df = pd.DataFrame(rows)
    return df


def plot_coverage(uplift_df: pd.DataFrame, out_path: Path) -> None:
    fms = uplift_df.formation.unique()
    fig, axes = plt.subplots(1, len(fms), figsize=(4.5 * len(fms), 4.5),
                             sharey=True)
    if len(fms) == 1:
        axes = [axes]
    for ax, fm in zip(axes, fms):
        sub = uplift_df[uplift_df.formation == fm].copy()
        nlog = sub[sub.source == "nlog"].sort_values("depth_bin_centre")
        rvg = sub[sub.source == "rvg"].sort_values("depth_bin_centre")
        ax.barh(nlog.depth_bin_centre, nlog.n_wells, height=8,
                color="C0", alpha=0.7, label="NLOG only")
        ax.barh(rvg.depth_bin_centre, rvg.n_wells, height=8,
                left=nlog.n_wells.values,
                color="C1", alpha=0.7, label="RVG (would add)")
        ax.axvline(30, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("# wells in bin")
        ax.set_title(f"Formation {fm}")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Depth below NAP (m)")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-formation per-10m-bin well counts: NLOG vs NLOG+RVG",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ------------------------------------------------------ top-depth distribution

def top_depth_shift(nlog_df: pd.DataFrame, rvg_strat: pd.DataFrame
                    ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare per-well top depth of each formation, NLOG vs RVG.

    Returns (table_of_quartiles, KS-distance dict).
    """
    rows = []
    ks = {}
    for fm in ["NU", "NM", "NL"]:
        nlog_tops = (nlog_df[nlog_df.formation == fm]
                     .groupby("borehole")["depth"].min().dropna())
        rvg_tops = (rvg_strat[rvg_strat.label == fm]
                    .groupby("borehole")["top_depth_below_nap"].min().dropna())
        rows.append({
            "formation": fm,
            "nlog_n": len(nlog_tops),
            "nlog_median": float(nlog_tops.median()) if len(nlog_tops) else np.nan,
            "nlog_p25": float(nlog_tops.quantile(0.25)) if len(nlog_tops) else np.nan,
            "nlog_p75": float(nlog_tops.quantile(0.75)) if len(nlog_tops) else np.nan,
            "rvg_n": len(rvg_tops),
            "rvg_median": float(rvg_tops.median()) if len(rvg_tops) else np.nan,
            "rvg_p25": float(rvg_tops.quantile(0.25)) if len(rvg_tops) else np.nan,
            "rvg_p75": float(rvg_tops.quantile(0.75)) if len(rvg_tops) else np.nan,
        })
        if len(nlog_tops) > 5 and len(rvg_tops) > 5:
            ks[fm] = float(stats.ks_2samp(nlog_tops, rvg_tops).statistic)
    return pd.DataFrame(rows), ks


def plot_top_depth(nlog_df: pd.DataFrame, rvg_strat: pd.DataFrame,
                   ks: dict[str, float], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, fm in zip(axes, ["NU", "NM", "NL"]):
        nlog_tops = (nlog_df[nlog_df.formation == fm]
                     .groupby("borehole")["depth"].min().dropna())
        rvg_tops = (rvg_strat[rvg_strat.label == fm]
                    .groupby("borehole")["top_depth_below_nap"].min().dropna())
        bins = np.linspace(0, max(500, nlog_tops.quantile(0.95)
                                   if len(nlog_tops) else 500), 40)
        if len(nlog_tops):
            ax.hist(nlog_tops.values, bins=bins, density=True,
                    color="C0", alpha=0.55,
                    label=f"NLOG (n={len(nlog_tops)})")
        if len(rvg_tops):
            ax.hist(rvg_tops.values, bins=bins, density=True,
                    color="C1", alpha=0.55,
                    label=f"RVG (n={len(rvg_tops)})")
        ax.set_xlabel("Top depth below NAP (m)")
        ax.set_ylabel("density")
        title = f"{fm} top-depth distribution"
        if fm in ks:
            title += f" [KS={ks[fm]:.2f}]"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ------------------------------------------------------- vocabulary alignment

def vocab_alignment(rvg_litho: pd.DataFrame) -> pd.DataFrame:
    """For each lithology code in RVG basisdata, count rows and
    report the mapped rock_type_fine (or 'unmapped')."""
    rows = []
    for code, n in rvg_litho._lith_raw.value_counts().items():
        mapped = RVG_LITHO_TO_FINE.get(code, "unmapped")
        rows.append({"rvg_litho_code": code, "n": int(n),
                     "rock_type_fine": mapped})
    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------- per-formation transitions

def transition_delta(nlog_df: pd.DataFrame, rvg_strat: pd.DataFrame,
                     rvg_litho: pd.DataFrame, max_depth: float = 100.0
                     ) -> pd.DataFrame:
    """Markov chain on rock_type_fine inside NU only (the formation
    where both NLOG and RVG have decent coverage in the upper 100 m).

    Compute the transition matrix from NLOG-only and from NLOG+RVG,
    then report Frobenius distance + matrix shapes.
    """
    rows = []
    # NLOG upper NU
    nlog_nu = (nlog_df[(nlog_df.formation == "NU")
                       & nlog_df.depth.between(0, max_depth)]
               .drop_duplicates(["borehole", "depth"])
               .sort_values(["borehole", "depth"])
               [["borehole", "depth", "rock_type_fine"]])
    # build sequence of (well, depth_bin) -> rock
    nlog_nu["bin"] = (nlog_nu.depth // 10).astype(int)
    nlog_seq = (nlog_nu.groupby(["borehole", "bin"]).rock_type_fine
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty
                     else None).reset_index().dropna())
    # transitions: pair within same borehole, consecutive bins
    nlog_pairs = nlog_seq.merge(nlog_seq, on="borehole", suffixes=("_a", "_b"))
    nlog_pairs = nlog_pairs[(nlog_pairs.bin_b - nlog_pairs.bin_a == 1)]
    nlog_trans = (nlog_pairs.groupby(["rock_type_fine_a", "rock_type_fine_b"])
                  .size().reset_index(name="n"))

    # RVG litho: only consider rows whose strat unit maps to NU
    rvg_nu_wells = set(rvg_strat[rvg_strat.label == "NU"].borehole.unique())
    rvg_nu_lit = rvg_litho[rvg_litho.borehole.isin(rvg_nu_wells)
                            & rvg_litho.label.notna()].copy()
    # Discretise rvg lithology intervals to 10m bins
    rvg_rows_bin = []
    for _, r in rvg_nu_lit.iterrows():
        lo, hi = r.top_depth_below_nap, r.bottom_depth_below_nap
        if lo < 0 or hi <= lo or hi > max_depth:
            continue
        for b in range(int(lo // 10), int(hi // 10) + 1):
            rvg_rows_bin.append({"borehole": r.borehole, "bin": b,
                                  "rock_type_fine": r.label})
    rvg_seq = pd.DataFrame(rvg_rows_bin).drop_duplicates(["borehole", "bin"])
    rvg_pairs = rvg_seq.merge(rvg_seq, on="borehole", suffixes=("_a", "_b"))
    rvg_pairs = rvg_pairs[(rvg_pairs.bin_b - rvg_pairs.bin_a == 1)]
    rvg_trans = (rvg_pairs.groupby(["rock_type_fine_a", "rock_type_fine_b"])
                 .size().reset_index(name="n"))

    # combined union of rocks
    all_rocks = sorted(set(nlog_trans.rock_type_fine_a)
                       | set(nlog_trans.rock_type_fine_b)
                       | set(rvg_trans.rock_type_fine_a)
                       | set(rvg_trans.rock_type_fine_b))
    idx = {r: i for i, r in enumerate(all_rocks)}
    n = len(all_rocks)

    def to_matrix(trans, n, idx):
        M = np.zeros((n, n))
        for _, r in trans.iterrows():
            M[idx[r.rock_type_fine_a], idx[r.rock_type_fine_b]] = r.n
        # row-normalise
        rsum = M.sum(axis=1, keepdims=True)
        rsum[rsum == 0] = 1.0
        return M / rsum

    M_nlog = to_matrix(nlog_trans, n, idx)
    M_combined = to_matrix(
        pd.concat([nlog_trans, rvg_trans]).groupby(
            ["rock_type_fine_a", "rock_type_fine_b"]).n.sum().reset_index(),
        n, idx)
    frob = float(np.linalg.norm(M_nlog - M_combined, ord="fro"))
    return frob, all_rocks


# -------------------------------------------------------------- report writer

def _md_table(df: pd.DataFrame, float_fmt: str = "{:.1f}") -> str:
    """Tiny markdown-table writer that does not need tabulate."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    out = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not np.isnan(v):
                cells.append(float_fmt.format(v))
            else:
                cells.append("" if (isinstance(v, float) and np.isnan(v))
                             else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(out_path: Path,
                 uplift_df: pd.DataFrame,
                 top_table: pd.DataFrame, ks: dict[str, float],
                 vocab_df: pd.DataFrame,
                 frob_nu: float, rocks_nu: list[str]) -> None:
    lines = [
        "# RVG gap analysis -- does the Garcon dataset improve our pipeline?",
        "",
        "*Generated by `scripts/diagnostics/rvg_gap_analysis.py`.*",
        "",
        "Question: would adding the 1,394 Roer Valley Graben (RVG) boreholes",
        "from Garzón et al. (2026) materially improve the simulator's",
        "shallow stratigraphy (NU/NM/NL = Upper/Middle/Lower North Sea Group)?",
        "",
        "All depths are in metres below NAP (positive going down).",
        "",
        "## 1. Per-formation per-10m-bin coverage uplift (0--200m)",
        "",
        "Number of unique wells per formation per 10m depth bin, NLOG vs",
        "what RVG would add.  KDE threshold ~30 samples shown as grey",
        "vertical line on the figure.",
        "",
        "![coverage uplift](../garcon/coverage_uplift.png)",
        "",
        "Headline:",
    ]
    for fm in ["NU", "NM", "NL"]:
        nlog_total = uplift_df[(uplift_df.formation == fm)
                                & (uplift_df.source == "nlog")].n_wells.sum()
        rvg_total = uplift_df[(uplift_df.formation == fm)
                               & (uplift_df.source == "rvg")].n_wells.sum()
        nlog_bins_30 = (uplift_df[(uplift_df.formation == fm)
                                   & (uplift_df.source == "nlog")
                                   & (uplift_df.n_wells >= 30)]
                        .depth_bin_centre.nunique())
        combined = (uplift_df[(uplift_df.formation == fm)
                               & (uplift_df.source.isin(["nlog", "rvg"]))]
                    .groupby("depth_bin_centre").n_wells.sum())
        combined_bins_30 = int((combined >= 30).sum())
        lines.append(
            f"- **{fm}**: NLOG places {nlog_total} well-bins in 0--200m; "
            f"RVG would add {rvg_total}. "
            f"Bins above KDE threshold ($\\geq$30 wells): "
            f"{nlog_bins_30} -> {combined_bins_30}."
        )
    lines += [
        "",
        "## 2. Top-depth distribution shift",
        "",
        "Per-well top-depth distribution for each formation, NLOG vs RVG,",
        "with the Kolmogorov-Smirnov distance between them:",
        "",
        "![top-depth shift](../garcon/top_depth_shift.png)",
        "",
        _md_table(top_table, "{:.1f}"),
        "",
    ]
    if ks:
        ks_lines = ", ".join(f"{fm} KS={v:.2f}" for fm, v in ks.items())
        lines.append(f"KS distances: {ks_lines}.")
    lines += [
        "",
        "## 3. Lithology vocabulary alignment",
        "",
        f"RVG basisdata has {vocab_df.n.sum():,} lithology intervals across",
        f"{len(vocab_df)} distinct codes.  Mapping to our `rock_type_fine`",
        "vocabulary:",
        "",
        _md_table(vocab_df.head(30), "{:.0f}"),
        "",
        f"Total mapped: {vocab_df[vocab_df.rock_type_fine != 'unmapped'].n.sum():,} "
        f"({100 * vocab_df[vocab_df.rock_type_fine != 'unmapped'].n.sum() / vocab_df.n.sum():.1f}%).",
        f"Unmapped: {vocab_df[vocab_df.rock_type_fine == 'unmapped'].n.sum():,}.",
        "",
        "## 4. Transition-matrix change in NU upper 100m",
        "",
        f"Frobenius distance between the NLOG-only and NLOG+RVG Markov chain",
        f"on `rock_type_fine` in formation NU, depths 0--100m, "
        f"on rocks {rocks_nu}: **{frob_nu:.3f}**.",
        "",
        f"For reference, the v3 simulator's worst per-formation Frobenius",
        f"distance against real data is around 1.0 (RB/RO before the calibration",
        f"fix).  Anything above 0.3 here would be a meaningful change.",
        "",
        "## Recommendation",
        "",
    ]
    # automated recommendation
    nu_uplift = uplift_df[(uplift_df.formation == "NU")
                           & (uplift_df.source == "rvg")].n_wells.sum()
    nm_uplift = uplift_df[(uplift_df.formation == "NM")
                           & (uplift_df.source == "rvg")].n_wells.sum()
    nl_uplift = uplift_df[(uplift_df.formation == "NL")
                           & (uplift_df.source == "rvg")].n_wells.sum()
    vocab_pct = (vocab_df[vocab_df.rock_type_fine != "unmapped"].n.sum()
                 / vocab_df.n.sum())
    ks_avg = np.mean(list(ks.values())) if ks else 0.0
    big_gap = (nm_uplift + nl_uplift > 200) or any(v > 0.30 for v in ks.values())
    if big_gap and vocab_pct > 0.85 and frob_nu > 0.30:
        verdict = ("**PROCEED with integration**: RVG fills a clear shallow "
                   "coverage gap (especially in NM and NL), the lithology "
                   "vocabulary aligns cleanly, and the transition matrix "
                   "shifts measurably.")
    elif big_gap and vocab_pct > 0.85:
        verdict = ("**PARTIAL integration**: RVG fills shallow gaps in NM/NL "
                   "and the vocabulary aligns. The transition-matrix change "
                   "is small, so consider adding RVG to the top-depth KDEs "
                   "only, leaving the Markov chains alone.")
    elif big_gap:
        verdict = ("**HOLD**: there is a real shallow data gap, but the "
                   "lithology vocabulary does not align cleanly enough for "
                   "automatic integration. Decide whether the gain is worth "
                   "manual mapping work.")
    else:
        verdict = ("**NOT WORTH integrating**: NLOG already covers the upper "
                   "section well enough, and RVG would not change the "
                   "shallow stratigraphy by a noticeable amount.")
    lines += [verdict, ""]
    lines += [
        "## Notes",
        "",
        "- RVG covers the southern Netherlands (Roer Valley Graben). "
        "Adding it tilts the simulator toward this basin's stratigraphy. "
        "That is desirable if the simulator is meant to represent Dutch "
        "subsurface variability; less so if the user expects a "
        "North-Netherlands-only generator.",
        "- RVG provides no wireline channels.  Any integration affects "
        "Formation Geometry only, not the Distribution Bank.",
        "- The Discovery Prior is untouched (no hydrocarbon labels in RVG).",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path,
                   default=Path("data/clean/samples.parquet"))
    p.add_argument("--garcon-dir", type=Path,
                   default=Path("data/garcon"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("plots/garcon"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path("plots/analysis").mkdir(parents=True, exist_ok=True)

    print("loading NLOG samples...")
    nlog_df = pd.read_parquet(args.samples)
    nlog_df = nlog_df[nlog_df.dataset == "NLOG"].copy()
    print(f"  NLOG: {len(nlog_df):,} rows, {nlog_df.borehole.nunique()} wells")

    print("loading RVG / Garcon data...")
    rvg_strat, rvg_litho = load_rvg(args.garcon_dir)
    print(f"  RVG strat (DGM-plus): {len(rvg_strat):,} intervals, "
          f"{rvg_strat.borehole.nunique()} wells")
    print(f"  RVG litho (basisdata): {len(rvg_litho):,} intervals, "
          f"{rvg_litho.borehole.nunique()} wells")

    print("\n[1] coverage uplift...")
    uplift = coverage_uplift(nlog_df, rvg_strat)
    plot_coverage(uplift, args.out_dir / "coverage_uplift.png")

    print("[2] top-depth distribution shift...")
    top_table, ks = top_depth_shift(nlog_df, rvg_strat)
    plot_top_depth(nlog_df, rvg_strat, ks, args.out_dir / "top_depth_shift.png")

    print("[3] vocabulary alignment...")
    vocab = vocab_alignment(rvg_litho)

    print("[4] transition matrix delta in NU...")
    frob_nu, rocks_nu = transition_delta(nlog_df, rvg_strat, rvg_litho)
    print(f"  Frobenius delta = {frob_nu:.3f}")

    # write report
    report_path = Path("plots/analysis/rvg_gap_report.md")
    write_report(report_path, uplift, top_table, ks, vocab, frob_nu, rocks_nu)

    # raw CSV (for plot reproducibility)
    uplift.to_csv("plots/analysis/rvg_gap_coverage.csv", index=False)
    top_table.to_csv("plots/analysis/rvg_gap_top_depths.csv", index=False)
    vocab.to_csv("plots/analysis/rvg_gap_vocab.csv", index=False)
    print(f"\ndone -- report at {report_path}")


if __name__ == "__main__":
    main()
