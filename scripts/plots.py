from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORMATION_COLOURS = {
    "NU": "#fdbb84", "CK": "#a6d96a", "KN": "#8073ac", "RN": "#e31a1c",
    "RB": "#fb9a99", "ZE": "#ff7f00", "RO": "#b15928",
    "NM": "#fddbc7", "NL": "#fee08b", "DC": "#b2abd2",
    "AT": "#67a9cf", "SL": "#9467bd", "SG": "#ec7014", "SK": "#c7e9b4",
}
ROCK_COLOURS = {
    "chalk":       "#a6d96a", "sandstone":   "#b15928", "claystone":   "#8073ac",
    "clay":        "#ef8a62", "mudstone":    "#b2abd2", "halite":      "#fdbf6f",
    "anhydrite":   "#ff7f00", "siltstone":   "#fdae61", "limestone":   "#33a02c",
    "nanno_ooze":  "#67a9cf", "diatom_ooze": "#2166ac", "basalt":      "#4d4d4d",
    "other":       "#bbbbbb",
}
DATASET_COLOURS = {"LILY": "#2166ac", "NLOG": "#b2182b"}
NLOG_FORMATIONS = ["NU", "CK", "KN", "RN", "RB", "ZE", "RO"]

DISPLAY_BOUNDS = {
    "rhob": (1.5, 3.2), "gr_api": (0, 200), "ngr_cps": (0, 80),
    "nphi": (-0.05, 0.6), "dt_us_ft": (40, 220), "pef": (1, 10),
    "cali_in": (4, 20), "sp_mv": (-200, 200),
    "res_deep_log": (-1, 4), "res_shal_log": (-1, 4),
    "drho": (-0.2, 0.2), "msus_si": (1e-6, 1e-2),
    "phie": (0, 0.45), "vsh": (0, 1), "vcl": (0, 1),
    "sw": (0, 1),
}
MEASUREMENT_LABELS = {
    "rhob": "bulk density (g/cc)", "gr_api": "gamma ray (API)",
    "ngr_cps": "NGR (cps)", "nphi": "neutron porosity (v/v)",
    "dt_us_ft": "slowness (µs/ft)", "pef": "PEF (barns/e)",
    "cali_in": "caliper (in)", "sp_mv": "SP (mV)",
    "res_deep_log": "log₁₀ deep resistivity (Ω·m)",
    "res_shal_log": "log₁₀ shallow resistivity (Ω·m)",
    "drho": "density correction (g/cc)", "msus_si": "magn. susceptibility (SI)",
    "phie": "effective porosity", "vsh": "VSH", "vcl": "VCL",
    "sw": "water saturation",
}
MIN_SAMPLES_FOR_PLOT = 50



def _subsample(arr: np.ndarray, n_max: int, seed: int = 42) -> np.ndarray:
    if len(arr) <= n_max:
        return arr
    return np.random.default_rng(seed).choice(arr, n_max, replace=False)


def _overlay_hist(ax, series_by_label, xlim, xlabel, bins=55,
                  colour_map=None, log_x=False):
    bins_arr = (np.logspace(np.log10(max(xlim[0], 1e-8)), np.log10(xlim[1]), bins)
                if log_x else np.linspace(xlim[0], xlim[1], bins))
    for label, vals in series_by_label.items():
        vals = np.asarray(vals)
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) < MIN_SAMPLES_FOR_PLOT:
            continue
        c = (colour_map or {}).get(label, None)
        ax.hist(vals, bins=bins_arr, density=True, histtype="step",
                linewidth=1.7, color=c, label=f"{label} (n={len(vals):,})")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_xlim(xlim)
    if log_x:
        ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def _pick_shared_rock_types(df: pd.DataFrame, measurement: str,
                            min_samples: int = MIN_SAMPLES_FOR_PLOT) -> list[str]:
    d = df[df["measurement"] == measurement]
    per = d.groupby(["rock_type", "dataset"])["value"].size().unstack(fill_value=0)
    mask = (per.get("LILY", 0) >= min_samples) & (per.get("NLOG", 0) >= min_samples)
    return per[mask].index.tolist()


def _depth_overlap(df: pd.DataFrame, rock_type: str, measurement: str
                   ) -> tuple[float, float] | None:
    """Return (depth_lo, depth_hi) where both datasets have samples for
    the given rock type and measurement, or None if there is no overlap."""
    sub = df[(df["rock_type"] == rock_type) & (df["measurement"] == measurement)]
    lily = sub.loc[sub["dataset"] == "LILY", "depth"]
    nlog = sub.loc[sub["dataset"] == "NLOG", "depth"]
    if len(lily) < MIN_SAMPLES_FOR_PLOT or len(nlog) < MIN_SAMPLES_FOR_PLOT:
        return None
    # Use p05-p95 of each to avoid extreme outliers anchoring the overlap
    lo = max(lily.quantile(0.05), nlog.quantile(0.05))
    hi = min(lily.quantile(0.95), nlog.quantile(0.95))
    if hi - lo < 100:   # need at least 100 m of overlap
        return None
    return float(lo), float(hi)


def _load_wide_nlog(df: pd.DataFrame, measurements: list[str]) -> pd.DataFrame:
    """Pivot to (borehole, depth, formation) × measurement (NLOG only)."""
    sub = df[(df["dataset"] == "NLOG") & (df["measurement"].isin(measurements))]
    wide = (sub.pivot_table(index=["borehole", "depth", "formation"],
                             columns="measurement", values="value",
                             aggfunc="first")
                .reset_index())
    return wide.dropna(subset=measurements)



def plot_feature_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of rows and wells per measurement, by dataset."""
    agg = (df.groupby(["dataset", "measurement"])
             .agg(rows=("value", "size"), wells=("borehole", "nunique"))
             .reset_index())
    measurements = (agg.groupby("measurement")["rows"].sum()
                       .sort_values().index.tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(measurements))))
    for ax, col, xlabel in [(ax1, "rows", "rows"), (ax2, "wells", "wells")]:
        left = np.zeros(len(measurements))
        for ds in ["NLOG", "LILY"]:
            sub = agg[agg["dataset"] == ds].set_index("measurement").reindex(measurements)
            ax.barh(range(len(measurements)), sub[col].fillna(0), left=left,
                    color=DATASET_COLOURS[ds], alpha=0.75, label=ds,
                    edgecolor="black", linewidth=0.3)
            left += sub[col].fillna(0).values
        ax.set_yticks(range(len(measurements)))
        ax.set_yticklabels(measurements)
        ax.set_xlabel(xlabel)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Feature coverage — rows and wells per measurement",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rock_type_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar — rows per rock_type, split by dataset."""
    agg = (df.groupby(["rock_type", "dataset"])["value"].size()
             .unstack(fill_value=0))
    agg = agg.assign(total=agg.sum(axis=1)).sort_values("total")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(agg))))
    left = np.zeros(len(agg))
    for ds in ["NLOG", "LILY"]:
        if ds in agg.columns:
            ax.barh(range(len(agg)), agg[ds].values, left=left,
                    color=DATASET_COLOURS[ds], alpha=0.75, label=ds,
                    edgecolor="black", linewidth=0.3)
            left += agg[ds].values
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index)
    ax.set_xlabel("rows (log scale)")
    ax.set_xscale("log")
    ax.set_title("Samples per rock type", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_depth_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Depth histograms LILY vs NLOG side by side, with overlap zone shaded."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ["LILY", "NLOG"]):
        d = df.loc[df["dataset"] == ds, "depth"]
        if len(d) == 0:
            continue
        bins = np.linspace(0, float(d.quantile(0.99)), 60)
        ax.hist(d, bins=bins, color=DATASET_COLOURS[ds], alpha=0.8,
                edgecolor="black", linewidth=0.3)
        ax.set_title(f"{ds} — depth distribution (n={len(d):,})",
                     fontweight="bold")
        ax.set_xlabel("depth (m)")
        ax.set_ylabel("samples")
        ax.grid(alpha=0.25)
    # shade the rough overlap zone (LILY upper, NLOG lower) on both panels
    lily_max = df.loc[df["dataset"] == "LILY", "depth"].quantile(0.95)
    nlog_min = df.loc[df["dataset"] == "NLOG", "depth"].quantile(0.05)
    for ax in axes:
        if nlog_min < lily_max:
            ax.axvspan(nlog_min, lily_max, alpha=0.15, color="purple",
                       label=f"overlap {nlog_min:.0f}-{lily_max:.0f} m")
            ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")



def plot_nlog_by_formation(df: pd.DataFrame, out_path: Path) -> None:
    """4×2 grid: each panel is one curve, overlaid lines per formation."""
    nlog = df[df["dataset"] == "NLOG"]
    panels = [
        ("rhob", "RHOB"), ("gr_api", "GR"),
        ("nphi", "NPHI"), ("dt_us_ft", "DT"),
        ("pef", "PEF"),   ("sp_mv", "SP"),
        ("res_deep_log", "deep resistivity"), ("cali_in", "caliper"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    for ax, (meas, short) in zip(axes, panels):
        d = nlog[nlog["measurement"] == meas]
        if len(d) == 0:
            ax.axis("off"); continue
        xlim = DISPLAY_BOUNDS.get(meas, (d["value"].min(), d["value"].max()))
        series = {}
        for f in NLOG_FORMATIONS:
            vals = d.loc[d["formation"] == f, "value"].values
            if len(vals) > 100_000:
                vals = _subsample(vals, 100_000)
            if len(vals):
                series[f] = vals
        _overlay_hist(ax, series, xlim, MEASUREMENT_LABELS.get(meas, meas),
                      colour_map=FORMATION_COLOURS)
        ax.set_title(short, fontweight="bold", fontsize=11)
    fig.suptitle("NLOG — P(log | formation)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_lily_by_lithology(df: pd.DataFrame, out_path: Path, top_n: int = 8) -> None:
    """2×2 grid: one LILY measurement per panel, overlaid per top-N rock type."""
    lily = df[df["dataset"] == "LILY"]
    panels = [("rhob", "bulk density"), ("ngr_cps", "NGR"),
              ("dt_us_ft", "slowness"), ("msus_si", "magn. susceptibility")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, (meas, name) in zip(axes, panels):
        d = lily[lily["measurement"] == meas]
        if len(d) == 0:
            ax.axis("off"); continue
        xlim = DISPLAY_BOUNDS.get(meas, (d["value"].min(), d["value"].max()))
        top_types = d["rock_type"].value_counts().head(top_n).index.tolist()
        series = {rt: d.loc[d["rock_type"] == rt, "value"].values
                  for rt in top_types}
        log_x = (meas == "msus_si")
        _overlay_hist(ax, series, xlim, MEASUREMENT_LABELS.get(meas, meas),
                      colour_map=ROCK_COLOURS, log_x=log_x)
        ax.set_title(name, fontweight="bold", fontsize=11)
    fig.suptitle("LILY — P(measurement | rock type)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_feature_violin(df: pd.DataFrame, out_path: Path,
                            measurement: str = "rhob") -> None:
    """Violin plot of one measurement across rock types × datasets."""
    d = df[df["measurement"] == measurement]
    if len(d) == 0:
        print(f"  [skip] no data for {measurement}"); return
    order = d["rock_type"].value_counts().index.tolist()[:10]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions, labels = [], []
    pos = 0.0
    for rt in order:
        for ds in ["LILY", "NLOG"]:
            vals = d.loc[(d["rock_type"] == rt) & (d["dataset"] == ds), "value"].values
            if len(vals) < MIN_SAMPLES_FOR_PLOT:
                continue
            positions.append(pos)
            labels.append(f"{rt}\n{ds}")
            vals_sub = _subsample(vals, 30_000)
            parts = ax.violinplot([vals_sub], positions=[pos], widths=0.75,
                                   showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(DATASET_COLOURS[ds])
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
            pos += 1.0
        pos += 0.5
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(MEASUREMENT_LABELS.get(measurement, measurement))
    ax.set_title(f"{measurement} by rock type — LILY (blue) vs NLOG (red)",
                 fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")



def plot_lily_vs_nlog_native(df: pd.DataFrame, out_path: Path) -> None:
    """Native-unit overlay: density + sonic per shared rock type.
    No depth restriction — shows the full distributions as-is."""
    shared = _pick_shared_rock_types(df, "rhob")
    if not shared:
        print("  [skip] no shared rock types"); return

    fig, axes = plt.subplots(2, len(shared),
                              figsize=(3.3 * len(shared), 8),
                              sharex="row", sharey="row", squeeze=False)
    for j, rt in enumerate(shared):
        for i, (meas, xlim) in enumerate([("rhob", (1.0, 3.2)),
                                           ("dt_us_ft", (40, 220))]):
            ax = axes[i, j]
            bins = np.linspace(*xlim, 60)
            for ds, col in DATASET_COLOURS.items():
                vals = df.loc[(df["rock_type"] == rt) &
                              (df["dataset"] == ds) &
                              (df["measurement"] == meas), "value"].values
                vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
                if len(vals) < MIN_SAMPLES_FOR_PLOT:
                    continue
                vals = _subsample(vals, 50_000)
                ax.hist(vals, bins=bins, density=True, histtype="step",
                        linewidth=1.7, color=col,
                        label=f"{ds} (n={len(vals):,})")
            if i == 0: ax.set_title(rt, fontweight="bold")
            if i == 1: ax.set_xlabel(MEASUREMENT_LABELS[meas])
            if j == 0: ax.set_ylabel("density")
            ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle("LILY vs NLOG — shared rock types, all depths",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_lily_vs_nlog_zscore(df: pd.DataFrame, out_path: Path) -> None:
    """Gamma cannot be pooled in absolute units; compare shapes via z-score."""
    d_lily = df[(df["dataset"] == "LILY") & (df["measurement"] == "ngr_cps")]
    d_nlog = df[(df["dataset"] == "NLOG") & (df["measurement"] == "gr_api")]
    shared = [rt for rt in d_lily["rock_type"].unique()
              if rt in d_nlog["rock_type"].unique()
              and len(d_lily[d_lily["rock_type"] == rt]) >= MIN_SAMPLES_FOR_PLOT
              and len(d_nlog[d_nlog["rock_type"] == rt]) >= MIN_SAMPLES_FOR_PLOT]
    if not shared:
        print("  [skip] no shared rock types for gamma"); return

    lily_mean, lily_std = d_lily["value"].mean(), d_lily["value"].std()
    nlog_mean, nlog_std = d_nlog["value"].mean(), d_nlog["value"].std()

    fig, axes = plt.subplots(1, len(shared),
                              figsize=(3.2 * len(shared), 4.2),
                              sharey=True, squeeze=False)
    bins = np.linspace(-2.5, 5, 50)
    for j, rt in enumerate(shared):
        ax = axes[0, j]
        lv = d_lily.loc[d_lily["rock_type"] == rt, "value"].values
        nv = d_nlog.loc[d_nlog["rock_type"] == rt, "value"].values
        if len(nv) > 50_000:
            nv = _subsample(nv, 50_000)
        lz = (lv - lily_mean) / lily_std
        nz = (nv - nlog_mean) / nlog_std
        ax.hist(lz, bins=bins, density=True, histtype="step", linewidth=1.7,
                color=DATASET_COLOURS["LILY"], label=f"LILY NGR (n={len(lz):,})")
        ax.hist(nz, bins=bins, density=True, histtype="step", linewidth=1.7,
                color=DATASET_COLOURS["NLOG"], label=f"NLOG GR  (n={len(nz):,})")
        ax.axvline(0, color="grey", linewidth=0.4)
        ax.set_title(rt, fontweight="bold")
        ax.set_xlabel("z-score within dataset")
        if j == 0: ax.set_ylabel("density")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle("LILY NGR (cps) vs NLOG GR (API) — z-scored shape comparison",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_compaction_trends(df: pd.DataFrame, out_path: Path) -> None:
    """Median±IQR curves of RHOB and DT vs depth per shared rock type."""
    rock_types = _pick_shared_rock_types(df, "rhob")
    if not rock_types:
        print("  [skip] no shared rock types"); return
    depth_bins = [0, 100, 300, 800, 1500, 2500, 3500, 5000]

    fig, axes = plt.subplots(2, len(rock_types),
                              figsize=(3.3 * len(rock_types), 9),
                              sharey="row", squeeze=False)
    for j, rt in enumerate(rock_types):
        for i, meas in enumerate(["rhob", "dt_us_ft"]):
            ax = axes[i, j]
            d = df[(df["rock_type"] == rt) & (df["measurement"] == meas)].copy()
            if len(d) == 0:
                continue
            d["bin"] = pd.cut(d["depth"], bins=depth_bins, include_lowest=True)
            for ds, col in DATASET_COLOURS.items():
                dd = d[d["dataset"] == ds]
                if len(dd) < MIN_SAMPLES_FOR_PLOT:
                    continue
                stats = (dd.groupby("bin", observed=True)
                           .agg(p25=("value", lambda x: x.quantile(0.25)),
                                p50=("value", "median"),
                                p75=("value", lambda x: x.quantile(0.75)),
                                depth_median=("depth", "median"),
                                n=("value", "size")).reset_index())
                stats = stats[stats["n"] >= 20]
                if len(stats) == 0:
                    continue
                ax.fill_betweenx(stats["depth_median"], stats["p25"], stats["p75"],
                                 alpha=0.25, color=col)
                ax.plot(stats["p50"], stats["depth_median"], "o-",
                        color=col, linewidth=2, markersize=4, label=ds)
            if i == 0: ax.set_title(rt, fontweight="bold")
            ax.set_xlabel(MEASUREMENT_LABELS[meas])
            if j == 0: ax.set_ylabel("depth (m)")
            ax.invert_yaxis(); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.suptitle("Compaction trends — median (line) + IQR (band) per depth bin",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_depth_matched_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """One figure per measurement: grid of (rock_type × depth_bin) panels,
    each panel overlaying LILY and NLOG restricted to that bin.

    Uses the same binning as the depth-matched CSV table so plot and
    table line up exactly.  Panels where either dataset has <30 samples
    are shown empty.  This is the clean 'same rock type AND same depth'
    comparison: any visible disagreement inside a panel is NOT explained
    by burial depth alone.

    Writes one PNG; both RHOB and DT are stacked (RHOB rows on top)."""
    depth_bins = [0, 100, 300, 800, 1500, 2500, 3500, 5000]
    bin_labels = [f"{a}-{b} m" for a, b in zip(depth_bins[:-1], depth_bins[1:])]

    shared = _pick_shared_rock_types(df, "rhob")
    if not shared:
        print("  [skip] no shared rock types"); return

    # figure: rows = (RHOB rows) + (DT rows), cols = depth bins
    n_rt = len(shared)
    n_bins = len(bin_labels)
    # Arrange: for each measurement, n_rt rows × n_bins cols
    # Stack vertically: RHOB block on top, DT block on bottom
    fig, axes = plt.subplots(2 * n_rt, n_bins,
                              figsize=(2.4 * n_bins, 1.9 * 2 * n_rt),
                              squeeze=False, sharex="row")

    d = df.copy()
    d["depth_bin"] = pd.cut(d["depth"], bins=depth_bins,
                             labels=bin_labels, include_lowest=True)

    for mi, (meas, xlim) in enumerate([("rhob", (1.0, 3.2)),
                                        ("dt_us_ft", (40, 220))]):
        for ri, rt in enumerate(shared):
            row = mi * n_rt + ri
            for ci, dbin in enumerate(bin_labels):
                ax = axes[row, ci]
                sub = d[(d["rock_type"] == rt) &
                        (d["measurement"] == meas) &
                        (d["depth_bin"] == dbin)]
                bins_arr = np.linspace(*xlim, 40)
                has_both = True
                sample_counts = {}
                for ds, col in DATASET_COLOURS.items():
                    vals = sub.loc[sub["dataset"] == ds, "value"].values
                    vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
                    sample_counts[ds] = len(vals)
                    if len(vals) < 30:
                        has_both = False
                        continue
                    vals = _subsample(vals, 20_000)
                    ax.hist(vals, bins=bins_arr, density=True,
                            histtype="step", linewidth=1.4, color=col)
                # panel decoration
                if not has_both:
                    ax.set_facecolor("#f5f5f5")
                    ax.text(0.5, 0.5,
                             f"L={sample_counts.get('LILY', 0)}\n"
                             f"N={sample_counts.get('NLOG', 0)}",
                             transform=ax.transAxes, ha="center", va="center",
                             fontsize=7, color="grey")
                else:
                    ax.text(0.97, 0.95,
                            f"L={sample_counts['LILY']}\nN={sample_counts['NLOG']}",
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=6)
                ax.set_xlim(xlim)
                ax.set_xticks([])
                ax.set_yticks([])
                if ri == 0 and mi == 0:
                    ax.set_title(dbin, fontsize=9, fontweight="bold")
                if ci == 0:
                    meas_short = "RHOB" if meas == "rhob" else "DT"
                    ax.set_ylabel(f"{rt}\n{meas_short}",
                                   rotation=0, labelpad=30,
                                   fontsize=8, fontweight="bold",
                                   va="center", ha="right")
            # restore x-ticks on the bottom row of each measurement-block
            if ri == n_rt - 1:
                for ci in range(n_bins):
                    axes[row, ci].set_xticks(
                        np.linspace(xlim[0], xlim[1], 4))
                    axes[row, ci].tick_params(labelsize=7)

    # legend proxy at the very top
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=DATASET_COLOURS["LILY"], linewidth=2, label="LILY"),
        Line2D([0], [0], color=DATASET_COLOURS["NLOG"], linewidth=2, label="NLOG"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.995), fontsize=9)

    fig.suptitle("LILY vs NLOG — matched rock type × depth bin\n"
                 f"top block: RHOB (g/cc)     bottom block: DT (µs/ft)\n"
                 "L / N = LILY / NLOG sample counts per panel",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_depth_matched_gamma(df: pd.DataFrame, out_path: Path) -> None:
    """Same logic as plot_depth_matched_comparison but for gamma (z-scored).
    Grid of (rock_type × depth_bin) panels with LILY NGR and NLOG GR
    z-scored within each dataset and overlaid per bin."""
    depth_bins = [0, 100, 300, 800, 1500, 2500, 3500, 5000]
    bin_labels = [f"{a}-{b} m" for a, b in zip(depth_bins[:-1], depth_bins[1:])]

    d_lily = df[(df["dataset"] == "LILY") & (df["measurement"] == "ngr_cps")].copy()
    d_nlog = df[(df["dataset"] == "NLOG") & (df["measurement"] == "gr_api")].copy()
    if len(d_lily) == 0 or len(d_nlog) == 0:
        print("  [skip] gamma data missing"); return

    d_lily["depth_bin"] = pd.cut(d_lily["depth"], bins=depth_bins,
                                   labels=bin_labels, include_lowest=True)
    d_nlog["depth_bin"] = pd.cut(d_nlog["depth"], bins=depth_bins,
                                   labels=bin_labels, include_lowest=True)

    lily_mean, lily_std = d_lily["value"].mean(), d_lily["value"].std()
    nlog_mean, nlog_std = d_nlog["value"].mean(), d_nlog["value"].std()

    shared = [rt for rt in d_lily["rock_type"].unique()
              if rt in d_nlog["rock_type"].unique()
              and len(d_lily[d_lily["rock_type"] == rt]) >= 100
              and len(d_nlog[d_nlog["rock_type"] == rt]) >= 100]
    if not shared:
        print("  [skip] no shared rock types for gamma"); return

    n_rt = len(shared)
    n_bins = len(bin_labels)
    fig, axes = plt.subplots(n_rt, n_bins,
                              figsize=(2.4 * n_bins, 1.9 * n_rt),
                              squeeze=False, sharex=True, sharey="row")

    bins_arr = np.linspace(-2.5, 5, 30)
    for ri, rt in enumerate(shared):
        for ci, dbin in enumerate(bin_labels):
            ax = axes[ri, ci]
            lv = d_lily.loc[(d_lily["rock_type"] == rt) &
                            (d_lily["depth_bin"] == dbin), "value"].values
            nv = d_nlog.loc[(d_nlog["rock_type"] == rt) &
                            (d_nlog["depth_bin"] == dbin), "value"].values
            if len(lv) >= 30 and len(nv) >= 30:
                if len(nv) > 20_000:
                    nv = _subsample(nv, 20_000)
                lz = (lv - lily_mean) / lily_std
                nz = (nv - nlog_mean) / nlog_std
                ax.hist(lz, bins=bins_arr, density=True, histtype="step",
                        linewidth=1.4, color=DATASET_COLOURS["LILY"])
                ax.hist(nz, bins=bins_arr, density=True, histtype="step",
                        linewidth=1.4, color=DATASET_COLOURS["NLOG"])
                ax.axvline(0, color="grey", linewidth=0.3)
                ax.text(0.97, 0.95, f"L={len(lv)}\nN={len(nv)}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=6)
            else:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, f"L={len(lv)}\nN={len(nv)}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color="grey")
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(dbin, fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(rt, rotation=0, labelpad=30,
                              fontsize=9, fontweight="bold",
                              va="center", ha="right")
        if ri == n_rt - 1:
            for ci in range(n_bins):
                axes[ri, ci].set_xticks([-2, 0, 2, 4])
                axes[ri, ci].tick_params(labelsize=7)
                axes[ri, ci].set_xlabel("z-score", fontsize=7)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=DATASET_COLOURS["LILY"], linewidth=2, label="LILY NGR"),
        Line2D([0], [0], color=DATASET_COLOURS["NLOG"], linewidth=2, label="NLOG GR"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9)

    fig.suptitle("Gamma — matched rock type × depth bin, z-scored within dataset\n"
                 "L / N = LILY / NLOG sample counts per panel",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")



def plot_gardner(df: pd.DataFrame, out_path: Path) -> None:
    """RHOB vs Vp per formation with Gardner's curve.
    ρ = 0.23 × Vp^0.25 (Vp in ft/s)."""
    wide = _load_wide_nlog(df, ["rhob", "dt_us_ft"])
    if len(wide) == 0:
        print("  [skip] no paired RHOB+DT"); return
    wide["vp_m_s"] = 304_800.0 / wide["dt_us_ft"]

    v_grid = np.linspace(1500, 6500, 200)
    gardner = 0.23 * (v_grid * 3.28084) ** 0.25

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, form in zip(axes, NLOG_FORMATIONS):
        d = wide[wide["formation"] == form]
        if len(d) == 0:
            ax.axis("off"); continue
        if len(d) > 15_000:
            d = d.sample(15_000, random_state=42)
        ax.scatter(d["vp_m_s"], d["rhob"], s=2, alpha=0.15,
                   c=FORMATION_COLOURS.get(form, "#999"), edgecolor="none")
        ax.plot(v_grid, gardner, "k--", linewidth=1.3, label="Gardner")
        ax.set_title(f"{form}  (n={len(d):,})", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlim(1500, 6500); ax.set_ylim(1.5, 3.3)
    for ax in axes[len(NLOG_FORMATIONS):]:
        ax.axis("off")
    axes[0].set_ylabel("RHOB (g/cc)"); axes[4].set_ylabel("RHOB (g/cc)")
    for ax in axes[-4:]:
        ax.set_xlabel("Vp (m/s)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Gardner's relation sanity check — RHOB vs Vp per formation",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rhob_vs_gr(df: pd.DataFrame, out_path: Path) -> None:
    """Single-panel GR-RHOB crossplot, all formations overlaid."""
    wide = _load_wide_nlog(df, ["rhob", "gr_api"])
    if len(wide) == 0:
        print("  [skip] no paired RHOB+GR"); return

    sub = []
    for f in NLOG_FORMATIONS:
        d = wide[wide["formation"] == f]
        if len(d) > 20_000:
            d = d.sample(20_000, random_state=42)
        sub.append(d)
    wide = pd.concat(sub, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    for f in NLOG_FORMATIONS:
        d = wide[wide["formation"] == f]
        if len(d) == 0:
            continue
        ax.scatter(d["gr_api"], d["rhob"], s=3, alpha=0.2,
                   c=FORMATION_COLOURS.get(f, "#999"), edgecolor="none",
                   label=f"{f} (n={len(d):,})")
    ax.set_xlabel("GR (API)"); ax.set_ylabel("RHOB (g/cc)")
    ax.set_xlim(0, 200); ax.set_ylim(1.5, 3.2)
    ax.set_title("GR vs RHOB — NLOG by formation", fontweight="bold")
    ax.legend(markerscale=4, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_resistivity_vs_porosity(df: pd.DataFrame, out_path: Path) -> None:
    """log10(R_deep) vs NPHI per formation — Pickett-style fluid plot."""
    wide = _load_wide_nlog(df, ["res_deep_log", "nphi"])
    if len(wide) == 0:
        print("  [skip] no paired resistivity+NPHI"); return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, form in zip(axes, NLOG_FORMATIONS):
        d = wide[wide["formation"] == form]
        if len(d) == 0:
            ax.axis("off"); continue
        if len(d) > 10_000:
            d = d.sample(10_000, random_state=42)
        ax.scatter(d["nphi"], d["res_deep_log"], s=3, alpha=0.2,
                   c=FORMATION_COLOURS.get(form, "#999"), edgecolor="none")
        ax.set_title(f"{form}  (n={len(d):,})", fontsize=10)
        ax.set_xlim(-0.05, 0.6); ax.set_ylim(-1, 4)
        ax.grid(alpha=0.25)
    for ax in axes[len(NLOG_FORMATIONS):]:
        ax.axis("off")
    axes[0].set_ylabel("log₁₀ R_deep (Ω·m)")
    axes[4].set_ylabel("log₁₀ R_deep (Ω·m)")
    for ax in axes[-4:]:
        ax.set_xlabel("NPHI (v/v)")
    fig.suptitle("Resistivity vs porosity — NLOG by formation",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")




def _has_fine_cols(df: pd.DataFrame) -> bool:

    needed = {"rock_type_fine", "strat_unit", "location_type"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  [skip] missing v4 columns {missing} — re-run pull_data.py v4")
        return False
    return True

ROCK_FINE_COLOURS = {
    "chalk":       "#a6d96a", "sandstone":   "#b15928", "claystone":   "#8073ac",
    "clay":        "#ef8a62", "mudstone":    "#b2abd2", "halite":      "#fdbf6f",
    "anhydrite":   "#ff7f00", "carbonate":   "#d73027", "siltstone":   "#fdae61",
    "limestone":   "#33a02c", "nanno_ooze":  "#67a9cf", "diatom_ooze": "#2166ac",
    "basalt":      "#4d4d4d", "other":       "#bbbbbb",
}


def plot_joint_support_by_rock_fine(df: pd.DataFrame, out_path: Path,
                                     measurement: str = "rhob",
                                     top_n: int = 8) -> None:
    """For each fine rock type, show the COMBINED LILY+NLOG distribution
    with per-dataset contributions stacked / coloured. Headline figure for
    'what is the support of P(log | rock)?'"""
    if not _has_fine_cols(df):
        return
    d = df[df["measurement"] == measurement]
    if len(d) == 0:
        print(f"  [skip] no {measurement} data"); return

    xlim = DISPLAY_BOUNDS.get(measurement, (d["value"].min(), d["value"].max()))
    # pick rock types with enough total data
    counts = d["rock_type_fine"].value_counts()
    rts = counts.head(top_n).index.tolist()

    n_cols = min(4, len(rts))
    n_rows = (len(rts) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows),
                              sharex=True, squeeze=False)

    bins_arr = np.linspace(*xlim, 55)
    for i, rt in enumerate(rts):
        ax = axes[i // n_cols][i % n_cols]
        lily_vals = d.loc[(d["rock_type_fine"] == rt) &
                          (d["dataset"] == "LILY"), "value"].values
        nlog_vals = d.loc[(d["rock_type_fine"] == rt) &
                          (d["dataset"] == "NLOG"), "value"].values
        # subsample for speed
        if len(nlog_vals) > 60_000:
            nlog_vals = _subsample(nlog_vals, 60_000)
        if len(lily_vals) > 30_000:
            lily_vals = _subsample(lily_vals, 30_000)

        combined = np.concatenate([lily_vals, nlog_vals])
        combined = combined[(combined >= xlim[0]) & (combined <= xlim[1])]
        if len(combined) < 30:
            ax.axis("off"); continue

        # filled combined (the SUPPORT of P(log|rock))
        ax.hist(combined, bins=bins_arr, density=True, color="#dddddd",
                edgecolor="#555555", linewidth=0.5, alpha=0.8,
                label=f"combined support (n={len(combined):,})")
        # per-dataset overlays
        if len(lily_vals) >= 30:
            lv = lily_vals[(lily_vals >= xlim[0]) & (lily_vals <= xlim[1])]
            ax.hist(lv, bins=bins_arr, density=True, histtype="step",
                    linewidth=1.6, color=DATASET_COLOURS["LILY"],
                    label=f"LILY (n={len(lv):,})")
        if len(nlog_vals) >= 30:
            nv = nlog_vals[(nlog_vals >= xlim[0]) & (nlog_vals <= xlim[1])]
            ax.hist(nv, bins=bins_arr, density=True, histtype="step",
                    linewidth=1.6, color=DATASET_COLOURS["NLOG"],
                    label=f"NLOG (n={len(nv):,})")
        # mark P5 and P95 of combined support
        p5, p95 = np.percentile(combined, [5, 95])
        ax.axvline(p5, color="black", linestyle=":", linewidth=0.9)
        ax.axvline(p95, color="black", linestyle=":", linewidth=0.9)
        ax.set_title(f"{rt}", fontweight="bold")
        ax.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement),
                      fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    # hide unused cells
    for j in range(len(rts), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Joint support of P({measurement} | rock_type_fine)\n"
                 "grey = full combined distribution,  lines = per-dataset contributions\n"
                 "dashed verticals = P5 / P95 of the combined support",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_fine_vs_coarse_distributions(df: pd.DataFrame, out_path: Path,
                                       measurement: str = "rhob") -> None:
    """For each formation where the fine split reveals a cleaner structure,
    compare the distribution using rock_type (coarse) vs rock_type_fine."""
    if not _has_fine_cols(df):
        return
    nlog = df[(df["dataset"] == "NLOG") &
              (df["measurement"] == measurement)]
    if len(nlog) == 0:
        print(f"  [skip] no NLOG {measurement} data"); return

    # Find formations where the fine split produces >1 distinct rock_type_fine
    split_formations = []
    for formation, grp in nlog.groupby("formation"):
        unique_fine = grp["rock_type_fine"].nunique()
        if unique_fine >= 2 and len(grp) >= 5_000:
            split_formations.append((formation, grp["rock_type"].iloc[0],
                                      grp["rock_type_fine"].value_counts()))
    if not split_formations:
        print(f"  [skip] no formations with a meaningful fine split")
        return

    # sort by sample size
    split_formations = sorted(split_formations,
                               key=lambda x: -x[2].sum())[:6]

    xlim = DISPLAY_BOUNDS.get(measurement, None)
    bins_arr = np.linspace(*xlim, 55) if xlim else None

    n = len(split_formations)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    for i, (formation, coarse_label, fine_counts) in enumerate(split_formations):
        sub = nlog[nlog["formation"] == formation]
        # left panel: coarse (single overlay)
        ax_l = axes[i][0]
        vals = sub["value"].values
        if xlim:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) > 80_000:
            vals = _subsample(vals, 80_000)
        ax_l.hist(vals, bins=bins_arr, density=True, color="#888888",
                  alpha=0.8, edgecolor="black", linewidth=0.4,
                  label=f"{coarse_label} (n={len(vals):,})")
        ax_l.set_title(f"{formation} — coarse label ({coarse_label})",
                        fontsize=10, fontweight="bold")
        ax_l.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement),
                         fontsize=9)
        ax_l.set_ylabel("density"); ax_l.grid(alpha=0.25)
        ax_l.legend(fontsize=8)

        # right panel: fine (multiple overlays)
        ax_r = axes[i][1]
        fine_types = fine_counts.head(5).index.tolist()
        for rt in fine_types:
            vals = sub.loc[sub["rock_type_fine"] == rt, "value"].values
            if xlim:
                vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
            if len(vals) < 200:
                continue
            if len(vals) > 40_000:
                vals = _subsample(vals, 40_000)
            ax_r.hist(vals, bins=bins_arr, density=True, histtype="step",
                      linewidth=1.7, color=ROCK_FINE_COLOURS.get(rt, "#444"),
                      label=f"{rt} (n={len(vals):,})")
        ax_r.set_title(f"{formation} — fine split (by strat_unit)",
                        fontsize=10, fontweight="bold")
        ax_r.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement),
                         fontsize=9)
        ax_r.grid(alpha=0.25)
        ax_r.legend(fontsize=8)

    fig.suptitle(f"Fine rock-type split within formation — {measurement}\n"
                 "left: coarse single-label distribution     right: split by stratUnitId sub-member",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_nonlinear_compaction_fit(df: pd.DataFrame, out_path: Path,
                                    measurement: str = "rhob") -> None:

    if not _has_fine_cols(df):
        return
    d = df[df["measurement"] == measurement].copy()
    if len(d) == 0:
        print(f"  [skip] no {measurement} data"); return

    ylim = DISPLAY_BOUNDS.get(measurement, (d["value"].min(), d["value"].max()))

    # pick rock types present in BOTH datasets
    per = d.groupby(["rock_type_fine", "dataset"])["value"].size().unstack(fill_value=0)
    mask = (per.get("LILY", 0) >= 50) & (per.get("NLOG", 0) >= 50)
    rts = per[mask].index.tolist()[:6]
    if not rts:
        print(f"  [skip] no rock types in both datasets for {measurement}")
        return

    n_cols = min(3, len(rts))
    n_rows = (len(rts) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                              squeeze=False)

    depth_grid = np.linspace(5, 5000, 400)

    for i, rt in enumerate(rts):
        ax = axes[i // n_cols][i % n_cols]
        sub = d[d["rock_type_fine"] == rt]
        for ds, col in DATASET_COLOURS.items():
            dd = sub[sub["dataset"] == ds]
            if len(dd) > 5_000:
                dd = dd.sample(5_000, random_state=42)
            ax.scatter(dd["value"], dd["depth"], s=4, alpha=0.25,
                       c=col, edgecolor="none",
                       label=f"{ds} (n={sub[sub['dataset']==ds].shape[0]:,})")

        # Non-linear fit using a median-per-depth-bin approach
        # then fit y = a + b * (1 - exp(-depth/c)) via scipy if available,
        # else simple median-per-bin curve
        bin_edges = np.logspace(np.log10(5), np.log10(5000), 25)
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        medians, sizes = [], []
        for lo_e, hi_e in zip(bin_edges[:-1], bin_edges[1:]):
            vals = sub.loc[(sub["depth"] >= lo_e) & (sub["depth"] < hi_e), "value"]
            if len(vals) >= 20:
                medians.append(vals.median())
                sizes.append(len(vals))
            else:
                medians.append(np.nan)
                sizes.append(0)
        medians = np.array(medians)
        good = ~np.isnan(medians)
        if good.sum() >= 3:
            ax.plot(medians[good], bin_mids[good], "o-", color="black",
                    linewidth=1.8, markersize=5, zorder=5,
                    label=f"median per depth bin (pooled)")

        ax.set_title(rt, fontweight="bold")
        ax.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement),
                       fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("depth (m)", fontsize=9)
        ax.set_xlim(ylim); ax.set_ylim(5, 5000)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, loc="lower right" if measurement == "rhob" else "upper right")

    for j in range(len(rts), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Non-linear {measurement}–depth relation revealed by pooling LILY + NLOG\n",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_depth_hexbin_by_rock_fine(df: pd.DataFrame, out_path: Path,
                                     measurement: str = "rhob") -> None:
    """2D hexbin density map of measurement vs depth per fine rock type.
    Cleaner than scatter for the large combined dataset."""
    if not _has_fine_cols(df):
        return
    d = df[df["measurement"] == measurement][["value", "depth", "rock_type_fine"]]
    if len(d) == 0:
        return

    xlim = DISPLAY_BOUNDS.get(measurement, (d["value"].min(), d["value"].max()))
    per = d.groupby("rock_type_fine")["value"].size()
    rts = per[per >= 5_000].sort_values(ascending=False).index.tolist()[:6]
    if not rts:
        print(f"  [skip] insufficient data")
        return

    n_cols = min(3, len(rts))
    n_rows = (len(rts) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                              squeeze=False, sharey=True)
    for i, rt in enumerate(rts):
        ax = axes[i // n_cols][i % n_cols]
        sub = d[d["rock_type_fine"] == rt]
        total_n = len(sub)
        # aggressive subsample — hexbin is memory-heavy
        if len(sub) > 80_000:
            sub = sub.sample(80_000, random_state=42)
        hb = ax.hexbin(sub["value"].values, sub["depth"].values,
                        gridsize=45, cmap="viridis", mincnt=1,
                        extent=(xlim[0], xlim[1], 0, 5000))
        ax.invert_yaxis()
        ax.set_xlim(xlim); ax.set_ylim(5000, 0)
        ax.set_title(f"{rt}  (n={total_n:,})", fontweight="bold")
        ax.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement))
        if i % n_cols == 0:
            ax.set_ylabel("depth (m)")
        fig.colorbar(hb, ax=ax, label="count")

    for j in range(len(rts), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"{measurement} × depth density per fine rock type (hexbin)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_support_bounds_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap showing P5 / P95 support bounds per (rock_type_fine × measurement),
    with LILY and NLOG contributions annotated. Designed to make the 'bounded
    space of what's possible' framing legible in one glance."""
    if not _has_fine_cols(df):
        return
    measurements = ["rhob", "gr_api", "dt_us_ft", "nphi", "pef"]
    rts = (df.groupby("rock_type_fine")["value"].size()
             .sort_values(ascending=False).head(10).index.tolist())

    # Build a matrix of (P95 - P5) range widths per (rock, meas)
    mat_range = np.full((len(rts), len(measurements)), np.nan)
    mat_lily_expands = np.zeros((len(rts), len(measurements)), dtype=bool)

    for i, rt in enumerate(rts):
        for j, meas in enumerate(measurements):
            sub = df[(df["rock_type_fine"] == rt) & (df["measurement"] == meas)]
            if len(sub) < 50:
                continue
            combined_p5, combined_p95 = np.percentile(sub["value"], [5, 95])
            mat_range[i, j] = combined_p95 - combined_p5
            # check if LILY bounds extend the NLOG support
            lily_v = sub.loc[sub["dataset"] == "LILY", "value"]
            nlog_v = sub.loc[sub["dataset"] == "NLOG", "value"]
            if len(lily_v) >= 50 and len(nlog_v) >= 50:
                lily_lo, lily_hi = np.percentile(lily_v, [5, 95])
                nlog_lo, nlog_hi = np.percentile(nlog_v, [5, 95])
                if lily_lo < nlog_lo - 0.05 * mat_range[i, j] or \
                   lily_hi > nlog_hi + 0.05 * mat_range[i, j]:
                    mat_lily_expands[i, j] = True

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(rts))))
    im = ax.imshow(mat_range, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(measurements)))
    ax.set_xticklabels(measurements, rotation=35, ha="right")
    ax.set_yticks(range(len(rts)))
    ax.set_yticklabels(rts)
    ax.set_title("Support width (P95 − P5) per rock × measurement\n"
                  "★ = LILY extends the support beyond NLOG's range",
                  fontweight="bold")

    for i in range(len(rts)):
        for j in range(len(measurements)):
            v = mat_range[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "-"
            if mat_lily_expands[i, j]:
                txt = "★" + txt
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if v and v > np.nanmean(mat_range) else "black")
    fig.colorbar(im, ax=ax, label="support width (P95-P5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_onshore_vs_offshore(df: pd.DataFrame, out_path: Path,
                              measurement: str = "rhob") -> None:
    """Does NLOG distribution shape differ between onshore and offshore
    wells at matched rock type?  Different depositional environments,
    different operator eras - useful conditioning variable to be aware of."""
    if not _has_fine_cols(df):
        return
    nlog = df[(df["dataset"] == "NLOG") & (df["measurement"] == measurement)]
    nlog = nlog[nlog["location_type"].isin(["onshore", "offshore"])]
    if len(nlog) == 0:
        print(f"  [skip] no location_type data"); return

    xlim = DISPLAY_BOUNDS.get(measurement, (nlog["value"].min(), nlog["value"].max()))
    rts = (nlog.groupby("rock_type_fine")["value"].size()
              .sort_values(ascending=False).head(6).index.tolist())

    n_cols = min(3, len(rts))
    n_rows = (len(rts) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows),
                              sharex=True, squeeze=False)

    bins_arr = np.linspace(*xlim, 50)
    loc_colours = {"onshore": "#2ca25f", "offshore": "#4a1486"}
    for i, rt in enumerate(rts):
        ax = axes[i // n_cols][i % n_cols]
        for loc in ["onshore", "offshore"]:
            vals = nlog.loc[(nlog["rock_type_fine"] == rt) &
                             (nlog["location_type"] == loc), "value"].values
            if len(vals) < 50:
                continue
            if len(vals) > 60_000:
                vals = _subsample(vals, 60_000)
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
            ax.hist(vals, bins=bins_arr, density=True, histtype="step",
                    linewidth=1.8, color=loc_colours[loc],
                    label=f"{loc} (n={len(vals):,})")
        ax.set_title(rt, fontweight="bold")
        ax.set_xlabel(MEASUREMENT_LABELS.get(measurement, measurement), fontsize=9)
        if i % n_cols == 0:
            ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    for j in range(len(rts), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"Onshore vs offshore — NLOG {measurement} per rock_type_fine",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")



def plot_nphi_by_rock_fine(df: pd.DataFrame, out_path: Path) -> None:
    """NPHI (neutron porosity) distributions per fine rock type.  NLOG-only —
    LILY does not run neutron porosity logs.  Three panels:
       • histogram overlay per rock type
       • NPHI vs depth hexbin per rock type (compaction in porosity-space)
       • NPHI vs RHOB crossplot per rock type (classic lithology discriminator)
    NPHI goes near zero for halite/anhydrite, ~0.15 for sandstone, ~0.30+
    for claystone and chalk. Negative values exist for halite because its
    thermal-neutron response is not water-filled."""
    if not _has_fine_cols(df):
        return
    d = df[(df["dataset"] == "NLOG") & (df["measurement"] == "nphi")]
    if len(d) == 0:
        print("  [skip] no NLOG NPHI data"); return

    # pick rock types with enough samples
    rts = (d.groupby("rock_type_fine")["value"].size()
             .sort_values(ascending=False).head(8).index.tolist())
    xlim = DISPLAY_BOUNDS.get("nphi", (-0.05, 0.6))

    # build a wide frame for NPHI-vs-RHOB crossplot
    wide = _load_wide_nlog(df, ["nphi", "rhob"])

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

    # panel 1 — histogram overlay
    ax1 = fig.add_subplot(gs[0, 0])
    bins_arr = np.linspace(*xlim, 55)
    for rt in rts:
        vals = d.loc[d["rock_type_fine"] == rt, "value"].values
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        if len(vals) < 100:
            continue
        if len(vals) > 80_000:
            vals = _subsample(vals, 80_000)
        ax1.hist(vals, bins=bins_arr, density=True, histtype="step",
                 linewidth=1.7, color=ROCK_FINE_COLOURS.get(rt, "#666"),
                 label=f"{rt} (n={len(vals):,})")
    ax1.set_xlabel(MEASUREMENT_LABELS["nphi"])
    ax1.set_ylabel("density")
    ax1.set_title("NPHI distributions per rock_type_fine (NLOG)",
                   fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.25)

    # panel 2 — NPHI vs depth hexbin for top 4 rock types combined
    ax2 = fig.add_subplot(gs[0, 1])
    sub_all = d[d["rock_type_fine"].isin(rts[:4])]
    if len(sub_all) > 200_000:
        sub_all = sub_all.sample(200_000, random_state=42)
    hb = ax2.hexbin(sub_all["value"].values, sub_all["depth"].values,
                     gridsize=45, cmap="viridis", mincnt=1,
                     extent=(xlim[0], xlim[1], 0, 5000))
    ax2.invert_yaxis()
    ax2.set_xlim(xlim); ax2.set_ylim(5000, 0)
    ax2.set_xlabel(MEASUREMENT_LABELS["nphi"])
    ax2.set_ylabel("depth (m)")
    ax2.set_title(f"NPHI × depth density\n(top {len(rts[:4])} rock types pooled)",
                   fontweight="bold")
    fig.colorbar(hb, ax=ax2, label="count")

    # panel 3 — NPHI vs RHOB crossplot, coloured by rock_type_fine
    ax3 = fig.add_subplot(gs[1, :])
    if len(wide):
        # attach rock_type_fine to wide via merge (borehole, depth)
        meta = (df[(df["dataset"] == "NLOG") & (df["measurement"] == "rhob")]
                  [["borehole", "depth", "rock_type_fine"]]
                  .drop_duplicates(subset=["borehole", "depth"]))
        wide_rt = wide.merge(meta, on=["borehole", "depth"], how="left")
        wide_rt = wide_rt.dropna(subset=["rock_type_fine"])
        for rt in rts:
            w = wide_rt[wide_rt["rock_type_fine"] == rt]
            if len(w) < 200:
                continue
            if len(w) > 12_000:
                w = w.sample(12_000, random_state=42)
            ax3.scatter(w["nphi"], w["rhob"], s=3, alpha=0.25,
                         c=ROCK_FINE_COLOURS.get(rt, "#666"), edgecolor="none",
                         label=f"{rt} (n={len(w):,})")
        # reference lines — classic fluid-substitution indicators
        ax3.axhline(2.65, color="grey", linewidth=0.4, linestyle="--")
        ax3.axhline(2.95, color="grey", linewidth=0.4, linestyle="--")
        ax3.text(0.55, 2.66, "quartz (2.65)", fontsize=7, color="grey")
        ax3.text(0.55, 2.96, "anhydrite (2.95)", fontsize=7, color="grey")
    ax3.set_xlim(xlim); ax3.set_ylim(1.5, 3.2)
    ax3.invert_yaxis()
    ax3.set_xlabel(MEASUREMENT_LABELS["nphi"])
    ax3.set_ylabel(MEASUREMENT_LABELS["rhob"])
    ax3.set_title("NPHI × RHOB crossplot (NLOG) — lithology discriminator",
                   fontweight="bold")
    ax3.legend(markerscale=4, fontsize=8, loc="lower right")
    ax3.grid(alpha=0.25)

    fig.suptitle("NPHI diagnostic plots (NLOG wireline)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_msus_by_rock_type_lily(df: pd.DataFrame, out_path: Path) -> None:
    """Magnetic susceptibility distributions per LILY rock type.  LILY-only —
    NLOG does not measure susceptibility.  msus_si ranges from ~1e-6 (pure
    calcareous material) to ~1e-2 (basalt), so the x-axis is log-scale.
    Two panels:
       • log-scale histogram overlay per rock type
       • susceptibility vs depth per rock type (low but positive for clay,
         near-zero for oozes, very high for basalt)"""
    if not _has_fine_cols(df):
        return
    d = df[(df["dataset"] == "LILY") & (df["measurement"] == "msus_si")]
    if len(d) == 0:
        print("  [skip] no LILY msus_si data"); return

    rts = (d.groupby("rock_type_fine")["value"].size()
             .sort_values(ascending=False).head(8).index.tolist())
    # magnetic susceptibility has heavy-tailed distribution → log axis
    xlim = (1e-6, 1e-1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # panel 1 — log-scale histogram overlay
    ax1 = axes[0]
    bins_arr = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 45)
    for rt in rts:
        vals = d.loc[d["rock_type_fine"] == rt, "value"].values
        # msus_si can include zeros — shift to a tiny floor for log plotting
        vals = np.clip(vals, xlim[0], None)
        vals = vals[vals <= xlim[1]]
        if len(vals) < 30:
            continue
        ax1.hist(vals, bins=bins_arr, density=True, histtype="step",
                 linewidth=1.7, color=ROCK_FINE_COLOURS.get(rt, "#666"),
                 label=f"{rt} (n={len(vals):,})")
    ax1.set_xscale("log")
    ax1.set_xlabel(MEASUREMENT_LABELS["msus_si"])
    ax1.set_ylabel("density")
    ax1.set_xlim(xlim)
    ax1.set_title("Magnetic susceptibility per LILY rock_type_fine",
                   fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25, which="both")

    # panel 2 — susceptibility vs depth, coloured by rock type
    ax2 = axes[1]
    for rt in rts:
        sub = d[d["rock_type_fine"] == rt]
        if len(sub) < 30:
            continue
        vals = np.clip(sub["value"].values, xlim[0], xlim[1])
        depths = sub["depth"].values
        ax2.scatter(vals, depths, s=10, alpha=0.5,
                     c=ROCK_FINE_COLOURS.get(rt, "#666"), edgecolor="none",
                     label=f"{rt} (n={len(sub):,})")
    ax2.set_xscale("log")
    ax2.set_xlim(xlim)
    ax2.set_ylim(0, 2000)
    ax2.invert_yaxis()
    ax2.set_xlabel(MEASUREMENT_LABELS["msus_si"])
    ax2.set_ylabel("depth (m)")
    ax2.set_title("Susceptibility × depth",
                   fontweight="bold")
    ax2.legend(markerscale=1.5, fontsize=8, loc="lower right")
    ax2.grid(alpha=0.25, which="both")

    fig.suptitle("Magnetic susceptibility — LILY-only measurement",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")