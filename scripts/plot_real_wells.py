"""
Plot real NLOG well logs with formation bands highlighted.

For each well: a single figure with N panels, one per variable
(rhob, gr_api, dt_us_ft, nphi, pef, res_deep_log). Y axis is depth
(increasing downwards, like geologists draw it). X axis is the
variable. Background is shaded by formation, with the ZE formation
highlighted in a distinct colour. A formation legend runs down the
right side.

Usage:
    python plot_real_wells.py --formation ZE --n 3
    python plot_real_wells.py --wells D15-03 P01-03 BLF-102
    python plot_real_wells.py --wells L04-A-03-S2 --highlight ZE
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


VARS = ["rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log"]
VAR_LABELS = {
    "rhob": "RHOB (g/cc)",
    "gr_api": "GR (API)",
    "dt_us_ft": "DT (us/ft)",
    "nphi": "NPHI",
    "pef": "PEF",
    "res_deep_log": "RES (log10)",
}

# realistic axis ranges — clip so spikes don't compress everything
VAR_LIMS = {
    "rhob": (1.5, 3.2),
    "gr_api": (0.0, 200.0),
    "dt_us_ft": (40.0, 200.0),
    "nphi": (-0.05, 0.6),
    "pef": (0.0, 8.0),
    "res_deep_log": (-1.0, 4.0),
}

# stratigraphic colours: greys/browns for non-target, accent for ZE
FM_COLOR = {
    "NU": "#f5f5f5",
    "NM": "#ebebeb",
    "NL": "#dcdcdc",
    "CK": "#fff5e1",
    "KN": "#e6ddc8",
    "SL": "#d4c8a8",
    "SG": "#c8baa0",
    "AT": "#bda88c",
    "RN": "#b09578",
    "RB": "#a08560",
    "ZE": "#9bcfb8",   # highlight
    "RO": "#cc9978",
    "DC": "#9c7050",
}


def _pivot_well_to_wide(well_long: pd.DataFrame) -> pd.DataFrame:
    var_long = well_long[well_long["measurement"].isin(VARS)].copy()
    var_wide = (
        var_long.pivot_table(
            index="depth", columns="measurement",
            values="value", aggfunc="first",
        )
        .reset_index()
    )
    for v in VARS:
        if v not in var_wide.columns:
            var_wide[v] = np.nan

    label_cols = [c for c in ("rock_type_fine", "formation")
                  if c in well_long.columns]
    if label_cols:
        labels = (
            well_long.groupby("depth")[label_cols]
            .first().reset_index()
        )
        var_wide = var_wide.merge(labels, on="depth", how="left")

    return var_wide.sort_values("depth").reset_index(drop=True)


def _formation_intervals(
    well_wide: pd.DataFrame,
) -> list[tuple[str, float, float]]:
    """Return list of (formation, top, bot) tuples for contiguous runs."""
    if "formation" not in well_wide.columns or len(well_wide) == 0:
        return []
    fms = well_wide["formation"].astype(str).fillna("")
    depths = well_wide["depth"].values

    intervals = []
    cur_fm = fms.iloc[0]
    cur_top = depths[0]
    for i in range(1, len(fms)):
        if fms.iloc[i] != cur_fm:
            intervals.append((cur_fm, cur_top, depths[i - 1]))
            cur_fm = fms.iloc[i]
            cur_top = depths[i]
    intervals.append((cur_fm, cur_top, depths[-1]))
    return intervals


def _plot_well(
    well_wide: pd.DataFrame,
    well_name: str,
    out_path: Path,
    highlight: str = "ZE",
) -> None:
    n_vars = len(VARS)
    fig, axes = plt.subplots(
        1, n_vars,
        figsize=(2.6 * n_vars + 2.0, 11),
        sharey=True,
    )
    if n_vars == 1:
        axes = [axes]

    intervals = _formation_intervals(well_wide)
    fms_present = sorted({fm for fm, _, _ in intervals if fm})

    d_min = float(well_wide["depth"].min())
    d_max = float(well_wide["depth"].max())

    for ax_i, var in enumerate(VARS):
        ax = axes[ax_i]

        # background formation bands
        for fm, top, bot in intervals:
            if not fm:
                continue
            color = FM_COLOR.get(fm, "#dddddd")
            alpha = 0.95 if fm == highlight else 0.55
            ax.axhspan(top, bot, facecolor=color, alpha=alpha,
                       edgecolor="none", zorder=0)

        # the variable curve
        if var == "res_deep_log":
            vals = np.log10(np.abs(well_wide[var].values) + 1e-3)
        else:
            vals = well_wide[var].values
        depths = well_wide["depth"].values

        # clip to display range so single spikes don't dominate
        lo, hi = VAR_LIMS[var]
        vals_clipped = np.clip(vals, lo, hi)

        ax.plot(vals_clipped, depths, color="black",
                linewidth=0.6, zorder=2)

        # mark formation tops with a horizontal line
        for fm, top, bot in intervals:
            if fm == highlight:
                ax.axhline(top, color="darkgreen", linewidth=1.0,
                            linestyle="-", alpha=0.7, zorder=3)
                ax.axhline(bot, color="darkgreen", linewidth=1.0,
                            linestyle="-", alpha=0.7, zorder=3)

        ax.set_xlim(lo, hi)
        ax.set_xlabel(VAR_LABELS[var])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")
        if ax_i == 0:
            ax.set_ylabel("Depth (m)")

    # depth axis: increasing downwards
    axes[0].set_ylim(d_max, d_min)

    # formation legend on the right
    legend_handles = [
        Patch(
            facecolor=FM_COLOR.get(fm, "#dddddd"),
            alpha=0.95 if fm == highlight else 0.55,
            edgecolor="black" if fm == highlight else "none",
            linewidth=1.5 if fm == highlight else 0,
            label=fm + (" (highlighted)" if fm == highlight else ""),
        )
        for fm in fms_present
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=True,
        title="Formation",
    )

    fig.suptitle(
        f"Real NLOG well: {well_name}    "
        f"depth {d_min:.0f}-{d_max:.0f}m    "
        f"highlight={highlight}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.985])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _pick_wells_with_formation(
    df: pd.DataFrame, formation: str, n: int,
    rng: np.random.Generator, min_thickness_m: float = 200.0,
) -> list[str]:
    sub = df[df["formation"] == formation]
    well_thick = sub.groupby("borehole")["depth"].agg(["min", "max"])
    well_thick["thk"] = well_thick["max"] - well_thick["min"]
    qualifying = set(
        well_thick[well_thick["thk"] >= min_thickness_m].index
    )
    if not qualifying:
        return []

    cand_df = df[df["borehole"].isin(qualifying)
                  & df["measurement"].isin(VARS)]
    var_per_well = (
        cand_df.dropna(subset=["value"])
        .groupby("borehole")["measurement"]
        .nunique()
    )
    good = var_per_well[var_per_well >= 4].index.tolist()
    if not good:
        good = var_per_well[var_per_well >= 2].index.tolist()
    if not good:
        good = list(qualifying)

    chosen = rng.choice(good, size=min(n, len(good)), replace=False)
    return [str(b) for b in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--formation", default="ZE",
                   help="formation that wells must contain")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--wells", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--highlight", default="ZE",
                   help="formation to highlight (default ZE)")
    p.add_argument("--max_depth", type=float, default=4400.0)
    p.add_argument("--out_dir", default="plots/real_wells")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading samples.parquet...")
    df = pd.read_parquet("data/clean/samples.parquet")
    df = df[df["dataset"] == "NLOG"]
    df = df.sort_values(["borehole", "depth"])
    print(f"  {len(df):,} rows")

    if args.wells:
        wells = args.wells
    else:
        rng = np.random.default_rng(args.seed)
        wells = _pick_wells_with_formation(
            df, args.formation, args.n, rng,
        )
        if not wells:
            print(f"no wells found for {args.formation}")
            return
    print(f"plotting {len(wells)} wells: {wells}")

    for well in wells:
        well_long = df[df["borehole"] == well].copy()
        well_long = well_long[well_long["depth"] <= args.max_depth]
        if len(well_long) == 0:
            print(f"  {well}: no rows below {args.max_depth}m, skipping")
            continue
        well_wide = _pivot_well_to_wide(well_long)
        out_path = out_dir / f"{well}.png"
        _plot_well(well_wide, well, out_path, highlight=args.highlight)

    print(f"\ndone — plots in {out_dir}/")


if __name__ == "__main__":
    main()