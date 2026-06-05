"""
Garzon et al. (2026) geology-informed metrics for sim-vs-real
stratigraphic validation.

Three metric families, applied to our simulator's output vs real
NLOG wells:

  1. Extent       per-formation presence rate
                  (P(formation in well), real vs sim).
                  Reported per formation + an overall IoU figure.

  2. Sequence     per-well formation order (top -> base).
                  Compared with normalised Levenshtein edit
                  distance and a co-occurrence frequency matrix.
                  Frobenius distance reported.

  3. Position     per-formation top-depth + base-depth
                  distributions, compared with Wasserstein-1.

Reference:
  Garzon, Dabekaussen, Busschers, De Boever, Mehrkanoon, Karssenberg
  (2026). "Assessment of automated stratigraphic interpretations of
  boreholes with geology-informed metrics."
  Computers & Geosciences 207:106043.
  doi:10.1016/j.cageo.2025.106043

Outputs:
  plots/garcon/metric_extent.png
  plots/garcon/metric_sequence.png
  plots/garcon/metric_position.png
  plots/analysis/garcon_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator import sample_column, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry


# Formations we report on; ordered top -> base in the typical
# Dutch column.  Anything else is rolled up to "other".
TARGET_FORMATIONS = [
    "NU", "NM", "NL", "CK", "KN", "AT", "SL", "SG",
    "ZE", "RO", "RB", "RN", "DC", "SK",
]


# ----------------------------------------------------------- real-data side --

def real_well_sequences(df: pd.DataFrame, n_max: int | None = None
                        ) -> pd.DataFrame:
    """For each real NLOG well, return its formation sequence (top->base)
    plus per-formation top/base depths.

    Returns dataframe with columns
        borehole, formation, top, bot
    sorted by (borehole, top).
    """
    nlog = df[df.dataset == "NLOG"].copy()
    if n_max is not None:
        wells = nlog.borehole.drop_duplicates().sample(
            n=min(n_max, nlog.borehole.nunique()),
            random_state=42).tolist()
        nlog = nlog[nlog.borehole.isin(wells)]

    # per (well, formation), compute top and bot from depth column
    grp = (nlog.groupby(["borehole", "formation"])
           .agg(top=("depth", "min"), bot=("depth", "max"))
           .reset_index())
    grp = grp[grp.formation.isin(TARGET_FORMATIONS)]
    grp = grp.sort_values(["borehole", "top"]).reset_index(drop=True)
    return grp


# -------------------------------------------------------- simulator-side ---

def sim_well_sequences(geom: FormationGeometry, n: int = 1000,
                       max_depth: float = 4400.0, seed: int = 42
                       ) -> pd.DataFrame:
    """Draw n synthetic columns; collapse each to a (borehole, formation,
    top, bot) row set, same schema as `real_well_sequences`."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        col = sample_column(rng=rng, geometry=geom, max_depth=max_depth)
        for fm, rocks, top, bot in col.layers:
            if fm in TARGET_FORMATIONS:
                rows.append({"borehole": f"SIM_{i:05d}",
                             "formation": fm,
                             "top": float(top), "bot": float(bot)})
    return pd.DataFrame(rows).sort_values(["borehole", "top"]).reset_index(drop=True)


# ------------------------------------------------------------ extent metric --

def extent_metric(real: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Per-formation presence rate, real vs sim.

    Presence rate = fraction of wells containing the formation at all.
    """
    real_wells = real.borehole.nunique()
    sim_wells = sim.borehole.nunique()
    rows = []
    for fm in TARGET_FORMATIONS:
        r_n = real[real.formation == fm].borehole.nunique()
        s_n = sim[sim.formation == fm].borehole.nunique()
        rows.append({
            "formation": fm,
            "real_presence": r_n / max(real_wells, 1),
            "sim_presence": s_n / max(sim_wells, 1),
            "abs_diff": abs(r_n / max(real_wells, 1)
                            - s_n / max(sim_wells, 1)),
        })
    df = pd.DataFrame(rows)
    return df


def plot_extent(ext: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(ext))
    ax.bar(x - 0.2, ext.real_presence * 100, width=0.4,
           color="C0", label="Real NLOG")
    ax.bar(x + 0.2, ext.sim_presence * 100, width=0.4,
           color="C1", label="Simulator")
    ax.set_xticks(x)
    ax.set_xticklabels(ext.formation, rotation=0)
    ax.set_ylabel("Presence rate (% of wells)")
    ax.set_title("Extent metric: per-formation presence rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for xi, d in zip(x, ext.abs_diff):
        ax.annotate(f"{d:.2f}", (xi, max(ext.real_presence.iloc[xi],
                                          ext.sim_presence.iloc[xi]) * 100 + 1),
                    ha="center", fontsize=7, color="dimgrey")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------- sequence metric

def _well_sequence(well_df: pd.DataFrame) -> list[str]:
    return well_df.sort_values("top").formation.tolist()


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Simple Levenshtein distance for short token lists."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def sequence_metric(real: pd.DataFrame, sim: pd.DataFrame
                    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-well normalised edit distance + per-pair co-occurrence
    matrices."""
    real_seqs = [_well_sequence(g) for _, g in real.groupby("borehole")]
    sim_seqs = [_well_sequence(g) for _, g in sim.groupby("borehole")]

    # Normalised edit distance from each sim well to its closest real
    # well (cross-well distribution).
    n_pick = min(len(sim_seqs), 200)
    rng = np.random.default_rng(0)
    chosen_sim = rng.choice(len(sim_seqs), size=n_pick, replace=False)
    chosen_real = rng.choice(len(real_seqs),
                              size=min(len(real_seqs), 200), replace=False)
    sim_norm_edits = []
    for i in chosen_sim:
        s = sim_seqs[i]
        best = min(_levenshtein(s, real_seqs[j]) for j in chosen_real)
        sim_norm_edits.append(best / max(len(s), 1))
    # baseline: real well -> closest real well
    real_norm_edits = []
    for i in chosen_real:
        s = real_seqs[i]
        best = min(_levenshtein(s, real_seqs[j])
                   for j in chosen_real if j != i)
        real_norm_edits.append(best / max(len(s), 1))

    # Pair co-occurrence (immediate successor frequency)
    idx = {fm: i for i, fm in enumerate(TARGET_FORMATIONS)}
    n = len(TARGET_FORMATIONS)
    def _pair_matrix(seqs):
        M = np.zeros((n, n))
        for s in seqs:
            for a, b in zip(s[:-1], s[1:]):
                if a in idx and b in idx:
                    M[idx[a], idx[b]] += 1
        rowsum = M.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        return M / rowsum
    M_real = _pair_matrix(real_seqs)
    M_sim = _pair_matrix(sim_seqs)
    frob = float(np.linalg.norm(M_real - M_sim, ord="fro"))

    summary = pd.DataFrame([
        {"metric": "median normalised edit (sim->real)",
         "value": float(np.median(sim_norm_edits))},
        {"metric": "median normalised edit (real->real)",
         "value": float(np.median(real_norm_edits))},
        {"metric": "P75 normalised edit (sim->real)",
         "value": float(np.quantile(sim_norm_edits, 0.75))},
        {"metric": "P75 normalised edit (real->real)",
         "value": float(np.quantile(real_norm_edits, 0.75))},
        {"metric": "Frobenius distance pair-matrix",
         "value": frob},
    ])
    return summary, M_real, M_sim


def plot_sequence(M_real: np.ndarray, M_sim: np.ndarray,
                  summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # left: real matrix
    im0 = axes[0].imshow(M_real, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Real successor frequencies")
    axes[0].set_xticks(range(len(TARGET_FORMATIONS)))
    axes[0].set_xticklabels(TARGET_FORMATIONS, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(TARGET_FORMATIONS)))
    axes[0].set_yticklabels(TARGET_FORMATIONS, fontsize=8)
    axes[0].set_ylabel("upper formation")
    axes[0].set_xlabel("lower formation")
    fig.colorbar(im0, ax=axes[0], fraction=0.04)
    # middle: sim matrix
    im1 = axes[1].imshow(M_sim, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Sim successor frequencies")
    axes[1].set_xticks(range(len(TARGET_FORMATIONS)))
    axes[1].set_xticklabels(TARGET_FORMATIONS, rotation=90, fontsize=8)
    axes[1].set_yticks(range(len(TARGET_FORMATIONS)))
    axes[1].set_yticklabels(TARGET_FORMATIONS, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.04)
    # right: difference
    delta = M_sim - M_real
    amax = max(0.05, float(np.max(np.abs(delta))))
    im2 = axes[2].imshow(delta, cmap="RdBu_r", aspect="auto",
                          vmin=-amax, vmax=amax)
    axes[2].set_title(f"sim - real (Frobenius "
                       f"= {np.linalg.norm(delta, ord='fro'):.2f})")
    axes[2].set_xticks(range(len(TARGET_FORMATIONS)))
    axes[2].set_xticklabels(TARGET_FORMATIONS, rotation=90, fontsize=8)
    axes[2].set_yticks(range(len(TARGET_FORMATIONS)))
    axes[2].set_yticklabels(TARGET_FORMATIONS, fontsize=8)
    fig.colorbar(im2, ax=axes[2], fraction=0.04)
    fig.suptitle("Sequence metric: formation-successor frequencies",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------- position metric

def position_metric(real: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Per-formation top + base depth distributions, real vs sim,
    compared with Wasserstein-1."""
    rows = []
    for fm in TARGET_FORMATIONS:
        r_top = real[real.formation == fm].top.dropna().values
        s_top = sim[sim.formation == fm].top.dropna().values
        r_bot = real[real.formation == fm].bot.dropna().values
        s_bot = sim[sim.formation == fm].bot.dropna().values
        if len(r_top) > 5 and len(s_top) > 5:
            w_top = float(stats.wasserstein_distance(r_top, s_top))
            w_bot = float(stats.wasserstein_distance(r_bot, s_bot))
        else:
            w_top = w_bot = np.nan
        rows.append({
            "formation": fm,
            "n_real": len(r_top), "n_sim": len(s_top),
            "real_top_median": float(np.median(r_top)) if len(r_top) else np.nan,
            "sim_top_median": float(np.median(s_top)) if len(s_top) else np.nan,
            "real_bot_median": float(np.median(r_bot)) if len(r_bot) else np.nan,
            "sim_bot_median": float(np.median(s_bot)) if len(s_bot) else np.nan,
            "wasserstein_top": w_top,
            "wasserstein_bot": w_bot,
        })
    return pd.DataFrame(rows)


def plot_position(real: pd.DataFrame, sim: pd.DataFrame,
                  pos: pd.DataFrame, out_path: Path) -> None:
    # only formations with reasonable counts
    fms = pos[(pos.n_real > 20) & (pos.n_sim > 20)].formation.tolist()
    if not fms:
        fms = TARGET_FORMATIONS
    ncols = 3
    nrows = (len(fms) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows),
                              squeeze=False)
    for i, fm in enumerate(fms):
        ax = axes[i // ncols][i % ncols]
        r_top = real[real.formation == fm].top.dropna().values
        s_top = sim[sim.formation == fm].top.dropna().values
        lo, hi = (min(r_top.min() if len(r_top) else 0,
                      s_top.min() if len(s_top) else 0),
                  max(r_top.max() if len(r_top) else 1,
                      s_top.max() if len(s_top) else 1))
        bins = np.linspace(lo, hi, 30)
        if len(r_top):
            ax.hist(r_top, bins=bins, density=True, color="C0",
                    alpha=0.55, label="Real")
        if len(s_top):
            ax.hist(s_top, bins=bins, density=True, color="C1",
                    alpha=0.55, label="Sim")
        w = pos[pos.formation == fm].wasserstein_top.iloc[0]
        ax.set_title(f"{fm} top  W1={w:.0f}m" if not np.isnan(w)
                     else f"{fm} top")
        ax.set_xlabel("Top depth (m)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    # hide unused axes
    for j in range(len(fms), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Position metric: per-formation top-depth distributions",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path,
                   default=Path("data/clean/samples.parquet"))
    p.add_argument("--geom", type=Path,
                   default=Path("data/clean/formation_geometry.pkl"))
    p.add_argument("--n-sim", type=int, default=1000,
                   help="number of synthetic columns to draw")
    p.add_argument("--n-real-max", type=int, default=None,
                   help="cap the number of real wells (None = all)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("plots/garcon"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path("plots/analysis").mkdir(parents=True, exist_ok=True)

    print("loading NLOG samples...")
    df = pd.read_parquet(args.samples)
    real = real_well_sequences(df, n_max=args.n_real_max)
    print(f"  real wells: {real.borehole.nunique()}, "
          f"{len(real)} (well, formation) intervals")

    print("loading FormationGeometry + sampling synthetic columns...")
    geom = FormationGeometry.load(args.geom)
    sim = sim_well_sequences(geom, n=args.n_sim)
    print(f"  sim wells: {sim.borehole.nunique()}, "
          f"{len(sim)} (well, formation) intervals")

    print("\n[1] extent metric...")
    ext = extent_metric(real, sim)
    plot_extent(ext, args.out_dir / "metric_extent.png")

    print("[2] sequence metric...")
    seq_summary, M_real, M_sim = sequence_metric(real, sim)
    plot_sequence(M_real, M_sim, seq_summary,
                  args.out_dir / "metric_sequence.png")
    print(seq_summary.to_string(index=False))

    print("\n[3] position metric...")
    pos = position_metric(real, sim)
    plot_position(real, sim, pos, args.out_dir / "metric_position.png")

    # combined CSV
    csv_path = Path("plots/analysis/garcon_metrics.csv")
    with open(csv_path, "w") as fh:
        fh.write("# Garzon et al. (2026) metrics, applied to current pipeline.\n")
        fh.write("\n# Extent metric (presence rate per formation)\n")
        ext.to_csv(fh, index=False, lineterminator="\n")
        fh.write("\n# Sequence metric summary\n")
        seq_summary.to_csv(fh, index=False, lineterminator="\n")
        fh.write("\n# Position metric (Wasserstein-1 on top + base depths)\n")
        pos.to_csv(fh, index=False, lineterminator="\n")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
