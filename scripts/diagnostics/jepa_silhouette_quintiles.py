"""
Per-quintile silhouette chart for a JEPA checkpoint.

Splits each synthetic borehole's depth axis into 5 equal windows
(0-20%, 20-40%, 40-60%, 60-80%, 80-100%) and computes the dominant
rock and dominant formation in each window. Then scores silhouette
on the embedding for each (label_set, quintile) combination — 10
silhouettes total, plotted as a grouped bar chart.

Why 5 windows instead of the existing 2x2 cuts: the 2x2 mix three
inconsistent strategies (middle 50%, bottom 25%, deepest formation,
bottom-25% rock). Uniform quintiles give a single consistent view of
how the encoder's discrimination changes with depth.

Usage:
    PYTHONPATH=. python -u scripts/diagnostics/jepa_silhouette_quintiles.py \
        --checkpoint checkpoints/jepa_final.pt \
        --out plots/encoders/jepa_final_silhouette_quintiles.png
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
# poster-legible default fonts
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from encoder.jepa_encoder import load_jepa_checkpoint
from encoder.autoencoder import load_checkpoint as load_ae_checkpoint, standardise
from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import SimConfig
from encoder.encoder_validations.latent_validation import (
    generate_independent_boreholes,
    dominant_in_window,
)
from encoder.encoder_validations.jepa_validation import encode_in_batches_jepa


def encode_ae(model, values: np.ndarray, stats, variables, device,
              batch_size: int = 256) -> np.ndarray:
    """Encode boreholes with the autoencoder's encoder. Appends a normalised
    absolute-depth row when the checkpoint was trained with include_depth=True,
    so a 6-channel AE is compared to JEPA on equal footing."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    include_depth = getattr(model.cfg, "include_depth", False)
    depth_row = (torch.linspace(0.0, 1.0, model.cfg.n_depth, device=device)
                 .view(1, 1, -1) if include_depth else None)
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            if include_depth:
                x = torch.cat([x, depth_row.expand(x.size(0), 1, -1)], dim=1)
            latents.append(model.encoder(x).cpu().numpy())
    return np.concatenate(latents, axis=0)


def _window_labels(n_windows: int) -> list[str]:
    """Generate human-readable labels for N equal windows.
    Example for N=5: ['top 20%', '20-40%', '40-60%', '60-80%',
    'deepest 20%']."""
    if n_windows < 2:
        raise ValueError("n_windows must be >= 2")
    step = 100 // n_windows
    labels = [f"top {step}%"]
    for k in range(1, n_windows - 1):
        labels.append(f"{k * step}-{(k + 1) * step}%")
    labels.append(f"deepest {step}%")
    return labels


def quintile_labels(
    rocks: np.ndarray, forms: np.ndarray, nz: int, n_windows: int = 5,
) -> dict[str, list[list[str]]]:
    """For each borehole, compute the dominant rock and dominant formation
    in each of `n_windows` equal depth windows. Returns dict with keys
    "rock" and "formation", each a list of n_windows label-lists (one
    list per window)."""
    edges = np.linspace(0, nz, n_windows + 1, dtype=int)
    out_rock: list[list[str]] = [[] for _ in range(n_windows)]
    out_form: list[list[str]] = [[] for _ in range(n_windows)]
    n = len(rocks)
    for i in range(n):
        for q in range(n_windows):
            lo, hi = int(edges[q]), int(edges[q + 1])
            out_rock[q].append(dominant_in_window(rocks[i], lo, hi))
            out_form[q].append(dominant_in_window(forms[i], lo, hi))
    return {"rock": out_rock, "formation": out_form}


def compute_silhouette(Z: np.ndarray, labels: list[str],
                       min_count_per_class: int = 15) -> tuple[float, int]:
    """Filter labels to non-noise classes with >= min_count members,
    then compute silhouette. Returns (score, n_used). Returns
    (nan, 0) if fewer than 2 classes survive."""
    counts = Counter(labels)
    keep = [i for i, l in enumerate(labels)
            if counts[l] >= min_count_per_class
            and l not in ("mixed", "other", "rare")]
    if len(keep) < 2 or len({labels[i] for i in keep}) < 2:
        return float("nan"), 0
    sub_Z = Z[keep]
    sub_labels = [labels[i] for i in keep]
    score = silhouette_score(sub_Z, sub_labels, sample_size=2000,
                              random_state=42)
    return float(score), len(keep)


def plot_bars(rock_scores: list[tuple[float, int]],
              form_scores: list[tuple[float, int]],
              window_labels: list[str],
              out_path: Path, title: str) -> None:
    n_windows = len(window_labels)
    # Poster: wide-and-short so the bars fill the full column width and the
    # value labels stay readable.
    width_in = max(11.0, 2.0 * n_windows + 1.0)
    fig, ax = plt.subplots(figsize=(width_in, 3.1))
    x = np.arange(n_windows)
    # Tighter bars when many windows so labels don't crowd.
    width = 0.38 if n_windows <= 6 else 0.42
    rock_vals = [s[0] for s in rock_scores]
    form_vals = [s[0] for s in form_scores]
    rock_ns = [s[1] for s in rock_scores]
    form_ns = [s[1] for s in form_scores]

    bars_rock = ax.bar(x - width / 2, rock_vals, width,
                       label="rock", color="#4C72B0", edgecolor="black",
                       linewidth=0.5)
    bars_form = ax.bar(x + width / 2, form_vals, width,
                       label="formation", color="#d6604d", edgecolor="black",
                       linewidth=0.5)

    # value labels on the bars (single line, large, no n= clutter)
    for bar, val, n in zip(bars_rock, rock_vals, rock_ns):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.012 if val >= 0 else -0.024),
                    f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=14,
                    fontweight="bold")
        else:
            # single-class window: silhouette is undefined. Annotate the
            # bar position so the gap isn't read as a bug.
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "n/a", ha="center", va="bottom",
                    fontsize=13, color="grey", style="italic")
    for bar, val, n in zip(bars_form, form_vals, form_ns):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.012 if val >= 0 else -0.024),
                    f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=14,
                    fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "n/a", ha="center", va="bottom",
                    fontsize=13, color="grey", style="italic")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    # Rotate labels slightly when many windows so they don't overlap.
    rot = 0 if n_windows <= 6 else 30
    ax.set_xticklabels(window_labels, rotation=rot,
                        ha="center" if rot == 0 else "right",
                        fontsize=15 if n_windows <= 6 else 11)
    ax.set_xlim(-0.55, n_windows - 0.45)
    ax.set_ylabel("silhouette (real data)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(min(-0.1, *(v for v in rock_vals + form_vals
                             if not np.isnan(v))) - 0.05,
                 max(0.8, *(v for v in rock_vals + form_vals
                             if not np.isnan(v))) + 0.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_compare(rock_j, rock_a, form_j, form_a,
                 window_labels: list[str], out_path: Path, title: str) -> None:
    """One panel; per depth window, four bars (JEPA/AE x rock/formation).
    Windows with no data for either encoder (e.g. the single-class top 20%)
    are dropped. Colour encodes the encoder, hatch encodes the label set."""
    # Anchor on the rock metric (the headline): drop windows where rock is a
    # single class (silhouette undefined), e.g. the surface top-20% window.
    keep = [i for i in range(len(window_labels))
            if not np.isnan(rock_j[i][0])]
    wl = [window_labels[i] for i in keep]
    n = len(keep)

    # Each score is (mean, sd) when averaged over runs; sd (if present) is
    # drawn as an error bar.
    def vals(s):
        return [s[i][0] for i in keep]

    def errs(s):
        return [0.0 if (len(s[i]) < 2 or np.isnan(s[i][1])) else s[i][1]
                for i in keep]

    cj, ca = "#4C72B0", "#dd8452"          # JEPA blue / AE orange
    x = np.arange(n)
    w = 0.20
    # Legend outside (right); value labels are rotated vertical so the four
    # close bars per group never overlap.
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    series = [
        (rock_j, x - 1.5 * w, cj, None, "JEPA · rock"),
        (rock_a, x - 0.5 * w, ca, None, "AE · rock"),
        (form_j, x + 0.5 * w, cj, "////", "JEPA · formation"),
        (form_a, x + 1.5 * w, ca, "////", "AE · formation"),
    ]
    for raw, xpos, color, hatch, lab in series:
        v, e = vals(raw), errs(raw)
        bars = ax.bar(xpos, [0 if np.isnan(t) else t for t in v], w,
                      yerr=e, capsize=2.5, ecolor="#333",
                      error_kw={"linewidth": 1.0},
                      color=color, edgecolor="black", linewidth=0.5,
                      hatch=hatch, label=lab)
        for b, t, er in zip(bars, v, e):
            if not np.isnan(t):
                top = (t + er) if t >= 0 else (t - er)
                ax.text(b.get_x() + b.get_width() / 2,
                        top + (0.006 if t >= 0 else -0.006), f"{t:.2f}",
                        ha="center", va="bottom" if t >= 0 else "top",
                        rotation=90, fontsize=7.5, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(wl, fontsize=10)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.set_ylabel("silhouette score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.25, 0.50)
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5,
              ncol=1, handlelength=1.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.10, right=0.78, top=0.88, bottom=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/jepa_final.pt"))
    p.add_argument("--distributions", type=Path,
                   default=Path("data/clean/distributions.pkl"))
    p.add_argument("--n-boreholes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=5,
                   help="number of equal-depth windows to split each "
                        "borehole into. 5 = quintiles, 10 = deciles.")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG path. Defaults to "
                        "plots/encoders/jepa_silhouette_n{N}.png.")
    p.add_argument("--ae-checkpoint", type=Path,
                   default=Path("checkpoints/ae_depth.pt"),
                   help="autoencoder checkpoint to compare against JEPA "
                        "(same 6 channels). Pass 'none' to skip the comparison.")
    args = p.parse_args()
    n_windows = args.n_windows
    window_labels = _window_labels(n_windows)
    out_path = (args.out if args.out is not None
                else Path(f"plots/encoders/jepa_silhouette_n{n_windows}.png"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"loading checkpoint: {args.checkpoint}")
    model, stats, variables = load_jepa_checkpoint(args.checkpoint,
                                                    device=device)
    model.eval()
    print(f"  variables: {variables}  latent={model.cfg.latent_dim}  "
          f"include_depth={getattr(model.cfg, 'include_depth', False)}")

    bank = DistributionBank.load(args.distributions)
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    sim_cfg = SimConfig()
    rng = np.random.default_rng(args.seed)

    print(f"\ngenerating {args.n_boreholes} boreholes (nz={sim_cfg.n_depth}) ...")
    values, rocks, forms, depth_axis = generate_independent_boreholes(
        n_boreholes=args.n_boreholes, variables=variables,
        bank=bank, geometry=geom, sim_cfg=sim_cfg, rng=rng,
    )
    print(f"  values: {values.shape}  NaN: {np.isnan(values).mean():.2%}")

    print("encoding (JEPA) ...")
    Z = encode_in_batches_jepa(model, values, stats, variables, device)
    print(f"  latents: {Z.shape}")

    Z_ae = None
    use_ae = str(args.ae_checkpoint).lower() != "none" and args.ae_checkpoint.exists()
    if use_ae:
        print(f"loading AE checkpoint: {args.ae_checkpoint}")
        ae_model, ae_stats, ae_vars = load_ae_checkpoint(args.ae_checkpoint,
                                                         device=device)
        print(f"  AE variables: {ae_vars}  latent={ae_model.cfg.latent_dim}  "
              f"include_depth={getattr(ae_model.cfg, 'include_depth', False)}")
        print("encoding (AE) ...")
        Z_ae = encode_ae(ae_model, values, ae_stats, ae_vars, device)
        print(f"  AE latents: {Z_ae.shape}")

    print(f"\ncomputing per-window labels (n_windows={n_windows}) ...")
    nz = values.shape[-1]
    labels = quintile_labels(rocks, forms, nz, n_windows=n_windows)
    edges = np.linspace(0, nz, n_windows + 1, dtype=int)
    for q in range(n_windows):
        lo, hi = int(edges[q]), int(edges[q + 1])
        depth_lo = float(depth_axis[lo])
        depth_hi = float(depth_axis[min(hi, nz - 1)])
        print(f"  q{q} {window_labels[q]:>14s}  "
              f"cells {lo:>3d}-{hi:>3d}  "
              f"depth {depth_lo:>5.0f}-{depth_hi:>5.0f} m")
        for name, lab_lists in labels.items():
            top3 = Counter(lab_lists[q]).most_common(3)
            top3_str = ", ".join(f"{k}={v}" for k, v in top3)
            print(f"      {name:9s}  top3: {top3_str}")

    print("\nsilhouette per (label, window):")
    rock_scores, form_scores = [], []
    ae_rock_scores, ae_form_scores = [], []
    for q in range(n_windows):
        r_score, r_n = compute_silhouette(Z, labels["rock"][q])
        f_score, f_n = compute_silhouette(Z, labels["formation"][q])
        rock_scores.append((r_score, r_n))
        form_scores.append((f_score, f_n))
        line = (f"  {window_labels[q]:>14s}   JEPA rock={r_score:+.3f} "
                f"(n={r_n})   formation={f_score:+.3f} (n={f_n})")
        if Z_ae is not None:
            ar, _ = compute_silhouette(Z_ae, labels["rock"][q])
            af, _ = compute_silhouette(Z_ae, labels["formation"][q])
            ae_rock_scores.append((ar, r_n))
            ae_form_scores.append((af, f_n))
            line += f"   |   AE rock={ar:+.3f}  formation={af:+.3f}"
        print(line)

    granularity = ("quintile" if n_windows == 5
                    else "decile" if n_windows == 10
                    else f"{n_windows}-window")
    if Z_ae is not None:
        title = (f"Per-{granularity} silhouette — JEPA vs AE "
                 f"(n={args.n_boreholes}, depth 0–{int(depth_axis[-1])} m)")
        plot_compare(rock_scores, ae_rock_scores, form_scores, ae_form_scores,
                     window_labels, out_path, title)
    else:
        title = (f"Per-{granularity} silhouette — n={args.n_boreholes} boreholes, "
                 f"depth 0–{int(depth_axis[-1])} m")
        plot_bars(rock_scores, form_scores, window_labels, out_path, title)


if __name__ == "__main__":
    main()
