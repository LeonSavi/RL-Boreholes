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
import numpy as np
import torch
from sklearn.metrics import silhouette_score

from encoder.jepa_encoder import load_jepa_checkpoint
from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import SimConfig
from encoder.encoder_validations.latent_validation import (
    generate_independent_boreholes,
    dominant_in_window,
)
from encoder.encoder_validations.jepa_validation import encode_in_batches_jepa


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
    # Wider figure when many windows; minimum 9 inches.
    width_in = max(9.0, 1.2 * n_windows + 3.0)
    fig, ax = plt.subplots(figsize=(width_in, 5))
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

    # value labels on the bars
    for bar, val, n in zip(bars_rock, rock_vals, rock_ns):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 if val >= 0 else -0.02),
                    f"{val:+.3f}\nn={n}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=7.5)
        else:
            # single-class window: silhouette is undefined. Annotate the
            # bar position so the gap isn't read as a bug.
            ax.text(bar.get_x() + bar.get_width() / 2, 0.015,
                    "n/a\n(single\nclass)", ha="center", va="bottom",
                    fontsize=7, color="grey", style="italic")
    for bar, val, n in zip(bars_form, form_vals, form_ns):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 if val >= 0 else -0.02),
                    f"{val:+.3f}\nn={n}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=7.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.015,
                    "n/a\n(single\nclass)", ha="center", va="bottom",
                    fontsize=7, color="grey", style="italic")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    # Rotate labels slightly when many windows so they don't overlap.
    rot = 0 if n_windows <= 6 else 30
    ax.set_xticklabels(window_labels, rotation=rot,
                        ha="center" if rot == 0 else "right",
                        fontsize=9 if n_windows <= 6 else 8)
    ax.set_xlim(-0.55, n_windows - 0.45)
    ax.set_ylabel("silhouette score (real-data validation)")
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

    print("encoding ...")
    Z = encode_in_batches_jepa(model, values, stats, variables, device)
    print(f"  latents: {Z.shape}")

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
    rock_scores = []
    form_scores = []
    for q in range(n_windows):
        r_score, r_n = compute_silhouette(Z, labels["rock"][q])
        f_score, f_n = compute_silhouette(Z, labels["formation"][q])
        rock_scores.append((r_score, r_n))
        form_scores.append((f_score, f_n))
        print(f"  {window_labels[q]:>14s}   "
              f"rock={r_score:+.3f} (n={r_n})   "
              f"formation={f_score:+.3f} (n={f_n})")

    granularity = ("quintile" if n_windows == 5
                    else "decile" if n_windows == 10
                    else f"{n_windows}-window")
    title = (f"Per-{granularity} silhouette — {args.checkpoint.name}\n"
             f"(real-data validation, n_boreholes={args.n_boreholes}, "
             f"depth axis 0–{int(depth_axis[-1])} m)")
    plot_bars(rock_scores, form_scores, window_labels, out_path, title)


if __name__ == "__main__":
    main()
