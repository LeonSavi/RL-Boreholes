"""Thesis-grade evaluation of the trained encoders (autoencoder + JEPA).

For each available checkpoint (`checkpoints/ae.pt`, `checkpoints/jepa.pt`):
  - encode the SAME shared set of N independent borehole columns
  - reduce latents to 2D (UMAP) with four label colourings
  - compute silhouette scores per (encoder × label set)
  - for the autoencoder also report per-variable reconstruction error

Outputs (plots/encoders/analysis/):
  ENCODER_REPORT.md
  silhouette_summary.csv
  ae_reconstruction.csv                   (per-variable SmoothL1 + R²)
  01_ae_umap.png                          (2×2 panel — same plot as latent_validation)
  02_jepa_umap.png                        (2×2 panel — same plot as jepa_validation)
  03_silhouette_comparison.png            (grouped bar chart, AE vs JEPA)
  04_ae_reconstruction.png                (per-variable scatter recon vs truth)

The script runs both encoder validations from the SAME shared borehole
set so silhouette comparisons are apples-to-apples.  Both checkpoints are
optional; the script reports on whichever ones it finds.

Run after training:
    python analysis_encoders.py
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from encoder.autoencoder import load_checkpoint as load_ae_checkpoint
from encoder.autoencoder import standardise
from encoder.jepa_encoder import load_jepa_checkpoint
from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import SimConfig
from encoder.encoder_validations.latent_validation import (
    generate_independent_boreholes,
    compute_all_labels,
    reduce_to_2d,
    formation_colour,
)


DEFAULT_AE       = Path("checkpoints/ae.pt")
DEFAULT_JEPA     = Path("checkpoints/jepa.pt")
DEFAULT_DISTR    = Path("data/clean/distributions.pkl")
DEFAULT_GEOM     = Path("data/clean/formation_geometry.pkl")
DEFAULT_OUT      = Path("plots/encoders/analysis")
DEFAULT_N        = 3000
DEFAULT_SEED     = 0


# ─── encoder loading + encoding ──────────────────────────────────────────

def encode_with_ae(model, values: np.ndarray, stats, variables, device,
                   batch_size: int = 256) -> np.ndarray:
    """Pass (N, V, D) → AE encoder, return (N, latent_dim) array."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            z = model.encoder(x).cpu().numpy()
            latents.append(z)
    return np.concatenate(latents, axis=0)


def encode_with_jepa(model, values: np.ndarray, stats, variables, device,
                     batch_size: int = 256) -> np.ndarray:
    """Pass (N, V, D) → JEPA target-encoder mean-pooled, return (N, dim)."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            tokens = model.target_encoder(x)              # (B, T, D)
            z = tokens.mean(dim=1).cpu().numpy()
            latents.append(z)
    return np.concatenate(latents, axis=0)


def reconstruct_with_ae(model, values: np.ndarray, stats, variables, device,
                        batch_size: int = 256
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Pass (N, V, D) → AE round-trip, return (truth_std, recon) both in
    standardised units, so per-variable errors are directly comparable."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    truth = std.copy()
    recon_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            r, _ = model(x)
            recon_chunks.append(r.cpu().numpy())
    return truth, np.concatenate(recon_chunks, axis=0)


# ─── silhouette metric helper ────────────────────────────────────────────

def silhouette_table(Z: np.ndarray, label_sets: dict[str, list[str]]
                     ) -> dict[str, tuple[float, int]]:
    """Compute silhouette per label set.  Returns {name: (score, n_used)}.
    Filters labels appearing in fewer than 15 boreholes and the "mixed"/
    "other"/"rare" buckets, matching the existing validation scripts."""
    from sklearn.metrics import silhouette_score
    result: dict[str, tuple[float, int]] = {}
    for name, labels in label_sets.items():
        counts = Counter(labels)
        keep = [i for i, l in enumerate(labels)
                if counts[l] >= 15 and l not in ("mixed", "other", "rare")]
        if len(set(labels[i] for i in keep)) < 2:
            result[name] = (float("nan"), len(keep))
            continue
        sub_Z = Z[keep]
        sub_labels = [labels[i] for i in keep]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = silhouette_score(sub_Z, sub_labels,
                                      sample_size=2000, random_state=42)
        result[name] = (float(score), len(keep))
    return result


# ─── plots ────────────────────────────────────────────────────────────────

def _plot_one_panel(ax, Z2: np.ndarray, labels: list[str], title: str,
                     skip_labels: set[str]) -> None:
    """Scatter the 2D embedding coloured by `labels`.  Repurposed from
    latent_validation.plot_four_panels but kept inline so we can call it
    independently per encoder."""
    classes = sorted(set(labels) - skip_labels)
    cmap = plt.get_cmap("tab20")
    colour_map = {c: cmap(i % 20) for i, c in enumerate(classes)}

    for c in classes:
        mask = np.array([l == c for l in labels])
        if not mask.any():
            continue
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=6, alpha=0.7,
                    color=colour_map[c], edgecolor="none", label=c)
    if any(l in skip_labels for l in labels):
        mask = np.array([l in skip_labels for l in labels])
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=4, alpha=0.25,
                    color="#bbbbbb", edgecolor="none")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    if len(classes) <= 12:
        ax.legend(fontsize=7, markerscale=2, loc="best")


def plot_umap_grid(Z2: np.ndarray, label_sets: dict[str, list[str]],
                    silhouettes: dict[str, tuple[float, int]],
                    out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()
    panel_order = ["rock (middle 50%)", "formation (middle 50%)",
                    "formation (deepest)", "rock (bottom 25%)"]
    skip = {"mixed", "other", "rare"}
    for ax, name in zip(axes, panel_order):
        if name not in label_sets:
            ax.set_visible(False)
            continue
        score, n = silhouettes.get(name, (float("nan"), 0))
        sub_title = (f"{name}\n"
                     f"silhouette = {score:+.3f}  (n={n})"
                     if not np.isnan(score) else
                     f"{name}\n(not enough labels)")
        _plot_one_panel(ax, Z2, label_sets[name], sub_title, skip_labels=skip)
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_silhouette_comparison(table: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: per label set, bars for AE and JEPA silhouette."""
    if table.empty:
        return
    encoders = sorted(table["encoder"].unique())
    labelsets = list(table["label_set"].unique())
    width = 0.4
    x = np.arange(len(labelsets))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colours = {"AE": "#1f77b4", "JEPA": "#d62728"}
    for i, enc in enumerate(encoders):
        sub = table[table["encoder"] == enc].set_index("label_set")\
                .reindex(labelsets)
        heights = sub["silhouette"].values
        ax.bar(x + (i - 0.5) * width, heights, width,
                color=colours.get(enc, "#888888"),
                alpha=0.85, edgecolor="black", linewidth=0.3,
                label=enc)
        for j, h in enumerate(heights):
            if np.isnan(h):
                continue
            ax.text(x[j] + (i - 0.5) * width,
                     h + (0.01 if h >= 0 else -0.02),
                     f"{h:+.3f}", ha="center",
                     fontsize=8,
                     va="bottom" if h >= 0 else "top")
    ax.set_xticks(x)
    ax.set_xticklabels(labelsets, rotation=20, ha="right")
    ax.set_ylabel("silhouette score")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Encoder silhouette comparison — "
                  "higher = labels separate better in latent space",
                  fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ae_reconstruction(truth: np.ndarray, recon: np.ndarray,
                            variables: list[str], out_path: Path,
                            n_points: int = 5000) -> None:
    """Per-variable scatter of truth vs reconstruction (sub-sampled),
    with the unity line and the R² value annotated."""
    rng = np.random.default_rng(0)
    n, V, D = truth.shape
    idx = rng.choice(n * D, size=min(n_points, n * D), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax, var, i in zip(axes, variables, range(V)):
        t = truth[:, i, :].ravel()[idx]
        r = recon[:, i, :].ravel()[idx]
        mask = np.isfinite(t) & np.isfinite(r)
        t, r = t[mask], r[mask]
        if len(t) < 50:
            ax.set_visible(False)
            continue
        ax.scatter(t, r, s=4, alpha=0.3, color="#2166ac")
        lo, hi = np.percentile(np.concatenate([t, r]), [1, 99])
        ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1)
        ss_res = float(np.sum((t - r) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2)) or 1
        r2 = 1 - ss_res / ss_tot
        smoothl1 = float(np.mean(np.where(np.abs(t - r) < 1.0,
                                            0.5 * (t - r) ** 2,
                                            np.abs(t - r) - 0.5)))
        ax.set_title(f"{var}\nR² = {r2:.3f}, SmoothL1 = {smoothl1:.4f}",
                      fontsize=10)
        ax.set_xlabel("truth (standardised)")
        ax.set_ylabel("recon (standardised)")
        ax.grid(alpha=0.25)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if V < 6:
        for ax in axes[V:]:
            ax.set_visible(False)
    fig.suptitle("Autoencoder reconstruction — per-variable truth vs recon",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── report ──────────────────────────────────────────────────────────────

def write_encoder_report(summary: pd.DataFrame,
                          ae_recon: pd.DataFrame | None,
                          ae_present: bool,
                          jepa_present: bool,
                          n_boreholes: int,
                          out_path: Path) -> None:
    L = []
    a = L.append
    a("# Encoder evaluation report")
    a("")
    a(f"Validation set: **{n_boreholes:,} independent borehole columns** "
      "(each column has its own freshly-sampled stratigraphic stack).")
    a("")
    encoders = []
    if ae_present: encoders.append("autoencoder")
    if jepa_present: encoders.append("JEPA")
    a("Encoders evaluated: " + ", ".join(encoders) + ".")
    a("")
    a("## 1. UMAP latent space, coloured by label scheme")
    a("")
    if ae_present:
        a("![AE UMAP](01_ae_umap.png)")
        a("")
    if jepa_present:
        a("![JEPA UMAP](02_jepa_umap.png)")
        a("")
    a("Each 2×2 panel reuses the same UMAP projection of the encoder's "
      "latents, coloured by four different label schemes:")
    a("")
    a("- **rock (middle 50%)** — dominant rock in the central depth window")
    a("- **formation (middle 50%)** — dominant formation in the same window")
    a("- **formation (deepest)** — formation reached at total depth")
    a("- **rock (bottom 25%)** — dominant rock in the deep window")
    a("")
    a("Silhouette score reads: > +0.3 strong, +0.1-0.3 modest, ~0 random.")
    a("")
    a("## 2. Silhouette comparison")
    a("")
    a("![silhouette](03_silhouette_comparison.png)")
    a("")
    if len(summary):
        piv = summary.pivot(index="label_set", columns="encoder",
                             values="silhouette")
        a("Per-label-set silhouette:")
        a("")
        cols = piv.columns.tolist()
        a("| label set | " + " | ".join(cols) + " |")
        a("|---" * (len(cols) + 1) + "|")
        for ls, row in piv.iterrows():
            cells = [ls] + [f"{v:+.3f}" if not pd.isna(v) else "—"
                              for v in row]
            a("| " + " | ".join(cells) + " |")
        a("")
    if ae_present and ae_recon is not None and len(ae_recon):
        a("## 3. Autoencoder reconstruction quality")
        a("")
        a("![AE reconstruction](04_ae_reconstruction.png)")
        a("")
        a("Per-variable reconstruction error (standardised units):")
        a("")
        a("| variable | R² | SmoothL1 | n cells |")
        a("|---|---|---|---|")
        for _, r in ae_recon.iterrows():
            a(f"| {r['variable']} | {r['R2']:.3f} | "
              f"{r['SmoothL1']:.4f} | {int(r['n']):,} |")
        a("")
        a("R² ≈ 1 and SmoothL1 ≪ 1 indicate the autoencoder can "
          "reconstruct each channel from the latent.  Drops on a "
          "specific variable point to information loss in the bottleneck.")
        a("")
    a("## 4. Files")
    a("")
    a("- `silhouette_summary.csv` — silhouette per (encoder × label set)")
    if ae_present:
        a("- `ae_reconstruction.csv` — per-variable R² and SmoothL1 for the AE")
    a("- `01_ae_umap.png`, `02_jepa_umap.png` — per-encoder UMAP grids")
    a("- `03_silhouette_comparison.png` — grouped bar chart")
    if ae_present:
        a("- `04_ae_reconstruction.png` — per-variable truth-vs-recon scatter")
    a("")
    out_path.write_text("\n".join(L))
    print(f"  wrote {out_path}")


# ─── orchestration ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ae",   type=Path, default=DEFAULT_AE,
                     help=f"autoencoder checkpoint (default {DEFAULT_AE})")
    ap.add_argument("--jepa", type=Path, default=DEFAULT_JEPA,
                     help=f"JEPA checkpoint (default {DEFAULT_JEPA})")
    ap.add_argument("--distributions", type=Path, default=DEFAULT_DISTR)
    ap.add_argument("--geometry", type=Path, default=DEFAULT_GEOM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boreholes", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ae_present   = args.ae.exists()
    jepa_present = args.jepa.exists()
    if not (ae_present or jepa_present):
        raise FileNotFoundError(
            f"neither checkpoint exists: {args.ae} or {args.jepa}")
    if not ae_present:
        print(f"  note: {args.ae} not found — skipping AE eval")
    if not jepa_present:
        print(f"  note: {args.jepa} not found — skipping JEPA eval")

    print(f"loading simulator artifacts ...")
    bank = DistributionBank.load(args.distributions)
    geom = FormationGeometry.load(args.geometry)
    sim_cfg = SimConfig()

    print(f"\ngenerating {args.n_boreholes} independent boreholes (shared "
          "between both encoders) ...")
    rng = np.random.default_rng(args.seed)
    values, rocks, forms, _ = generate_independent_boreholes(
        n_boreholes=args.n_boreholes,
        variables=list(sim_cfg.variables),
        bank=bank, geometry=geom, sim_cfg=sim_cfg, rng=rng,
    )
    nz = values.shape[-1]
    print(f"  values shape: {values.shape}")

    label_sets = compute_all_labels(rocks, forms, nz=nz)
    for name, labels in label_sets.items():
        top = Counter(labels).most_common(3)
        print(f"  {name:28s} top3: {top}")

    summary_rows = []
    ae_recon_rows = []

    # ── autoencoder ───────────────────────────────────────────────────
    if ae_present:
        print(f"\nloading AE: {args.ae}")
        ae, ae_stats, ae_vars = load_ae_checkpoint(args.ae, device=device)
        print(f"  AE variables: {ae_vars}")

        Z_ae = encode_with_ae(ae, values, ae_stats, ae_vars, device)
        print(f"  AE latents: {Z_ae.shape}")
        print(f"  reducing to 2D ...")
        Z2_ae = reduce_to_2d(Z_ae, method="umap")
        sil_ae = silhouette_table(Z_ae, label_sets)
        for name, (score, n) in sil_ae.items():
            summary_rows.append({"encoder": "AE", "label_set": name,
                                  "silhouette": score, "n_used": n})
        plot_umap_grid(Z2_ae, label_sets, sil_ae,
                        out_path=args.out / "01_ae_umap.png",
                        title=f"Autoencoder UMAP — {args.n_boreholes} boreholes")

        print(f"  computing reconstruction quality ...")
        truth, recon = reconstruct_with_ae(ae, values, ae_stats, ae_vars,
                                            device)
        plot_ae_reconstruction(truth, recon, ae_vars,
                                args.out / "04_ae_reconstruction.png")
        for i, v in enumerate(ae_vars):
            t = truth[:, i, :].ravel()
            r = recon[:, i, :].ravel()
            mask = np.isfinite(t) & np.isfinite(r)
            t, r = t[mask], r[mask]
            ss_res = float(np.sum((t - r) ** 2))
            ss_tot = float(np.sum((t - np.mean(t)) ** 2)) or 1
            r2 = 1 - ss_res / ss_tot
            smoothl1 = float(np.mean(np.where(np.abs(t - r) < 1.0,
                                                0.5 * (t - r) ** 2,
                                                np.abs(t - r) - 0.5)))
            ae_recon_rows.append({
                "variable": v, "R2": round(r2, 4),
                "SmoothL1": round(smoothl1, 4), "n": int(len(t)),
            })

    # ── JEPA ──────────────────────────────────────────────────────────
    if jepa_present:
        print(f"\nloading JEPA: {args.jepa}")
        jepa, jepa_stats, jepa_vars = load_jepa_checkpoint(
            args.jepa, device=device)
        print(f"  JEPA variables: {jepa_vars}")
        Z_jepa = encode_with_jepa(jepa, values, jepa_stats, jepa_vars, device)
        print(f"  JEPA latents: {Z_jepa.shape}")
        print(f"  reducing to 2D ...")
        Z2_jepa = reduce_to_2d(Z_jepa, method="umap")
        sil_jepa = silhouette_table(Z_jepa, label_sets)
        for name, (score, n) in sil_jepa.items():
            summary_rows.append({"encoder": "JEPA", "label_set": name,
                                  "silhouette": score, "n_used": n})
        plot_umap_grid(Z2_jepa, label_sets, sil_jepa,
                        out_path=args.out / "02_jepa_umap.png",
                        title=f"JEPA UMAP — {args.n_boreholes} boreholes")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "silhouette_summary.csv", index=False)
    print(f"\n  wrote silhouette_summary.csv  ({len(summary)} rows)")

    ae_recon = pd.DataFrame(ae_recon_rows) if ae_recon_rows else None
    if ae_recon is not None:
        ae_recon.to_csv(args.out / "ae_reconstruction.csv", index=False)
        print(f"  wrote ae_reconstruction.csv  ({len(ae_recon)} rows)")

    plot_silhouette_comparison(summary, args.out
                                / "03_silhouette_comparison.png")

    print("\nwriting ENCODER_REPORT.md ...")
    write_encoder_report(summary, ae_recon, ae_present, jepa_present,
                          n_boreholes=args.n_boreholes,
                          out_path=args.out / "ENCODER_REPORT.md")

    print(f"\n→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()
