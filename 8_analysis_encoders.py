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
    python 8_analysis_encoders.py
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
# Thesis-legible defaults: figures are shrunk to ~column width on the page, so
# native figsizes are kept close to the display width with modest fonts (text
# renders near body size, ~11 pt) and figures are saved at 300 dpi.
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})

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


DEFAULT_AE                = Path("checkpoints/ae.pt")
DEFAULT_JEPA              = Path("checkpoints/jepa.pt")
DEFAULT_AE_FORMATION      = Path("checkpoints/ae_formation.pt")
DEFAULT_JEPA_FORMATION    = Path("checkpoints/jepa_formation.pt")
DEFAULT_DISTR             = Path("data/clean/distributions.pkl")
DEFAULT_GEOM              = Path("data/clean/formation_geometry.pkl")
DEFAULT_OUT               = Path("plots/encoders/analysis")
DEFAULT_N                 = 3000
DEFAULT_SEED              = 0

# Plot colours per encoder. Same hue per family (AE blue-ish, JEPA red-ish);
# darker = rock-resolution baseline, lighter = formation-resolution variant.
ENCODER_COLOURS = {
    "AE":           "#1f77b4",      # rock-resolution AE baseline
    "AE_formation": "#9ecae1",      # formation-resolution AE
    "JEPA":         "#d62728",      # rock-resolution JEPA baseline
    "JEPA_formation": "#fb6a4a",    # formation-resolution JEPA
}


# ─── encoder loading + encoding ──────────────────────────────────────────

def encode_with_ae(model, values: np.ndarray, stats, variables, device,
                   batch_size: int = 256, noise_sd: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    """Pass (N, V, D) → AE encoder, return (N, latent_dim) array.

    Appends a normalised absolute-depth row when the checkpoint was trained
    with `include_depth=True`, matching the encoder's channel count (so a
    6-channel AE is compared to JEPA on equal footing).  `noise_sd` adds
    Gaussian noise to the standardised input (for the robustness test)."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    if noise_sd:
        std = (std + np.random.default_rng(seed)
               .normal(0.0, noise_sd, std.shape)).astype(np.float32)
    include_depth = getattr(model.cfg, "include_depth", False)
    if include_depth:
        depth_row = (
            torch.linspace(0.0, 1.0, model.cfg.n_depth, device=device)
            .view(1, 1, -1)
        )
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            if include_depth:
                x = torch.cat([x, depth_row.expand(x.size(0), 1, -1)], dim=1)
            z = model.encoder(x).cpu().numpy()
            latents.append(z)
    return np.concatenate(latents, axis=0)


def encode_with_jepa(model, values: np.ndarray, stats, variables, device,
                     batch_size: int = 256, noise_sd: float = 0.0,
                     seed: int = 0) -> np.ndarray:
    """Pass (N, V, D) → JEPA target-encoder mean-pooled, return (N, dim).

    Appends a normalised absolute-depth row when the checkpoint was
    trained with `include_depth=True` so the input matches the encoder's
    expected channel count.  `noise_sd` adds Gaussian noise to the
    standardised input (for the robustness test).
    """
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    if noise_sd:
        std = (std + np.random.default_rng(seed)
               .normal(0.0, noise_sd, std.shape)).astype(np.float32)
    include_depth = getattr(model.cfg, "include_depth", False)
    if include_depth:
        depth_row = (
            torch.linspace(0.0, 1.0, model.cfg.n_depth, device=device)
            .view(1, 1, -1)
        )
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            if include_depth:
                x = torch.cat([x, depth_row.expand(x.size(0), 1, -1)], dim=1)
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
    n_var = std.shape[1]
    include_depth = getattr(model.cfg, "include_depth", False)
    if include_depth:
        depth_row = (
            torch.linspace(0.0, 1.0, model.cfg.n_depth, device=device)
            .view(1, 1, -1)
        )
    recon_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i:i + batch_size]).to(device)
            if include_depth:
                x = torch.cat([x, depth_row.expand(x.size(0), 1, -1)], dim=1)
            r, _ = model(x)
            # drop the reconstructed depth channel so per-variable errors
            # compare the 5 wireline channels against `truth`.
            recon_chunks.append(r[:, :n_var, :].cpu().numpy())
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


def _filter_for_probe(Z: np.ndarray, labels: list[str]):
    """Keep classes with >=15 members, drop mixed/other/rare. Returns
    (sub_Z, integer-encoded y) or (None, None) if <2 usable classes."""
    counts = Counter(labels)
    keep = [i for i, l in enumerate(labels)
            if counts[l] >= 15 and l not in ("mixed", "other", "rare")]
    if len(set(labels[i] for i in keep)) < 2:
        return None, None
    classes = sorted(set(labels[i] for i in keep))
    enc = {c: k for k, c in enumerate(classes)}
    sub_Z = Z[keep]
    y = np.array([enc[labels[i]] for i in keep])
    return sub_Z, y


def knn_label_accuracy(Z: np.ndarray, labels: list[str], k: int = 5,
                       cv: int = 5) -> float:
    """Stratified k-fold k-NN classification accuracy of labels from latents.
    A direct read of 'do nearest neighbours in latent space share geology?'."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    sub_Z, y = _filter_for_probe(Z, labels)
    if sub_Z is None:
        return float("nan")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, sub_Z, y, cv=skf, scoring="accuracy")
    return float(scores.mean())


def probe_accuracy(Z: np.ndarray, labels: list[str], cv: int = 5) -> float:
    """Stratified k-fold linear (logistic-regression) probe accuracy: tests
    how linearly separable the labels are in latent space."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    sub_Z, y = _filter_for_probe(Z, labels)
    if sub_Z is None:
        return float("nan")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, sub_Z, y, cv=skf, scoring="accuracy")
    return float(scores.mean())


def few_shot_probe(Z: np.ndarray, labels: list[str], n_per_class: int = 3,
                   repeats: int = 30, seed: int = 42) -> float:
    """Low-label logistic-probe accuracy: train on only `n_per_class` labelled
    boreholes per class, test on the rest, averaged over `repeats` random
    draws. A latent that organises geology well needs fewer labels, so this
    exposes representation-quality gaps that full-data accuracy (saturated near
    1.0) hides."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    sub_Z, y = _filter_for_probe(Z, labels)
    if sub_Z is None:
        return float("nan")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if min((y == c).sum() for c in classes) <= n_per_class:
        return float("nan")
    accs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(repeats):
            tr = np.concatenate([rng.choice(np.where(y == c)[0],
                                            size=n_per_class, replace=False)
                                 for c in classes])
            mask = np.ones(len(y), bool); mask[tr] = False
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000))
            clf.fit(sub_Z[tr], y[tr])
            accs.append(clf.score(sub_Z[mask], y[mask]))
    return float(np.mean(accs))


# ─── plots ────────────────────────────────────────────────────────────────

def _plot_one_panel(ax, Z2: np.ndarray, labels: list[str], title: str,
                     skip_labels: set[str], show_legend: bool = True,
                     title_fs: int = 12, point_s: int = 8) -> None:
    """Scatter the 2D embedding coloured by `labels`.  Repurposed from
    latent_validation.plot_four_panels but kept inline so we can call it
    independently per encoder.  `show_legend=False` suppresses the per-panel
    legend (for a single shared legend); `title_fs`/`point_s` tune sizes."""
    classes = sorted(set(labels) - skip_labels)
    cmap = plt.get_cmap("tab20")
    colour_map = {c: cmap(i % 20) for i, c in enumerate(classes)}

    for c in classes:
        mask = np.array([l == c for l in labels])
        if not mask.any():
            continue
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=point_s, alpha=0.75,
                    color=colour_map[c], edgecolor="none", label=c)
    if any(l in skip_labels for l in labels):
        mask = np.array([l in skip_labels for l in labels])
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=max(10, point_s // 2), alpha=0.22,
                    color="#bbbbbb", edgecolor="none")
    ax.set_title(title, fontsize=title_fs, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    if show_legend and len(classes) <= 12:
        ax.legend(fontsize=8, markerscale=2.0, loc="best", framealpha=0.9)


def plot_umap_grid(Z2: np.ndarray, label_sets: dict[str, list[str]],
                    silhouettes: dict[str, tuple[float, int]],
                    out_path: Path, title: str) -> None:
    # Show the two strong mid-column panels large and legible; the deep-cut
    # behaviour is conveyed quantitatively by the per-quintile silhouette
    # figure instead of cramped extra UMAP panels.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
    axes = axes.flatten()
    panel_order = ["rock (middle 50%)", "formation (middle 50%)"]
    skip = {"mixed", "other", "rare"}
    for ax, name in zip(axes, panel_order):
        if name not in label_sets:
            ax.set_visible(False)
            continue
        score, n = silhouettes.get(name, (float("nan"), 0))
        sub_title = (f"{name}\nsilhouette = {score:+.2f}"
                     if not np.isnan(score) else
                     f"{name}\n(not enough labels)")
        _plot_one_panel(ax, Z2, label_sets[name], sub_title, skip_labels=skip)
    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_jepa_vs_ae_umap(z2_jepa: np.ndarray, z2_ae: np.ndarray,
                         labels_rock: list[str],
                         sil_jepa: float, sil_ae: float,
                         out_path: Path,
                         suptitle: str = "Same boreholes, two encoders, "
                                         "coloured by rock (middle 50%)") -> None:
    """Side-by-side UMAP of the SAME boreholes under each encoder, coloured by
    rock. JEPA and the AE baseline separate the lithologies about equally;
    `suptitle` lets callers relabel it (e.g. for the real-well version)."""
    from matplotlib.lines import Line2D
    skip = {"mixed", "other", "rare"}
    # One shared legend on the right instead of a legend inside each panel.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    # Panel titles carry no per-run silhouette: that number is noisy run-to-run
    # (JEPA and AE are tied); the averaged comparison lives in the table.
    _plot_one_panel(axes[0], z2_jepa, labels_rock, "JEPA", skip,
                    show_legend=False, title_fs=12, point_s=7)
    _plot_one_panel(axes[1], z2_ae, labels_rock, "AE baseline", skip,
                    show_legend=False, title_fs=12, point_s=7)
    classes = sorted(set(labels_rock) - skip)
    cmap = plt.get_cmap("tab20")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      color=cmap(i % 20), label=c)
               for i, c in enumerate(classes)]
    fig.legend(handles, classes, loc="center left", bbox_to_anchor=(0.83, 0.5),
               fontsize=8.5, frameon=True, framealpha=0.9, handletextpad=0.3,
               borderpad=0.4, labelspacing=0.35)
    fig.suptitle(suptitle, fontweight="bold", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.82, top=0.85, bottom=0.04, wspace=0.05)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_silhouette_comparison(table: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: per label set, one bar per encoder.

    Supports 2 encoders (AE vs JEPA — rock-resolution baseline) or up to
    4 encoders when the formation-resolution variants are also present.
    """
    if table.empty:
        return
    # Stable order: rock-resolution baselines first, formation second
    preferred = ["AE", "JEPA", "AE_formation", "JEPA_formation"]
    encoders = [e for e in preferred if e in set(table["encoder"].unique())]
    labelsets = list(table["label_set"].unique())
    n_enc = len(encoders)
    width = 0.8 / max(n_enc, 1)
    x = np.arange(len(labelsets))
    has_sd = "silhouette_sd" in table.columns
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, enc in enumerate(encoders):
        sub = table[table["encoder"] == enc].set_index("label_set")\
                .reindex(labelsets)
        heights = sub["silhouette"].values
        errs = (sub["silhouette_sd"].values if has_sd
                else np.zeros(len(labelsets)))
        offset = (i - (n_enc - 1) / 2) * width
        ax.bar(x + offset, heights, width,
                yerr=errs, capsize=2.5, ecolor="#333",
                error_kw={"linewidth": 0.9},
                color=ENCODER_COLOURS.get(enc, "#888888"),
                alpha=0.85, edgecolor="black", linewidth=0.3,
                label=enc.replace("_", " "))
        for j, h in enumerate(heights):
            if np.isnan(h):
                continue
            e = errs[j] if has_sd and not np.isnan(errs[j]) else 0.0
            top = (h + e) if h >= 0 else (h - e)
            ax.text(x[j] + offset,
                     top + (0.008 if h >= 0 else -0.018),
                     f"{h:+.2f}", ha="center",
                     fontsize=8, fontweight="bold",
                     va="bottom" if h >= 0 else "top")
    ax.set_xticks(x)
    ax.set_xticklabels(labelsets, rotation=20, ha="right")
    ax.set_ylabel("silhouette score")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Encoder silhouette by label set (higher = better separation)",
                  fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
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
    # One row of V panels, sized to the text column.
    fig, axes = plt.subplots(1, V, figsize=(2.4 * V, 3.6))
    axes = np.atleast_1d(axes).ravel()
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
        ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.2)
        ss_res = float(np.sum((t - r) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2)) or 1
        r2 = 1 - ss_res / ss_tot
        ax.set_title(f"{var}\n$R^2$ = {r2:.2f}", fontsize=15, fontweight="bold")
        ax.set_xlabel("truth", fontsize=14)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=12)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axes[0].set_ylabel("reconstruction", fontsize=14)
    fig.suptitle("Autoencoder reconstruction: truth vs recon per channel "
                 "(standardised)", fontweight="bold", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_noise_robustness(rows: list[dict], out_path: Path) -> None:
    """Mid-column rock silhouette vs input-noise level, one line per encoder.
    A representation that degrades more slowly under noisy logs is the more
    robust (and the more useful for real, noisy wells)."""
    df = pd.DataFrame(rows)
    colours = {"JEPA": "#4C72B0", "AE": "#dd8452"}
    has_sd = "silhouette_sd" in df.columns
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for enc in ("JEPA", "AE"):
        sub = df[df["encoder"] == enc].sort_values("sigma")
        if sub.empty:
            continue
        ax.plot(sub["sigma"], sub["silhouette"], marker="o", ms=6, lw=2.0,
                color=colours.get(enc, "#888"), label=enc)
        if has_sd:
            ax.fill_between(sub["sigma"],
                            sub["silhouette"] - sub["silhouette_sd"],
                            sub["silhouette"] + sub["silhouette_sd"],
                            color=colours.get(enc, "#888"), alpha=0.18)
    ax.set_xlabel(r"input-noise level  $\sigma$  (standardised units)")
    ax.set_ylabel("rock silhouette (middle 50%)")
    ax.set_title("Robustness to input noise: JEPA vs AE", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ─── report ──────────────────────────────────────────────────────────────

def write_encoder_report(summary: pd.DataFrame,
                          ae_recon_by_encoder: dict[str, list[dict]],
                          present_encoders: list[str],
                          n_boreholes: int,
                          out_path: Path) -> None:
    L = []
    a = L.append
    a("# Encoder evaluation report")
    a("")
    a(f"Validation set: **{n_boreholes:,} independent borehole columns** "
      "(each column has its own freshly-sampled stratigraphic stack).")
    a("")
    a("Encoders evaluated: " + ", ".join(present_encoders) + ".")
    a("")
    a("Naming: `*_formation` variants were trained on the formation-"
      "resolution dataset (`data/dataset_formation/`); the bare names are "
      "the rock-resolution baselines (`data/dataset/`).")
    a("")
    a("## 1. UMAP latent space, coloured by label scheme")
    a("")
    # auto-list the umap pngs that exist (one per present encoder)
    umap_map = {
        "AE":              "01_ae_umap.png",
        "JEPA":            "02_jepa_umap.png",
        "AE_formation":    "05_ae_formation_umap.png",
        "JEPA_formation":  "06_jepa_formation_umap.png",
    }
    for enc in present_encoders:
        png = umap_map.get(enc)
        if png is not None:
            a(f"![{enc} UMAP]({png})")
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
        # Stable column order: baselines first, then formation variants
        col_order = [c for c in ["AE", "JEPA", "AE_formation",
                                  "JEPA_formation"] if c in piv.columns]
        piv = piv.reindex(columns=col_order)
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
    if ae_recon_by_encoder:
        a("## 3. Autoencoder reconstruction quality")
        a("")
        recon_map = {
            "AE":           "04_ae_reconstruction.png",
            "AE_formation": "08_ae_formation_reconstruction.png",
        }
        for enc, rows in ae_recon_by_encoder.items():
            png = recon_map.get(enc)
            if png is not None:
                a(f"### {enc}")
                a("")
                a(f"![{enc} reconstruction]({png})")
                a("")
                a("| variable | R² | SmoothL1 | n cells |")
                a("|---|---|---|---|")
                for r in rows:
                    a(f"| {r['variable']} | {r['R2']:.3f} | "
                      f"{r['SmoothL1']:.4f} | {int(r['n']):,} |")
                a("")
        a("R² ≈ 1 and SmoothL1 ≪ 1 indicate the autoencoder can "
          "reconstruct each channel from the latent. Drops on a "
          "specific variable point to information loss in the bottleneck.")
        a("")
    a("## 4. Files")
    a("")
    a("- `silhouette_summary.csv` — silhouette per (encoder × label set)")
    if ae_recon_by_encoder:
        a("- `ae_reconstruction.csv` — per-(encoder, variable) R² and SmoothL1")
    a("- `0N_<encoder>_umap.png` — per-encoder UMAP grids")
    a("- `03_silhouette_comparison.png` — grouped bar chart")
    if ae_recon_by_encoder:
        a("- `0N_<encoder>_reconstruction.png` — per-variable truth-vs-recon "
          "scatter (AE family only)")
    a("")
    out_path.write_text("\n".join(L))
    print(f"  wrote {out_path}")


# ─── orchestration ───────────────────────────────────────────────────────

def _compute_ae_recon_rows(truth: np.ndarray, recon: np.ndarray,
                            variables: list[str]) -> list[dict]:
    rows = []
    for i, v in enumerate(variables):
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
        rows.append({
            "variable": v, "R2": round(r2, 4),
            "SmoothL1": round(smoothl1, 4), "n": int(len(t)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ae",   type=Path, default=DEFAULT_AE,
                     help=f"AE rock-resolution checkpoint (default {DEFAULT_AE})")
    ap.add_argument("--jepa", type=Path, default=DEFAULT_JEPA,
                     help=f"JEPA rock-resolution checkpoint (default {DEFAULT_JEPA})")
    ap.add_argument("--ae-formation", type=Path, default=DEFAULT_AE_FORMATION,
                     help=f"AE formation-resolution checkpoint")
    ap.add_argument("--jepa-formation", type=Path, default=DEFAULT_JEPA_FORMATION,
                     help=f"JEPA formation-resolution checkpoint")
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

    # Encoder specs: (display_name, family, panel_idx, checkpoint_path)
    # Index controls output file naming so legacy outputs (01_ae_umap.png,
    # 02_jepa_umap.png) keep their stable names when only the baselines run.
    specs = [
        ("AE",              "ae",    1, args.ae),
        ("JEPA",            "jepa",  2, args.jepa),
        ("AE_formation",    "ae",    5, args.ae_formation),
        ("JEPA_formation",  "jepa",  6, args.jepa_formation),
    ]
    present = [s for s in specs if s[3].exists()]
    missing = [s for s in specs if not s[3].exists()]
    if not present:
        raise FileNotFoundError(
            "no encoder checkpoints found — looked for: "
            + ", ".join(str(s[3]) for s in specs))
    for s in missing:
        print(f"  note: {s[3]} not found — skipping {s[0]}")
    print(f"  evaluating: {[s[0] for s in present]}")

    print(f"loading simulator artifacts ...")
    bank = DistributionBank.load(args.distributions)
    geom = FormationGeometry.load(args.geometry)
    sim_cfg = SimConfig()

    print(f"\ngenerating {args.n_boreholes} independent boreholes (shared "
          "across all encoders) ...")
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

    summary_rows: list[dict] = []
    ae_recon_by_encoder: dict[str, list[dict]] = {}
    z2_by_encoder: dict[str, np.ndarray] = {}
    sil_by_encoder: dict[str, dict] = {}
    model_by_encoder: dict[str, tuple] = {}

    for display_name, family, panel_idx, ckpt_path in present:
        print(f"\n── {display_name} ──  loading {ckpt_path}")
        if family == "ae":
            model, stats, vars_ = load_ae_checkpoint(ckpt_path, device=device)
            Z = encode_with_ae(model, values, stats, vars_, device)
        else:
            model, stats, vars_ = load_jepa_checkpoint(ckpt_path, device=device)
            Z = encode_with_jepa(model, values, stats, vars_, device)
        model_by_encoder[display_name] = (model, stats, vars_, family)
        print(f"  latents: {Z.shape}  variables: {vars_}")
        Z2 = reduce_to_2d(Z, method="umap")
        sil = silhouette_table(Z, label_sets)
        z2_by_encoder[display_name] = Z2
        sil_by_encoder[display_name] = sil
        knn = {name: knn_label_accuracy(Z, labels)
               for name, labels in label_sets.items()}
        probe = {name: probe_accuracy(Z, labels)
                 for name, labels in label_sets.items()}
        fewshot = {name: few_shot_probe(Z, labels)
                   for name, labels in label_sets.items()}
        for name, (score, n) in sil.items():
            summary_rows.append({"encoder": display_name, "label_set": name,
                                  "silhouette": score,
                                  "knn_acc": knn.get(name, float("nan")),
                                  "probe_acc": probe.get(name, float("nan")),
                                  "fewshot_acc": fewshot.get(name, float("nan")),
                                  "n_used": n})
        for nm in ("rock (middle 50%)", "formation (middle 50%)"):
            print(f"    {nm:24s}  knn={knn.get(nm, float('nan')):.3f}  "
                  f"probe={probe.get(nm, float('nan')):.3f}  "
                  f"fewshot(3)={fewshot.get(nm, float('nan')):.3f}")

        umap_path = args.out / f"{panel_idx:02d}_{display_name.lower()}_umap.png"
        plot_umap_grid(Z2, label_sets, sil,
                        out_path=umap_path,
                        title=f"{display_name} UMAP — "
                              f"{args.n_boreholes} boreholes")

        if family == "ae":
            print(f"  computing reconstruction quality ...")
            truth, recon = reconstruct_with_ae(model, values, stats, vars_,
                                                device)
            recon_panel_idx = panel_idx + 3 if panel_idx > 2 else 4
            plot_ae_reconstruction(
                truth, recon, vars_,
                args.out
                / f"{recon_panel_idx:02d}_{display_name.lower()}_reconstruction.png",
            )
            ae_recon_by_encoder[display_name] = _compute_ae_recon_rows(
                truth, recon, vars_,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "silhouette_summary.csv", index=False)
    print(f"\n  wrote silhouette_summary.csv  ({len(summary)} rows)")

    if ae_recon_by_encoder:
        # Long-form CSV: one row per (encoder, variable) so the rock and
        # formation AEs sit side-by-side in tooling.
        recon_rows = []
        for enc, rows in ae_recon_by_encoder.items():
            for r in rows:
                recon_rows.append({"encoder": enc, **r})
        pd.DataFrame(recon_rows).to_csv(
            args.out / "ae_reconstruction.csv", index=False)
        print(f"  wrote ae_reconstruction.csv  ({len(recon_rows)} rows)")

    plot_silhouette_comparison(summary, args.out
                                / "03_silhouette_comparison.png")

    # Side-by-side JEPA vs AE latent maps (same boreholes, rock-coloured).
    if "JEPA" in z2_by_encoder and "AE" in z2_by_encoder:
        rock_key = "rock (middle 50%)"
        sj = sil_by_encoder["JEPA"].get(rock_key, (float("nan"), 0))[0]
        sa = sil_by_encoder["AE"].get(rock_key, (float("nan"), 0))[0]
        plot_jepa_vs_ae_umap(z2_by_encoder["JEPA"], z2_by_encoder["AE"],
                             label_sets[rock_key], sj, sa,
                             args.out / "jepa_vs_ae_umap.png")

    # Robustness to input noise: degrade the standardised logs and watch the
    # mid-column rock silhouette of each encoder fall.
    if "JEPA" in model_by_encoder and "AE" in model_by_encoder:
        rock_key = "rock (middle 50%)"
        rock_labels = label_sets[rock_key]
        sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
        noise_rows = []
        print("\nnoise-robustness sweep (rock-mid silhouette) ...")
        for enc in ("JEPA", "AE"):
            model, stats, vars_, family = model_by_encoder[enc]
            enc_fn = encode_with_jepa if family == "jepa" else encode_with_ae
            for sd in sigmas:
                Zn = enc_fn(model, values, stats, vars_, device,
                            noise_sd=sd, seed=0)
                s = silhouette_table(Zn, {rock_key: rock_labels})[rock_key][0]
                noise_rows.append({"encoder": enc, "sigma": sd,
                                   "silhouette": s})
                print(f"  {enc:5s}  sigma={sd:.2f}  silhouette={s:+.3f}")
        pd.DataFrame(noise_rows).to_csv(args.out / "noise_robustness.csv",
                                        index=False)
        plot_noise_robustness(noise_rows, args.out / "noise_robustness.png")

    print("\nwriting ENCODER_REPORT.md ...")
    write_encoder_report(
        summary=summary,
        ae_recon_by_encoder=ae_recon_by_encoder,
        present_encoders=[s[0] for s in present],
        n_boreholes=args.n_boreholes,
        out_path=args.out / "ENCODER_REPORT.md",
    )

    print(f"\n→ done. outputs in {args.out}/")


if __name__ == "__main__":
    main()
