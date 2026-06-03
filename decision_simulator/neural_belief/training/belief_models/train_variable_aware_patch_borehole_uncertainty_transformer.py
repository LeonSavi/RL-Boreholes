"""Training pipeline for VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer.

Uncertainty-head experiment: identical training setup to the variable-aware patch
experiment (train_variable_aware_patch_borehole_transformer.py) but adds a second
decoder head that predicts per-cell spatial uncertainty alongside the ore map.

Training objective
------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + cfg.uncertainty_weight * uncertainty_loss

The false-positive penalty (if enabled) is applied to pred_ore only.

This file is intentionally parallel to train_variable_aware_patch_borehole_transformer.py.
Dataset, collate, normalizer, and checkpoint utilities are fully reused.
Validation helpers are re-implemented here to handle the (ore, uncertainty) tuple
output of the uncertainty model; drill-bin and no-ore helpers reuse the originals
via a thin adapter wrapper.

Checkpoints
-----------
variable_aware_patch_uncertainty_best.pt  — lowest validation MSE (ore-value space)
variable_aware_patch_uncertainty_last.pt  — final epoch
"""

from __future__ import annotations

import datetime
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decision_simulator.resources import DecisionSimulationResources
from ...utils import TargetNormalizer
from ...training_utils import (
    DRILL_BINS,
    export_history,
    false_positive_loss,
    load_model_encoder_checkpoint,
    pearson_correlation,
    save_checkpoint_model,
    save_no_ore_metrics,
)
from ...models.belief_models.end_to_end.variable_aware_patch_borehole_transformer import (
    VariableAwarePatchBoreholeEndToEndConfig,
)
from ...models.belief_models.end_to_end.variable_aware_patch_borehole_uncertainty_transformer import (
    VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
)
from .train_end_to_end_map_belief import E2EMapDataset
from .helpers import (
    validate_e2e_map_by_drill_bins,
    validate_e2e_map_by_step,
    validate_no_ore_e2e_map,
    collate_e2e_map,
)


# ---------------------------------------------------------------------------
# Gradient helper
# ---------------------------------------------------------------------------


def _ore_gradient_magnitude_np(ore_map: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a 2-D ore map via numpy finite differences."""
    dy, dx = np.gradient(ore_map)
    return np.sqrt(dy ** 2 + dx ** 2)


# ---------------------------------------------------------------------------
# Adapter: expose ore-only forward() for reusing existing validation helpers
# ---------------------------------------------------------------------------


class _OreWrapper(nn.Module):
    """Wraps the uncertainty model so that forward() returns only the ore prediction.

    Allows reusing validation helpers from helpers/ that expect a
    model whose forward() returns a single (B, 1, n_x, n_y) tensor.
    """

    def __init__(
        self,
        model: VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    ) -> None:
        super().__init__()
        self._model = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        pred_ore, _ = self._model(*args, **kwargs)
        return pred_ore


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class VariableAwarePatchBoreholeUncertaintyE2ETrainingConfig:
    """Training hyperparameters for the uncertainty-head variant."""

    # Dataset
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # Sequential dataset mode
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])

    # Borehole dimensions — resolved from resources at training time
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder: depth patch size
    bh_patch_size: int = 20

    # Borehole encoder: transformer over (variable, patch) tokens
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Borehole embedding output dimension
    latent_dim: int = 128

    # Spatial grid — set automatically from cache in the training function
    n_x: int = 32
    n_y: int = 32

    # Map belief transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.20
    head_hidden_dim: int = 128
    pe_max_freq: float = 10000.0

    # Target normalisation
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # Optimisation
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50
    grad_clip_norm: float = 1.0  # 0.0 = disabled

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.0

    # False-positive penalty (applied to pred_ore only)
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    false_positive_threshold: float = 0.05

    # Uncertainty head
    use_uncertainty_head: bool = True
    uncertainty_weight: float = 0.1

    # Misc
    seed: int = 42
    borehole_encoder: str = "variable_aware_patch_uncertainty"
    n_val_plots: int = 20

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    def to_model_config(self) -> VariableAwarePatchBoreholeEndToEndConfig:
        """Build a VariableAwarePatchBoreholeEndToEndConfig for model construction."""
        return VariableAwarePatchBoreholeEndToEndConfig(
            n_variables=self.n_variables,
            n_depth=self.n_depth,
            bh_patch_size=self.bh_patch_size,
            bh_d_model=self.bh_d_model,
            bh_n_heads=self.bh_n_heads,
            bh_n_layers=self.bh_n_layers,
            latent_dim=self.latent_dim,
            n_x=self.n_x,
            n_y=self.n_y,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_hidden_dim=self.head_hidden_dim,
            pe_max_freq=self.pe_max_freq,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_uncertainty_e2e_map(
    model: VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Ore reconstruction metrics + uncertainty calibration metrics.

    Ore metrics (ore-value / denormalised space)
    --------------------------------------------
    val_mse, val_mae, val_corr

    Uncertainty metrics (normalised space — model targets |pred_norm - tgt_norm|)
    ------------------------------------------------------------------------------
    val_unc_mse       — MSE between pred_uncertainty and actual absolute error
    val_unc_mae       — MAE between pred_uncertainty and actual absolute error
    val_unc_corr      — Pearson correlation between pred_uncertainty and abs error
    val_unc_top10_ratio — mean(error in top-10% uncertain cells) / mean(error overall)
                          > 1.0 means the model correctly flags high-error regions
    """
    model.eval()
    mse_total = mae_total = corr_total = 0.0
    unc_mse_total = unc_mae_total = unc_corr_total = unc_top10_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_ore_norm, pred_unc_norm = model(bh, ov, pos, pm)

            # Ore metrics in denormalised (ore-value) space
            pred = normalizer.inverse_tensor(pred_ore_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)
            mse_total += F.mse_loss(pred, tgt_raw).item()
            mae_total += (pred - tgt_raw).abs().mean().item()
            corr_total += pearson_correlation(pred, tgt_raw)

            # Uncertainty calibration in normalised space
            abs_err_norm = (pred_ore_norm - tgt).abs()
            unc_mse_total += F.mse_loss(pred_unc_norm, abs_err_norm).item()
            unc_mae_total += (pred_unc_norm - abs_err_norm).abs().mean().item()
            unc_corr_total += pearson_correlation(pred_unc_norm, abs_err_norm)

            # Top-10% uncertainty coverage (per-sample, averaged over batch)
            B = pred_unc_norm.shape[0]
            pu = pred_unc_norm.view(B, -1)
            ae = abs_err_norm.view(B, -1)
            k = max(1, int(pu.shape[1] * 0.10))
            top_idx = pu.topk(k, dim=1).indices
            top10_err = ae.gather(1, top_idx).mean(dim=1)
            mean_err = ae.mean(dim=1).clamp(min=1e-8)
            unc_top10_total += (top10_err / mean_err).mean().item()

            n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_mse": mse_total / n,
        "val_mae": mae_total / n,
        "val_corr": corr_total / n,
        "val_unc_mse": unc_mse_total / n,
        "val_unc_mae": unc_mae_total / n,
        "val_unc_corr": unc_corr_total / n,
        "val_unc_top10_ratio": unc_top10_total / n,
    }


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------


def _compute_uncertainty_correlation_diagnostics(
    model: VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    val_loader: DataLoader,
    normalizer: TargetNormalizer,
    device: str,
) -> dict[str, float]:
    """Pixel-level Pearson correlations between predicted uncertainty and related signals.

    All quantities are in normalised space (matching the training objective):
      - absolute error  : |pred_norm - target_norm|
      - scaled error    : above / (|pred_norm| + |target_norm| + 1e-8)
      - predicted ore   : pred_norm
      - ore gradient    : ||grad(pred_norm)||

    Returns
    -------
    {corr_unc_abs_err, corr_unc_scaled_err, corr_unc_pred_ore, corr_unc_ore_grad}
    """
    model.eval()

    unc_buf: list[np.ndarray] = []
    ae_buf:  list[np.ndarray] = []
    se_buf:  list[np.ndarray] = []
    po_buf:  list[np.ndarray] = []
    og_buf:  list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            bh  = batch["boreholes"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm  = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_ore_norm, pred_unc_norm = model(bh, ov, pos, pm)

            ae_norm = (pred_ore_norm - tgt).abs()
            se_norm = ae_norm / (pred_ore_norm.abs() + tgt.abs() + 1e-8)

            B = pred_ore_norm.shape[0]
            for b in range(B):
                p  = pred_ore_norm[b, 0].cpu().numpy()
                u  = pred_unc_norm[b, 0].cpu().numpy()
                ae = ae_norm[b, 0].cpu().numpy()
                se = se_norm[b, 0].cpu().numpy()
                og = _ore_gradient_magnitude_np(p)

                unc_buf.append(u.ravel())
                ae_buf.append(ae.ravel())
                se_buf.append(se.ravel())
                po_buf.append(p.ravel())
                og_buf.append(og.ravel())

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    unc = np.concatenate(unc_buf)
    return {
        "corr_unc_abs_err":    _pearson(unc, np.concatenate(ae_buf)),
        "corr_unc_scaled_err": _pearson(unc, np.concatenate(se_buf)),
        "corr_unc_pred_ore":   _pearson(unc, np.concatenate(po_buf)),
        "corr_unc_ore_grad":   _pearson(unc, np.concatenate(og_buf)),
    }


# ---------------------------------------------------------------------------
# Drill-count uncertainty analysis
# ---------------------------------------------------------------------------


def _analyze_uncertainty_by_drill_count(
    model: VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    val_ds: E2EMapDataset,
    normalizer: TargetNormalizer,
    device: str,
) -> dict:
    """Per-drill-count uncertainty statistics, split by ore / no-ore maps.

    Iterates each validation sample individually, converts predicted uncertainty
    to approximate ore-value space, and groups by drill count.

    Returns
    -------
    {
      "by_drill_count":        {drill_count: {n, unc_sum_mean, unc_mean_mean, unc_max_mean}},
      "ore_by_drill_count":    same, ore-body maps only,
      "no_ore_by_drill_count": same, no-ore maps only,
    }
    """
    model.eval()
    rows: list[dict] = []

    for sample in val_ds.samples:
        bh  = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
        ov  = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
        pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_ore_norm, pred_unc_norm = model(bh, ov, pos)

        pred_ore_np = pred_ore_norm.squeeze().cpu().numpy()
        unc_raw = np.abs(
            normalizer.inverse(
                (pred_ore_norm + pred_unc_norm).squeeze().cpu().numpy()
            )
            - normalizer.inverse(pred_ore_np)
        )

        target_raw = normalizer.inverse(sample["target_map"])
        is_no_ore  = float(target_raw.max()) < 1e-6

        rows.append({
            "drill_count": int(sample["drill_count"]),
            "is_no_ore":   is_no_ore,
            "unc_sum":     float(unc_raw.sum()),
            "unc_mean":    float(unc_raw.mean()),
            "unc_max":     float(unc_raw.max()),
        })

    def _aggregate(group_rows: list[dict]) -> dict:
        if not group_rows:
            return {}
        by_dc: dict[int, list[dict]] = defaultdict(list)
        for r in group_rows:
            by_dc[r["drill_count"]].append(r)
        return {
            str(dc): {
                "n":             len(dc_rows),
                "unc_sum_mean":  float(np.mean([r["unc_sum"]  for r in dc_rows])),
                "unc_mean_mean": float(np.mean([r["unc_mean"] for r in dc_rows])),
                "unc_max_mean":  float(np.mean([r["unc_max"]  for r in dc_rows])),
            }
            for dc, dc_rows in sorted(by_dc.items())
        }

    return {
        "by_drill_count":        _aggregate(rows),
        "ore_by_drill_count":    _aggregate([r for r in rows if not r["is_no_ore"]]),
        "no_ore_by_drill_count": _aggregate([r for r in rows if     r["is_no_ore"]]),
    }


# ---------------------------------------------------------------------------
# Training sample inspection
# ---------------------------------------------------------------------------


def _save_training_samples(
    train_ds: E2EMapDataset,
    normalizer: TargetNormalizer,
    run_dir: Path,
    n_samples: int = 5,
) -> None:
    """Save a handful of training samples for input inspection.

    Each 3-panel figure shows: observed ore values, target ore map
    (ore-value space), and normalised target (what the model sees).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(run_dir) / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = np.linspace(0, len(train_ds) - 1, n_samples, dtype=int)
    for k, idx in enumerate(indices):
        sample = train_ds.samples[int(idx)]

        n_x, n_y = sample["target_map"].shape
        sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
        obs_mask   = np.zeros((n_x, n_y), dtype=np.float32)
        for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
            i = int(round(float(px) * (n_x - 1)))
            j = int(round(float(py) * (n_y - 1)))
            sparse_ore[i, j] = float(ov_val)
            obs_mask[i, j]   = 1.0

        target_norm = sample["target_map"]
        target_raw  = normalizer.inverse(target_norm)
        drill_rows, drill_cols = np.where(obs_mask > 0)
        vmax_ore = float(max(float(target_raw.max()), 1e-3))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Train sample {k}  ({int(obs_mask.sum())} drills)", fontsize=11)
        kw = dict(origin="lower", cmap="viridis")

        ax = axes[0]
        ax.set_title(f"Observations ({int(obs_mask.sum())} drills)")
        im = ax.imshow(sparse_ore.T, vmin=0, vmax=vmax_ore, **kw)
        ax.scatter(drill_rows, drill_cols, c="red", s=10, marker="x", linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1]
        ax.set_title("Target ore map (ore-value space)")
        im = ax.imshow(target_raw.T, vmin=0, vmax=vmax_ore, **kw)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[2]
        ax.set_title("Normalised target (model input)")
        im = ax.imshow(target_norm.T, origin="lower", cmap="plasma")
        fig.colorbar(im, ax=ax, fraction=0.046)

        fig.tight_layout()
        fig.savefig(out_dir / f"train_sample_{k:02d}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  training samples -> {out_dir}")


# ---------------------------------------------------------------------------
# Validation plots
# ---------------------------------------------------------------------------


def _save_uncertainty_val_plots(
    model: VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 20,
) -> None:
    """Save 7-panel validation figures.

    Panels: observations / true ore / predicted ore / absolute error /
            scaled error / predicted uncertainty / ore gradient magnitude.

    Auto-detects sequential mode by checking for ``sequence_id`` in dataset samples.

    Non-sequential: one 7-panel PNG per selected sample.
    Sequential: per-step 7-panel PNGs + ``evolution.png`` grid (rows=steps, cols=7)
    with global colour scales shared across all steps.

    Absolute error and predicted uncertainty share the same colour scale so they
    are directly visually comparable.  Uncertainty is converted from normalised
    space to approximate ore-value space via
    ``abs(inverse(pred_norm + unc_norm) - inverse(pred_norm))``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_sequential = bool(val_ds.samples and "sequence_id" in val_ds.samples[0])

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model.eval()

    def _run_sample(sample):
        """Return (sparse_ore, obs_mask, true_ore, pred_ore,
                   abs_err, scaled_err, ore_grad, pred_unc_raw)."""
        bh  = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
        ov  = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
        pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_ore_norm, pred_unc_norm = model(bh, ov, pos)
        pred_ore = normalizer.inverse(pred_ore_norm.squeeze().cpu().numpy())
        # Approximate ore-value-space uncertainty via the nonlinear (log1p)
        # inverse: abs(inverse(pred + unc) - inverse(pred)).
        pred_unc_raw = np.abs(
            normalizer.inverse((pred_ore_norm + pred_unc_norm).squeeze().cpu().numpy())
            - pred_ore
        )
        n_x, n_y = sample["target_map"].shape
        sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
        obs_mask   = np.zeros((n_x, n_y), dtype=np.float32)
        for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
            i = int(round(float(px) * (n_x - 1)))
            j = int(round(float(py) * (n_y - 1)))
            sparse_ore[i, j] = float(ov_val)
            obs_mask[i, j]   = 1.0
        true_ore   = normalizer.inverse(sample["target_map"])
        abs_err    = np.abs(pred_ore - true_ore)
        scaled_err = abs_err / (np.abs(pred_ore) + np.abs(true_ore) + 1e-8)
        ore_grad   = _ore_gradient_magnitude_np(pred_ore)
        return sparse_ore, obs_mask, true_ore, pred_ore, abs_err, scaled_err, ore_grad, pred_unc_raw

    def _draw_7panel(fig, axes, sparse_ore, obs_mask, true_ore, pred_ore,
                     abs_err, scaled_err, ore_grad, pred_unc_raw,
                     vmax_ore, vmax_err, vmax_scaled, vmax_grad,
                     title: str = "") -> None:
        if title:
            fig.suptitle(title, fontsize=11)
        drill_rows, drill_cols = np.where(obs_mask > 0)
        kw_ore = dict(origin="lower", cmap="viridis", vmin=0, vmax=vmax_ore)

        ax = axes[0]
        ax.set_title(f"Observations ({int(obs_mask.sum())} drills)")
        im = ax.imshow(sparse_ore.T, **kw_ore)
        ax.scatter(drill_rows, drill_cols, c="red", s=10, marker="x", linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1]
        ax.set_title("True ore map")
        im = ax.imshow(true_ore.T, **kw_ore)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[2]
        ax.set_title("Predicted ore map")
        im = ax.imshow(pred_ore.T, **kw_ore)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[3]
        ax.set_title("Absolute error")
        im = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=vmax_err, cmap="Reds")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[4]
        ax.set_title("Scaled error\n|err|/(|pred|+|true|+ε)")
        im = ax.imshow(scaled_err.T, origin="lower", vmin=0, vmax=vmax_scaled, cmap="YlOrRd")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[5]
        ax.set_title("Predicted uncertainty\n(ore-value space, approx.)")
        im = ax.imshow(pred_unc_raw.T, origin="lower", vmin=0, vmax=vmax_err, cmap="Oranges")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[6]
        ax.set_title("Ore gradient magnitude")
        im = ax.imshow(ore_grad.T, origin="lower", vmin=0, vmax=vmax_grad, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # -------------------------------------------------------------------------
    # Non-sequential: one 7-panel figure per selected sample
    # -------------------------------------------------------------------------
    if not is_sequential:
        indices = np.linspace(0, len(val_ds) - 1, n_plots, dtype=int)
        for plot_k, idx in enumerate(indices):
            sample = val_ds.samples[int(idx)]
            sparse_ore, obs_mask, true_ore, pred_ore, abs_err, scaled_err, ore_grad, pred_unc_raw = _run_sample(sample)

            vmax_ore    = float(max(true_ore.max(), pred_ore.max(), 1e-3))
            vmax_err    = float(max(abs_err.max(), pred_unc_raw.max(), 1e-3))
            vmax_scaled = float(max(scaled_err.max(), 1e-3))
            vmax_grad   = float(max(ore_grad.max(), 1e-3))

            fig, axes = plt.subplots(1, 7, figsize=(30, 4))
            _draw_7panel(
                fig, axes, sparse_ore, obs_mask, true_ore, pred_ore,
                abs_err, scaled_err, ore_grad, pred_unc_raw,
                vmax_ore, vmax_err, vmax_scaled, vmax_grad,
                title=f"Val sample {plot_k}  ({sample['drill_count']} drills)",
            )
            fig.tight_layout()
            fig.text(0.5, 0.01, fig_ts, ha="center", va="bottom", fontsize=8, color="gray")
            fig.savefig(out_dir / f"val_sample_{plot_k:02d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Sequential: per-step PNGs + evolution.png grid (rows=steps, cols=7)
    # -------------------------------------------------------------------------
    else:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for idx, s in enumerate(val_ds.samples):
            key = (int(s["map_idx"]), int(s["sequence_id"]))
            groups[key].append((int(s["drill_count"]), idx))
        for key in groups:
            groups[key].sort(key=lambda x: x[0])

        all_keys = list(groups.keys())
        n_select = min(n_plots, len(all_keys))
        sel_indices = np.linspace(0, len(all_keys) - 1, n_select, dtype=int)
        selected_keys = [all_keys[i] for i in sel_indices]

        for map_idx, seq_id in selected_keys:
            seq_dir = out_dir / f"seq_{map_idx:04d}_{seq_id:02d}"
            seq_dir.mkdir(parents=True, exist_ok=True)

            step_pairs = groups[(map_idx, seq_id)]
            n_steps = len(step_pairs)

            # Run all steps first so global colour scales are available.
            # Tuple order: (step, sparse_ore, obs_mask, true_ore, pred_ore,
            #               abs_err, scaled_err, ore_grad, pred_unc_raw)
            panel_data = []
            for step, sample_idx in step_pairs:
                arrays = _run_sample(val_ds.samples[sample_idx])
                panel_data.append((step, *arrays))

            global_vmax_ore = max(
                max(float(t.max()), float(p.max()))
                for _, _, _, t, p, _, _, _, _ in panel_data
            )
            global_vmax_ore = max(global_vmax_ore, 1e-3)

            global_vmax_err = max(
                max(float(e.max()), float(u.max()))
                for _, _, _, _, _, e, _, _, u in panel_data
            )
            global_vmax_err = max(global_vmax_err, 1e-3)

            global_vmax_scaled = max(
                float(se.max()) for _, _, _, _, _, _, se, _, _ in panel_data
            )
            global_vmax_scaled = max(global_vmax_scaled, 1e-3)

            global_vmax_grad = max(
                float(g.max()) for _, _, _, _, _, _, _, g, _ in panel_data
            )
            global_vmax_grad = max(global_vmax_grad, 1e-3)

            # Per-step individual 7-panel PNGs
            for step, sparse_ore, obs_mask, true_ore, pred_ore, abs_err, scaled_err, ore_grad, pred_unc_raw in panel_data:
                fig, axes = plt.subplots(1, 7, figsize=(30, 4))
                _draw_7panel(
                    fig, axes, sparse_ore, obs_mask, true_ore, pred_ore,
                    abs_err, scaled_err, ore_grad, pred_unc_raw,
                    global_vmax_ore, global_vmax_err, global_vmax_scaled, global_vmax_grad,
                    title=f"Map {map_idx} / Seq {seq_id} / Step {step} ({step} drills)",
                )
                fig.tight_layout()
                fig.text(0.5, 0.01, fig_ts, ha="center", va="bottom", fontsize=8, color="gray")
                fig.savefig(seq_dir / f"step_{step:03d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

            # Combined evolution.png grid (n_steps × 7)
            fig, axes = plt.subplots(n_steps, 7, figsize=(36, 4 * n_steps))
            if n_steps == 1:
                axes = axes[np.newaxis, :]

            fig.suptitle(
                f"Belief evolution — Map {map_idx} / Seq {seq_id}"
                f"  (ore max={global_vmax_ore:.3f}  err max={global_vmax_err:.3f})",
                fontsize=12,
            )

            col_titles = [
                "Observations", "True ore map", "Predicted ore map",
                "Abs error", "Scaled error", "Uncertainty\n(ore-value)", "Ore gradient",
            ]
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, fontsize=10)

            im_pred_last = im_err_last = im_se_last = im_unc_last = im_grad_last = None
            for row_idx, (step, sparse_ore, obs_mask, true_ore, pred_ore, abs_err, scaled_err, ore_grad, pred_unc_raw) in enumerate(panel_data):
                drill_rows, drill_cols = np.where(obs_mask > 0)
                kw_ore = dict(origin="lower", cmap="viridis", vmin=0, vmax=global_vmax_ore)

                ax = axes[row_idx, 0]
                ax.imshow(sparse_ore.T, **kw_ore)
                ax.scatter(drill_rows, drill_cols, c="red", s=8, marker="x", linewidths=0.6)
                ax.set_ylabel(f"step {step}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 1]
                ax.imshow(true_ore.T, **kw_ore)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 2]
                im_pred_last = ax.imshow(pred_ore.T, **kw_ore)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 3]
                im_err_last = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=global_vmax_err, cmap="Reds")
                ax.set_xlabel(f"max err={abs_err.max():.3f}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 4]
                im_se_last = ax.imshow(scaled_err.T, origin="lower", vmin=0, vmax=global_vmax_scaled, cmap="YlOrRd")
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 5]
                im_unc_last = ax.imshow(pred_unc_raw.T, origin="lower", vmin=0, vmax=global_vmax_err, cmap="Oranges")
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axes[row_idx, 6]
                im_grad_last = ax.imshow(ore_grad.T, origin="lower", vmin=0, vmax=global_vmax_grad, cmap="Blues")
                ax.set_xticks([])
                ax.set_yticks([])

            fig.colorbar(im_pred_last, ax=axes[:, 2], shrink=0.6, label="ore value")
            fig.colorbar(im_err_last,  ax=axes[:, 3], shrink=0.6, label="abs error")
            fig.colorbar(im_se_last,   ax=axes[:, 4], shrink=0.6, label="scaled error")
            fig.colorbar(im_unc_last,  ax=axes[:, 5], shrink=0.6, label="uncertainty")
            fig.colorbar(im_grad_last, ax=axes[:, 6], shrink=0.6, label="ore gradient")
            fig.tight_layout()
            fig.savefig(seq_dir / "evolution.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    print(f"  uncertainty plots saved -> {out_dir}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_variable_aware_patch_uncertainty_borehole_transformer(
    resources: DecisionSimulationResources,
    cfg: VariableAwarePatchBoreholeUncertaintyE2ETrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: E2EMapDataset | None = None,
    val_ds: E2EMapDataset | None = None,
) -> tuple[
    VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    TargetNormalizer,
    Path,
]:
    """Train VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer.

    Creates a timestamped run directory inside checkpoint_dir and saves all
    outputs there (checkpoints, metric JSON files, training history, plots):
      <checkpoint_dir>/<timestamp>/variable_aware_patch_uncertainty_best.pt
      <checkpoint_dir>/<timestamp>/variable_aware_patch_uncertainty_last.pt
      <checkpoint_dir>/<timestamp>/training_history.csv
      <checkpoint_dir>/<timestamp>/val_metrics_*.json
      <checkpoint_dir>/<timestamp>/val_sample_*.png  (or seq_*/…)

    Parameters
    ----------
    resources        : shared resources (norm_stats for borehole standardisation)
    cfg              : training hyperparameters
    device           : torch device string
    checkpoint_dir   : parent directory; a timestamped sub-directory is created here
    plot_dir         : if given, validation plots are saved (to the run directory)
    verbose          : print per-epoch metrics
    train_ds/val_ds  : pre-built E2EMapDataset (required)

    Returns
    -------
    (trained model with best weights, fitted TargetNormalizer, run_dir)
    """
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path(checkpoint_dir) / run_ts
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- resolve borehole dimensions from resources / dataset ----------------
    if resources.variable_names:
        cfg.n_variables = len(resources.variable_names)

    if train_ds is None or val_ds is None:
        raise ValueError("train_ds and val_ds must be provided.")

    if len(train_ds) > 0:
        cfg.n_depth = train_ds.samples[0]["boreholes"].shape[2]
        sample_map = train_ds.samples[0]["target_map"]
        cfg.n_x, cfg.n_y = int(sample_map.shape[0]), int(sample_map.shape[1])

    # ---- target normalisation ------------------------------------------------
    normalizer = TargetNormalizer(mode=cfg.norm_mode)
    normalizer.fit(train_ds.raw_targets())
    train_ds.apply_target_normalizer(normalizer)
    val_ds.apply_target_normalizer(normalizer)

    if verbose:
        if cfg.norm_mode == "zscore":
            print(
                f"  target norm   : zscore  mean={normalizer.mean:.4f}"
                f"  std={normalizer.std:.4f}"
            )
        elif cfg.norm_mode != "none":
            print(f"  target norm   : {cfg.norm_mode}")

    # ---- data loaders --------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_e2e_map,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_e2e_map,
    )

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ---------------------------------------------------------------
    model_cfg = cfg.to_model_config()
    model = VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer(
        model_cfg
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        n_bh_tokens = cfg.n_variables * n_patches
        print(f"  model params  : {n_params:,}")
        print(f"  variable_aware: True")
        print(f"  uncertainty   : True (weight={cfg.uncertainty_weight})")
        print(f"  n_variables   : {cfg.n_variables}")
        print(f"  patch_size    : {cfg.bh_patch_size}")
        print(f"  n_patches     : {n_patches}")
        print(f"  bh_tokens/bh  : {n_bh_tokens} + 1 CLS = {n_bh_tokens + 1}")
        print(f"  bh_d_model    : {cfg.bh_d_model}")
        print(f"  bh_n_layers   : {cfg.bh_n_layers}")
        print(f"  d_model       : {cfg.d_model}")
        print(f"  n_enc_layers  : {cfg.n_encoder_layers}")
        print(f"  n_heads       : {cfg.n_heads}")
        print(f"  latent_dim    : {cfg.latent_dim}")
        print(f"  dropout       : {cfg.dropout}")
        print(f"  grid          : {cfg.n_x} × {cfg.n_y}")

    # ---- training loop -------------------------------------------------------
    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        total_loss_sum = 0.0
        ore_loss_sum = 0.0
        unc_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)  # (B, 1, n_x, n_y) normalised

            pred_ore, pred_uncertainty = model(bh, ov, pos, pm)

            ore_loss = F.mse_loss(pred_ore, tgt)
            loss = ore_loss

            if cfg.use_uncertainty_head:
                uncertainty_target = torch.abs(pred_ore.detach() - tgt)
                uncertainty_loss = F.mse_loss(pred_uncertainty, uncertainty_target)
                loss = ore_loss + cfg.uncertainty_weight * uncertainty_loss
                unc_loss_sum += uncertainty_loss.item()

            if cfg.use_false_positive_penalty:
                loss = loss + cfg.false_positive_weight * false_positive_loss(
                    pred_ore, tgt, normalizer
                )

            optimiser.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimiser.step()
            total_loss_sum += loss.item()
            ore_loss_sum += ore_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        val_metrics = _validate_uncertainty_e2e_map(model, val_loader, device, normalizer)

        row = {
            "epoch": epoch,
            "train_loss": total_loss_sum / n,
            "train_ore_loss": ore_loss_sum / n,
            "train_unc_loss": unc_loss_sum / n if cfg.use_uncertainty_head else float("nan"),
            **val_metrics,
        }
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{cfg.n_epochs}"
                f"  train={total_loss_sum / n:.4f}"
                f"  ore={ore_loss_sum / n:.4f}"
                + (f"  unc={unc_loss_sum / n:.4f}" if cfg.use_uncertainty_head else "")
                + f"  val_mse={val_metrics['val_mse']:.4f}"
                f"  val_corr={val_metrics['val_corr']:.4f}"
                f"  val_unc_mse={val_metrics['val_unc_mse']:.4f}"
                f"  unc_corr={val_metrics['val_unc_corr']:.4f}"
            )

        if val_metrics["val_mse"] < best_val_mse - cfg.min_delta:
            best_val_mse = val_metrics["val_mse"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint_model(
                checkpoint_dir / "variable_aware_patch_uncertainty_best.pt",
                model,
                cfg,
                epoch,
                history,
                normalizer,
                model_cfg=model_cfg,
                n_x=cfg.n_x,
                n_y=cfg.n_y,
                latent_dim=cfg.latent_dim,
            )
        else:
            patience_counter += 1

        if cfg.early_stopping and patience_counter >= cfg.patience:
            if verbose:
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best val MSE: {best_val_mse:.4f} at epoch {best_epoch}."
                )
            break

    save_checkpoint_model(
        checkpoint_dir / "variable_aware_patch_uncertainty_last.pt",
        model,
        cfg,
        epoch,
        history,
        normalizer,
        model_cfg=model_cfg,
        n_x=cfg.n_x,
        n_y=cfg.n_y,
        latent_dim=cfg.latent_dim,
    )
    export_history(history, checkpoint_dir)

    # ---- reload best weights -------------------------------------------------
    best_ckpt = torch.load(
        checkpoint_dir / "variable_aware_patch_uncertainty_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\n  best val epoch: {best_epoch}/{cfg.n_epochs}")

    # ---- ore-only adapter for existing drill-bin / step / no-ore helpers -----
    ore_wrapper = _OreWrapper(model)

    # ---- drill-bin metrics (best model) --------------------------------------
    bin_metrics = validate_e2e_map_by_drill_bins(
        ore_wrapper, val_loader, device, normalizer
    )
    if bin_metrics:
        if verbose:
            print("\nValidation metrics by drill count (best model, ore-value space):")
            for lo, hi in DRILL_BINS:
                key = f"{lo}_{hi}"
                n = bin_metrics.get(f"n_{key}", 0)
                mse = bin_metrics.get(f"mse_{key}", float("nan"))
                mae = bin_metrics.get(f"mae_{key}", float("nan"))
                corr = bin_metrics.get(f"corr_{key}", float("nan"))
                print(
                    f"  drills {lo:2d}-{hi:2d}"
                    f"  n={n:5d}"
                    f"  mse={mse:.4f}"
                    f"  mae={mae:.4f}"
                    f"  corr={corr:.4f}"
                )
        bin_path = checkpoint_dir / "val_metrics_by_drills.json"
        with open(bin_path, "w") as f:
            json.dump(bin_metrics, f, indent=2)
        if verbose:
            print(f"  drill-bin metrics -> {bin_path}")

    # ---- per-step metrics ----------------------------------------------------
    step_metrics = validate_e2e_map_by_step(
        ore_wrapper, val_loader, device, normalizer
    )
    if step_metrics:
        label = "step" if cfg.use_sequential_dataset else "drill count"
        if verbose:
            print(f"\nValidation metrics by {label} (best model, ore-value space):")
            for k, m in sorted(step_metrics.items()):
                print(
                    f"  K={k:3d}  n={m['n']:5d}"
                    f"  mse={m['mse']:.4f}"
                    f"  mae={m['mae']:.4f}"
                    f"  corr={m['corr']:.4f}"
                )
        step_path = checkpoint_dir / "val_metrics_by_step.json"
        with open(step_path, "w") as f:
            json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)
        if verbose:
            print(f"  step metrics -> {step_path}")

    # ---- no-ore false-positive metrics ---------------------------------------
    no_ore_metrics = validate_no_ore_e2e_map(
        ore_wrapper,
        val_loader,
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    # ---- uncertainty calibration + correlation diagnostics (final) -----------
    final_val  = _validate_uncertainty_e2e_map(model, val_loader, device, normalizer)
    corr_diag  = _compute_uncertainty_correlation_diagnostics(
        model, val_loader, normalizer, device
    )
    all_unc_metrics = {**final_val, **corr_diag}
    unc_metrics_path = checkpoint_dir / "val_metrics_uncertainty.json"
    with open(unc_metrics_path, "w") as f:
        json.dump(all_unc_metrics, f, indent=2)
    if verbose:
        print(
            f"\nUncertainty calibration (best model, normalised space):"
            f"\n  unc_mse             : {final_val['val_unc_mse']:.4f}"
            f"\n  unc_mae             : {final_val['val_unc_mae']:.4f}"
            f"\n  unc_corr            : {final_val['val_unc_corr']:.4f}"
            f"\n  top10_ratio         : {final_val['val_unc_top10_ratio']:.4f}"
            f"  (>1.0 = high-unc cells have higher error)"
            f"\n  corr_unc_abs_err    : {corr_diag['corr_unc_abs_err']:.4f}"
            f"\n  corr_unc_scaled_err : {corr_diag['corr_unc_scaled_err']:.4f}"
            f"\n  corr_unc_pred_ore   : {corr_diag['corr_unc_pred_ore']:.4f}"
            f"\n  corr_unc_ore_grad   : {corr_diag['corr_unc_ore_grad']:.4f}"
            f"\n  uncertainty metrics -> {unc_metrics_path}"
        )

    # ---- per-drill-count uncertainty analysis --------------------------------
    unc_by_dc = _analyze_uncertainty_by_drill_count(model, val_ds, normalizer, device)
    dc_path = checkpoint_dir / "val_uncertainty_by_step.json"
    with open(dc_path, "w") as f:
        json.dump(unc_by_dc, f, indent=2)
    if verbose:
        print(f"  drill-count uncertainty -> {dc_path}")

    # ---- training sample inspection ------------------------------------------
    _save_training_samples(train_ds, normalizer, checkpoint_dir)

    # ---- optional validation plots (saved to run directory) ------------------
    if plot_dir is not None:
        _save_uncertainty_val_plots(
            model, val_ds, normalizer, checkpoint_dir, device, n_plots=cfg.n_val_plots
        )

    if verbose:
        print(
            f"\nTraining complete.  Best val MSE: {best_val_mse:.4f}"
            f"  (epoch {best_epoch}/{cfg.n_epochs})"
        )
        print(f"  run directory -> {checkpoint_dir}")

    return model, normalizer, checkpoint_dir


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_variable_aware_patch_uncertainty_borehole_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[
    VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    VariableAwarePatchBoreholeUncertaintyE2ETrainingConfig,
    TargetNormalizer,
    list[dict],
]:
    """Load a VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(
        ckpt: dict,
    ) -> VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer:
        if "model_cfg" in ckpt:
            return VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer(
                ckpt["model_cfg"]
            )
        return VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer(
            ckpt["cfg"].to_model_config()
        )

    return load_model_encoder_checkpoint(
        path,
        _model_fn,
        VariableAwarePatchBoreholeUncertaintyE2ETrainingConfig,
        device,
    )
