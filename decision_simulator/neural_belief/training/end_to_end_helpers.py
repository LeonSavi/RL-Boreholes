"""Shared validation and plot helpers for end-to-end map belief experiments.

These helpers work with any model that exposes the same forward signature as
EndToEndMapBeliefTransformer / PatchBoreholeEndToEndMapBeliefTransformer:

    model(boreholes, ore_vals, positions, padding_mask) -> (B, 1, n_x, n_y)
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import TargetNormalizer
from ..training_utils import DRILL_BINS, pearson_correlation


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_e2e_map(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Compute MSE, MAE, and Pearson in ore-value (denormalised) space."""
    model.eval()
    mse_total = mae_total = corr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)

            mse_total += F.mse_loss(pred, tgt_raw).item()
            mae_total += (pred - tgt_raw).abs().mean().item()
            corr_total += pearson_correlation(pred, tgt_raw)
            n_batches += 1

    return {
        "val_mse": mse_total / n_batches,
        "val_mae": mae_total / n_batches,
        "val_corr": corr_total / n_batches,
    }


def _validate_e2e_map_by_drill_bins(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    bins: list[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Compute MSE/MAE/Pearson grouped by number of drilled boreholes."""
    if bins is None:
        bins = DRILL_BINS

    model.eval()
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            all_pred.append(normalizer.inverse_tensor(pred_norm).cpu())
            all_tgt.append(normalizer.inverse_tensor(tgt).cpu())
            all_counts.append(batch["drill_counts"])

    preds = torch.cat(all_pred, dim=0)     # (N, 1, n_x, n_y)
    tgts = torch.cat(all_tgt, dim=0)      # (N, 1, n_x, n_y)
    counts = torch.cat(all_counts, dim=0)  # (N,)

    result: dict[str, float | int] = {}
    for lo, hi in bins:
        key = f"{lo}_{hi}"
        sel = (counts >= lo) & (counts <= hi)
        n = int(sel.sum().item())
        result[f"n_{key}"] = n
        if n == 0:
            result[f"mse_{key}"] = float("nan")
            result[f"mae_{key}"] = float("nan")
            result[f"corr_{key}"] = float("nan")
            continue
        p = preds[sel]
        t = tgts[sel]
        result[f"mse_{key}"] = F.mse_loss(p, t).item()
        result[f"mae_{key}"] = (p - t).abs().mean().item()
        result[f"corr_{key}"] = pearson_correlation(p, t)

    return result


def _validate_no_ore_e2e_map(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """False-positive metrics on samples whose true ore map is entirely zero."""
    model.eval()
    pred_totals: list[torch.Tensor] = []
    pred_maxes: list[torch.Tensor] = []
    fp_areas: list[torch.Tensor] = []
    n_no_ore = 0

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)

            B = pred.shape[0]
            no_ore = tgt_raw.view(B, -1).sum(dim=1) == 0
            if not no_ore.any():
                continue

            p = pred[no_ore].cpu()
            n = p.shape[0]
            n_no_ore += n
            pv = p.view(n, -1)
            pred_totals.append(pv.sum(dim=1))
            pred_maxes.append(pv.max(dim=1).values)
            fp_areas.append((pv > threshold).float().mean(dim=1))

    if n_no_ore == 0:
        return {
            "no_ore_n": 0,
            "no_ore_pred_total": float("nan"),
            "no_ore_pred_max": float("nan"),
            "no_ore_fp_area": float("nan"),
        }

    return {
        "no_ore_n": n_no_ore,
        "no_ore_pred_total": torch.cat(pred_totals).mean().item(),
        "no_ore_pred_max": torch.cat(pred_maxes).mean().item(),
        "no_ore_fp_area": torch.cat(fp_areas).mean().item(),
    }


def _validate_e2e_map_by_step(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[int, dict[str, float]]:
    """MSE/MAE/Pearson grouped by exact drill count K.

    In sequential mode K equals the prefix step, giving per-step metrics.
    In random mode K is drawn from [min_drills, max_drills].
    """
    model.eval()
    groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm).cpu()
            tgt_raw = normalizer.inverse_tensor(tgt).cpu()
            ks = batch["drill_counts"].tolist()

            for k, p, t in zip(ks, pred, tgt_raw):
                groups[int(k)][0].append(p.unsqueeze(0))
                groups[int(k)][1].append(t.unsqueeze(0))

    metrics: dict[int, dict[str, float]] = {}
    for k in sorted(groups):
        ps = torch.cat(groups[k][0])  # (n, 1, n_x, n_y)
        ts = torch.cat(groups[k][1])
        mse = F.mse_loss(ps, ts).item()
        mae = (ps - ts).abs().mean().item()
        corr = pearson_correlation(ps, ts)
        metrics[k] = {"n": len(ps), "mse": mse, "mae": mae, "corr": corr}

    return metrics


# ---------------------------------------------------------------------------
# Validation plots
# ---------------------------------------------------------------------------

def _save_e2e_map_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 20,
) -> None:
    """Save n_plots 4-panel belief-map figures to a timestamped subdirectory."""
    import matplotlib
    matplotlib.use("Agg")

    from ..visualize import plot_belief_sample

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(plot_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    indices = np.linspace(0, len(val_ds) - 1, n_plots, dtype=int)

    for plot_k, idx in enumerate(indices):
        sample = val_ds.samples[int(idx)]

        bh = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)  # (1, K, V, D)
        ov = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)   # (1, K)
        pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device) # (1, K, 2)

        with torch.no_grad():
            pred_norm = model(bh, ov, pos)  # (1, 1, n_x, n_y) — no padding needed
        pred_ore_map = normalizer.inverse(pred_norm.squeeze().cpu().numpy())

        n_x, n_y = sample["target_map"].shape
        sparse_ore_map = np.zeros((n_x, n_y), dtype=np.float32)
        observation_mask = np.zeros((n_x, n_y), dtype=np.float32)
        for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
            i = int(round(float(px) * (n_x - 1)))
            j = int(round(float(py) * (n_y - 1)))
            sparse_ore_map[i, j] = float(ov_val)
            observation_mask[i, j] = 1.0

        true_ore_map = normalizer.inverse(sample["target_map"])

        plot_belief_sample(
            sparse_ore_map=sparse_ore_map,
            observation_mask=observation_mask,
            true_ore_map=true_ore_map,
            predicted_ore_map=pred_ore_map,
            save_path=out_dir / f"val_sample_{plot_k:02d}.png",
            title=f"Val sample {plot_k}  ({sample['drill_count']} drills)",
            timestamp=timestamp,
        )

    print(f"  plots saved -> {out_dir}")


def _save_e2e_map_sequential_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_sequences: int = 3,
) -> None:
    """Save per-step belief-evolution plots for sequential validation sequences.

    For each selected (map, sequence) pair, saves:
    * individual PNGs per step: ``step_001.png``, ``step_003.png``, …
    * a combined ``evolution.png`` grid (rows=steps, cols=4 panels)

    Falls back to ``_save_e2e_map_val_plots`` when ``sequence_id`` is not
    present in the dataset samples (i.e. non-sequential datasets).
    """
    if not val_ds.samples or "sequence_id" not in val_ds.samples[0]:
        _save_e2e_map_val_plots(model, val_ds, normalizer, plot_dir, device, n_plots=n_sequences)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..visualize import plot_belief_sample

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(plot_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, s in enumerate(val_ds.samples):
        key = (int(s["map_idx"]), int(s["sequence_id"]))
        groups[key].append((int(s["drill_count"]), idx))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    all_keys = list(groups.keys())
    n_select = min(n_sequences, len(all_keys))
    sel_indices = np.linspace(0, len(all_keys) - 1, n_select, dtype=int)
    selected_keys = [all_keys[i] for i in sel_indices]

    model.eval()

    for map_idx, seq_id in selected_keys:
        seq_dir = base_dir / f"seq_{map_idx:04d}_{seq_id:02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        step_pairs = groups[(map_idx, seq_id)]
        n_steps = len(step_pairs)

        panel_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for step, sample_idx in step_pairs:
            sample = val_ds.samples[sample_idx]

            bh = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
            ov = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
            pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(bh, ov, pos)
            pred_ore = normalizer.inverse(pred_norm.squeeze().cpu().numpy())

            n_x, n_y = sample["target_map"].shape
            sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
            obs_mask = np.zeros((n_x, n_y), dtype=np.float32)
            for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
                i = int(round(float(px) * (n_x - 1)))
                j = int(round(float(py) * (n_y - 1)))
                sparse_ore[i, j] = float(ov_val)
                obs_mask[i, j] = 1.0

            true_ore = normalizer.inverse(sample["target_map"])

            plot_belief_sample(
                sparse_ore_map=sparse_ore,
                observation_mask=obs_mask,
                true_ore_map=true_ore,
                predicted_ore_map=pred_ore,
                save_path=seq_dir / f"step_{step:03d}.png",
                title=f"Map {map_idx} / Seq {seq_id} / Step {step} ({step} drills)",
                timestamp=timestamp,
            )

            panel_rows.append((sparse_ore, obs_mask, true_ore, pred_ore))

        global_vmax = max(float(max(t.max(), p.max())) for _, _, t, p in panel_rows)
        global_vmax = max(global_vmax, 1e-3)

        fig, axes = plt.subplots(n_steps, 4, figsize=(18, 4 * n_steps))
        if n_steps == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Belief evolution — Map {map_idx} / Seq {seq_id}"
            f"  (colour scale max = {global_vmax:.3f})",
            fontsize=12,
        )

        col_titles = ["Observations", "True ore map", "Predicted ore map", "Abs error"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        for row_idx, ((step, _), (sparse_ore, obs_mask, true_ore, pred_ore)) in enumerate(
            zip(step_pairs, panel_rows)
        ):
            kw = dict(origin="lower", cmap="viridis", vmin=0, vmax=global_vmax)
            drill_rows, drill_cols = np.where(obs_mask > 0)

            ax = axes[row_idx, 0]
            ax.imshow(sparse_ore.T, **kw)
            ax.scatter(drill_rows, drill_cols, c="red", s=8, marker="x", linewidths=0.6)
            ax.set_ylabel(f"step {step}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 1]
            ax.imshow(true_ore.T, **kw)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 2]
            im_pred = ax.imshow(pred_ore.T, **kw)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 3]
            abs_err = np.abs(pred_ore - true_ore)
            im_err = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=global_vmax, cmap="Reds")
            ax.set_xlabel(f"max err = {abs_err.max():.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_pred, ax=axes[:, 2], shrink=0.6, label="ore value")
        fig.colorbar(im_err, ax=axes[:, 3], shrink=0.6, label="abs error")
        fig.tight_layout()

        evo_path = seq_dir / "evolution.png"
        fig.savefig(evo_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  sequential plots -> {base_dir}")
