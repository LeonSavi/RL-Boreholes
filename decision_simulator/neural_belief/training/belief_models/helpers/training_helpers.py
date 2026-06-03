from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ....utils import TargetNormalizer
from ....training_utils import DRILL_BINS, export_history, save_no_ore_metrics
from .validation_helpers import (
    validate_e2e_map_by_drill_bins,
    validate_e2e_map_by_step,
    validate_no_ore_e2e_map,
)
from .validation_plots import save_e2e_map_sequential_val_plots, save_e2e_map_val_plots


def model_validation(
    model: nn.Module,
    checkpoint_dir: Path,
    best_ckpt_filename: str,
    history: dict,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    *,
    false_positive_threshold: float = 0.0,
    use_sequential_dataset: bool = False,
    verbose: bool = True,
    best_val_mse: float | None = None,
    best_epoch: int | None = None,
    n_epochs: int | None = None,
    plot_dir: Path | None = None,
    val_ds: Dataset | None = None,
    n_val_plots: int = 5,
) -> None:
    export_history(history, checkpoint_dir)

    best_ckpt = torch.load(
        checkpoint_dir / best_ckpt_filename, map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose and best_epoch is not None and n_epochs is not None:
        print(f"\n  best val epoch: {best_epoch}/{n_epochs}")

    # drill-bin metrics
    bin_metrics = validate_e2e_map_by_drill_bins(model, val_loader, device, normalizer)
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
                    f"  drills {lo:2d}-{hi:2d}  n={n:5d}"
                    f"  mse={mse:.4f}  mae={mae:.4f}  corr={corr:.4f}"
                )
        bin_path = checkpoint_dir / "val_metrics_by_drills.json"
        with open(bin_path, "w") as f:
            json.dump(bin_metrics, f, indent=2)
        if verbose:
            print(f"  drill-bin metrics -> {bin_path}")

    # per-step metrics
    step_metrics = validate_e2e_map_by_step(model, val_loader, device, normalizer)
    if step_metrics:
        label = "step" if use_sequential_dataset else "drill count"
        if verbose:
            print(f"\nValidation metrics by {label} (best model, ore-value space):")
            for k, m in sorted(step_metrics.items()):
                print(
                    f"  K={k:3d}  n={m['n']:5d}"
                    f"  mse={m['mse']:.4f}  mae={m['mae']:.4f}  corr={m['corr']:.4f}"
                )
        step_path = checkpoint_dir / "val_metrics_by_step.json"
        with open(step_path, "w") as f:
            json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)
        if verbose:
            print(f"  step metrics -> {step_path}")

    # no-ore false-positive metrics
    no_ore_metrics = validate_no_ore_e2e_map(
        model, val_loader, device, normalizer, threshold=false_positive_threshold
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    # optional validation plots
    if plot_dir is not None and val_ds is not None:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        if use_sequential_dataset:
            save_e2e_map_sequential_val_plots(
                model, val_ds, normalizer, plot_dir, device, n_sequences=n_val_plots
            )
        else:
            save_e2e_map_val_plots(
                model, val_ds, normalizer, plot_dir, device, n_plots=n_val_plots
            )

    if verbose:
        mse_str = f"{best_val_mse:.4f}" if best_val_mse is not None else "n/a"
        epoch_str = f"(epoch {best_epoch}/{n_epochs})" if best_epoch is not None else ""
        print(f"\nTraining complete.  Best val MSE: {mse_str}  {epoch_str}".rstrip())
        print(f"  checkpoints -> {checkpoint_dir}")


def collate_e2e_map(batch: list[dict]) -> dict:
    """Pad borehole sequences to the longest K in the batch.

    Returns
    -------
    dict with keys:
        boreholes    (B, max_K, V, D) float32
        ore_vals     (B, max_K)       float32  — 0 at padding
        positions    (B, max_K, 2)    float32  — 0 at padding
        padding_mask (B, max_K)       bool     — True at padded rows
        target_map   (B, 1, n_x, n_y) float32
        drill_counts (B,)             int64
    """
    max_K = max(s["boreholes"].shape[0] for s in batch)
    B = len(batch)
    V, D = batch[0]["boreholes"].shape[1], batch[0]["boreholes"].shape[2]
    n_x, n_y = batch[0]["target_map"].shape[0], batch[0]["target_map"].shape[1]

    boreholes_pad = np.zeros((B, max_K, V, D), dtype=np.float32)
    ore_pad = np.zeros((B, max_K), dtype=np.float32)
    pos_pad = np.zeros((B, max_K, 2), dtype=np.float32)
    padding_mask = np.ones((B, max_K), dtype=bool)  # True = padded
    target_maps = np.zeros((B, 1, n_x, n_y), dtype=np.float32)
    drill_counts = np.zeros(B, dtype=np.int64)

    for i, s in enumerate(batch):
        K = s["boreholes"].shape[0]
        boreholes_pad[i, :K] = s["boreholes"]
        ore_pad[i, :K] = s["ore_vals"]
        pos_pad[i, :K] = s["positions"]
        padding_mask[i, :K] = False
        target_maps[i, 0] = s["target_map"]
        drill_counts[i] = s.get("drill_count", K)

    return {
        "boreholes": torch.from_numpy(boreholes_pad),
        "ore_vals": torch.from_numpy(ore_pad),
        "positions": torch.from_numpy(pos_pad),
        "padding_mask": torch.from_numpy(padding_mask),
        "target_map": torch.from_numpy(target_maps),
        "drill_counts": torch.from_numpy(drill_counts),
    }
