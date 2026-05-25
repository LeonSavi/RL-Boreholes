"""Training pipeline for UNetBelief.

Trains the convolutional U-Net reconstruction model end-to-end using MSE
loss in normalised target space, with optional latent normalisation
dimensionality reduction, and spatial coordinate channels.

Checkpoint files: ``belief_best.pt``, ``belief_last.pt``
"""

from __future__ import annotations

import json
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..datasets import GeologicalBeliefDataset
from ..models.map_encoders.unet_belief import UNetBelief
from ..utils import TargetNormalizer
from ..baselines import evaluate_baselines
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    validate_no_ore,
    false_positive_loss,
    save_no_ore_metrics,
    save_val_plots,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from .training_configs import NeuralBeliefTrainingConfig


def train_neural_belief(
    cfg: NeuralBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: GeologicalBeliefDataset | None = None,
    val_ds: GeologicalBeliefDataset | None = None,
    normalizer: TargetNormalizer | None = None,
) -> tuple[UNetBelief, TargetNormalizer]:
    """Train the neural geological belief updater end-to-end.

    Saves two checkpoints to ``checkpoint_dir``:
      * ``belief_best.pt``  — lowest validation MSE (ore-value space)
      * ``belief_last.pt``  — final epoch

    Parameters
    ----------
    resources      : JEPA model + map-generation components
    cfg            : training hyper-parameters
    device         : torch device string
    checkpoint_dir : directory for saved checkpoints
    plot_dir       : if given, save 4 validation plots here after training
    verbose        : print per-epoch metrics

    Returns
    -------
    (trained UNetBelief with best weights, fitted TargetNormalizer)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- datasets -------------------------------------------------------------
    if train_ds is None or val_ds is None:
        raise ValueError("train_ds and val_ds must be provided.")

    n_x, n_y = train_ds.targets.shape[2], train_ds.targets.shape[3]

    # ---- target normalization -------------------------------------------------
    if normalizer is None:
        raise ValueError("normalizer must be provided.")

    # ---- coordinate channels -------------------------------------------------
    if cfg.use_coordinate_channels:
        train_ds.apply_coordinate_channels()
        val_ds.apply_coordinate_channels()
        cfg.in_channels += 2
        if verbose:
            print(f"  coord channels: enabled  →  in_channels={cfg.in_channels}")
    else:
        if verbose:
            print("  coord channels: disabled")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ----------------------------------------------------------------
    model = UNetBelief(in_channels=cfg.in_channels, base_channels=cfg.base_channels).to(
        device
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        print(f"  model params  : {sum(p.numel() for p in model.parameters()):,}")

    # ---- training loop --------------------------------------------------------
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch = 0
    history: list[dict] = []

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)  # loss in normalized space
            if cfg.use_false_positive_penalty:
                loss = loss + cfg.false_positive_weight * false_positive_loss(
                    pred, y, normalizer
                )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_mse_norm = train_loss_sum / n_batches
        val_metrics = validate(model, val_loader, device, normalizer)

        row = {"epoch": epoch, "train_mse_norm": train_mse_norm, **val_metrics}
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{cfg.n_epochs}"
                f"  train_mse(norm)={train_mse_norm:.4f}"
                f"  val_mse={val_metrics['val_mse']:.4f}"
                f"  val_mae={val_metrics['val_mae']:.4f}"
                f"  val_corr={val_metrics['val_corr']:.4f}"
            )

        if val_metrics["val_mse"] < best_val_mse - cfg.min_delta:
            best_val_mse = val_metrics["val_mse"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint_model(
                checkpoint_dir / "belief_best.pt",
                model,
                cfg,
                epoch,
                history,
                normalizer,
                n_x=n_x,
                n_y=n_y,
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
        checkpoint_dir / "belief_last.pt",
        model,
        cfg,
        epoch,
        history,
        normalizer,
        n_x=n_x,
        n_y=n_y,
        latent_dim=cfg.latent_dim,
    )

    # ---- baseline comparison --------------------------------------------------
    if verbose:
        print("\nBaseline comparison (val set, ore-value space):")
        for name, m in evaluate_baselines(val_loader, normalizer).items():
            print(
                f"  {name:22s}  mse={m['mse']:.4f}  mae={m['mae']:.4f}  corr={m['corr']:.4f}"
            )
        best = min(history, key=lambda r: r["val_mse"])
        print(
            f"  {'neural_belief':22s}  mse={best['val_mse']:.4f}"
            f"  mae={best['val_mae']:.4f}  corr={best['val_corr']:.4f}"
            f"  (epoch {best['epoch']})"
        )

    # ---- optional plots -------------------------------------------------------
    if plot_dir is not None:
        if cfg.use_sequential_dataset:
            from ..sequential_eval import save_sequential_val_plots

            save_sequential_val_plots(
                model,
                val_ds,
                normalizer,
                plot_dir,
                device,
                n_sequences=cfg.n_val_plots,
            )
        else:
            save_val_plots(
                model, val_ds, normalizer, plot_dir, device, n_plots=cfg.n_val_plots
            )

    # Reload best weights before returning
    best_ckpt = torch.load(
        checkpoint_dir / "belief_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\nTraining complete. Best val MSE: {best_val_mse:.4f}")
        print(f"  checkpoints -> {checkpoint_dir}")

    # ---- per-bin validation ---------------------------------------------------
    bin_metrics = validate_by_drill_bins(model, val_ds, normalizer, device)
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

    # ---- sequential step metrics -----------------------------------------------
    if cfg.use_sequential_dataset:
        from ..sequential_eval import validate_by_step

        step_metrics = validate_by_step(model, val_ds, normalizer, device)
        if step_metrics:
            if verbose:
                print("\nValidation metrics by step (best model, ore-value space):")
                for step, m in sorted(step_metrics.items()):
                    print(
                        f"  step {step:3d}  n={m['n']:5d}"
                        f"  mse={m['mse']:.4f}  mae={m['mae']:.4f}"
                        f"  corr={m['corr']:.4f}"
                    )
            step_path = checkpoint_dir / "val_metrics_by_step.json"
            with open(step_path, "w") as f:
                json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)
            if verbose:
                print(f"  step metrics -> {step_path}")

    # ---- no-ore false-positive metrics -----------------------------------------
    no_ore_metrics = validate_no_ore(
        model,
        val_loader,
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    if cfg.use_sequential_dataset:
        from ..sequential_eval import validate_no_ore_by_step

        no_ore_step = validate_no_ore_by_step(
            model,
            val_ds,
            normalizer,
            device,
            threshold=cfg.false_positive_threshold,
        )
        if no_ore_step:
            if verbose:
                print("\nNo-ore metrics by step (best model):")
                for step, m in sorted(no_ore_step.items()):
                    print(
                        f"  step {step:3d}  n={m['no_ore_n']:5d}"
                        f"  pred_total={m['no_ore_pred_total']:.4f}"
                        f"  pred_max={m['no_ore_pred_max']:.4f}"
                        f"  fp_area={m['no_ore_fp_area']:.4f}"
                    )
            no_ore_step_path = checkpoint_dir / "val_metrics_no_ore_by_step.json"
            with open(no_ore_step_path, "w") as f:
                json.dump({str(k): v for k, v in no_ore_step.items()}, f, indent=2)
            if verbose:
                print(f"  no-ore by-step metrics -> {no_ore_step_path}")

    return model, normalizer


def load_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[UNetBelief, NeuralBeliefTrainingConfig, TargetNormalizer, list[dict]]:
    def _model_fn(ckpt: dict) -> UNetBelief:
        cfg = ckpt["cfg"]
        return UNetBelief(in_channels=cfg.in_channels, base_channels=cfg.base_channels)

    return load_model_encoder_checkpoint(
        path, _model_fn, NeuralBeliefTrainingConfig, device
    )
