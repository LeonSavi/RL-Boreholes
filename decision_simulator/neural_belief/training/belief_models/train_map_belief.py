"""Training pipeline for MapBeliefTransformer.

Trains the transformer-based geological belief encoder end-to-end using MSE
loss in normalised target space, with gradient clipping for stability.

Key differences from train_unet_belief.py:
  * Gradient clipping is enabled by default (transformers are sensitive to
    gradient explosions, especially in the early epochs).
  * Smaller default batch size (16 vs 32) to keep transformer memory footprint
    manageable on a typical research GPU.
  * Lower default learning rate (1e-4 vs 3e-4) — transformers generally train
    more stably with a smaller initial lr.
  * No PCA or coordinate-channel preprocessing: coordinates are computed
    internally by SpatialTokenEmbedding, and PCA would invalidate the latent
    dim stored in MapBeliefConfig.

Checkpoint files: ``map_belief_best.pt``, ``map_belief_last.pt``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...datasets import GeologicalBeliefDataset
from ...utils import TargetNormalizer
from ...baselines import evaluate_baselines
from ...training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    validate_no_ore,
    false_positive_loss,
    save_no_ore_metrics,
    save_val_plots,
    export_history,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from ...models.belief_models.map_encoders.map_belief_transformer import MapBeliefTransformer
from .training_configs import MapBeliefTrainingConfig

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_map_belief(
    cfg: MapBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: GeologicalBeliefDataset | None = None,
    val_ds: GeologicalBeliefDataset | None = None,
    normalizer: TargetNormalizer | None = None,
) -> tuple[MapBeliefTransformer, TargetNormalizer]:
    """Train the MapBeliefTransformer end-to-end.

    Saves two checkpoints to ``checkpoint_dir``:
      * ``map_belief_best.pt``  — lowest validation MSE (ore-value space)
      * ``map_belief_last.pt``  — final epoch

    The signature is intentionally identical to ``train_neural_belief()`` so
    that the two can be swapped without changing calling code.

    Parameters
    ----------
    resources      : JEPA model + map-generation components
    cfg            : training hyper-parameters
    device         : torch device string  (e.g. "cuda" or "cpu")
    checkpoint_dir : directory for saved checkpoints
    plot_dir       : if given, save 4 validation plots here after training
    verbose        : print per-epoch metrics
    train_ds / val_ds : optional pre-built datasets (skip generation if provided)

    Returns
    -------
    (trained MapBeliefTransformer with best weights, fitted TargetNormalizer)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- datasets ----------------------------------------------------------------
    if train_ds is None or val_ds is None:
        raise ValueError("train_ds and val_ds must be provided.")

    n_x, n_y = train_ds.targets.shape[2], train_ds.targets.shape[3]

    # ---- target normalisation ---------------------------------------------------
    if normalizer is None:
        raise ValueError("normalizer must be provided.")

    # ---- data loaders -----------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ------------------------------------------------------------------
    cfg.n_x = n_x
    cfg.n_y = n_y
    model_cfg = cfg.to_model_config()
    model = MapBeliefTransformer(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        print(f"  model params  : {sum(p.numel() for p in model.parameters()):,}")
        print(f"  d_model       : {cfg.d_model}")
        print(f"  n_layers      : {cfg.n_encoder_layers}")
        print(f"  n_heads       : {cfg.n_heads}")

    # ---- training loop ----------------------------------------------------------
    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)  # MSE in normalised space
            if cfg.use_false_positive_penalty:
                loss = loss + cfg.false_positive_weight * false_positive_loss(
                    pred, y, normalizer
                )

            optimiser.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

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
                checkpoint_dir / "map_belief_best.pt",
                model,
                cfg,
                epoch,
                history,
                normalizer,
                model_cfg=model_cfg,
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
        checkpoint_dir / "map_belief_last.pt",
        model,
        cfg,
        epoch,
        history,
        normalizer,
        model_cfg=model_cfg,
        n_x=n_x,
        n_y=n_y,
        latent_dim=cfg.latent_dim,
    )

    # ---- export training history ------------------------------------------------
    export_history(history, checkpoint_dir)

    # ---- baseline comparison ----------------------------------------------------
    if verbose:
        print("\nBaseline comparison (val set, ore-value space):")
        for name, m in evaluate_baselines(val_loader, normalizer).items():
            print(
                f"  {name:22s}  mse={m['mse']:.4f}  mae={m['mae']:.4f}"
                f"  corr={m['corr']:.4f}"
            )
        best_row = min(history, key=lambda r: r["val_mse"])
        print(
            f"  {'map_belief_transformer':22s}  mse={best_row['val_mse']:.4f}"
            f"  mae={best_row['val_mae']:.4f}  corr={best_row['val_corr']:.4f}"
            f"  (epoch {best_row['epoch']})"
        )

    # ---- optional validation plots ----------------------------------------------
    if plot_dir is not None:
        if cfg.use_sequential_dataset:
            from ...sequential_eval import save_sequential_val_plots

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

    # ---- per-bin validation (best model) ----------------------------------------
    # Reload best weights first
    best_ckpt = torch.load(
        checkpoint_dir / "map_belief_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

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
        from ...sequential_eval import validate_by_step

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
        from ...sequential_eval import validate_no_ore_by_step

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

    if verbose:
        print(f"\nTraining complete.  Best val MSE: {best_val_mse:.4f}")
        print(f"  checkpoints -> {checkpoint_dir}")

    return model, normalizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_map_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[MapBeliefTransformer, MapBeliefTrainingConfig, TargetNormalizer, list[dict]]:
    def _model_fn(ckpt: dict) -> MapBeliefTransformer:
        if "model_cfg" in ckpt:
            return MapBeliefTransformer(ckpt["model_cfg"])
        return MapBeliefTransformer(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(
        path, _model_fn, MapBeliefTrainingConfig, device
    )
