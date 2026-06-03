"""Training pipeline for VariableAwarePatchBoreholeEndToEndMapBeliefTransformer.

Variable-aware patch experiment: identical training setup to the CLS-token patch
experiment (train_patch_borehole_cls_transformer.py) but the borehole encoder
creates one token per (variable, depth patch) instead of one flattened token per
depth patch.  A learned variable embedding is added to each token so the
transformer can explicitly model cross-variable interactions.

This file is intentionally parallel to train_patch_borehole_cls_transformer.py.
Dataset, collate, validation, and loss logic are fully reused.

Checkpoints
-----------
variable_aware_patch_best.pt  — lowest validation MSE (ore-value space)
variable_aware_patch_last.pt  — final epoch
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decision_simulator.resources import DecisionSimulationResources
from ....training_utils import TargetNormalizer
from ....training_utils import (
    false_positive_loss,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from ....models.belief_models.end_to_end.variable_aware_patch_borehole_transformer import (
    VariableAwarePatchBoreholeEndToEndMapBeliefTransformer,
)
from .train_end_to_end_map_belief import E2EMapDataset
from .helpers import validate_e2e_map, model_validation, collate_e2e_map
from ..training_configs import VariableAwarePatchBoreholeConfig


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_variable_aware_patch_borehole_transformer(
    resources: DecisionSimulationResources,
    cfg: VariableAwarePatchBoreholeConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: E2EMapDataset | None = None,
    val_ds: E2EMapDataset | None = None,
) -> tuple[VariableAwarePatchBoreholeEndToEndMapBeliefTransformer, TargetNormalizer]:
    """Train VariableAwarePatchBoreholeEndToEndMapBeliefTransformer.

    Saves two checkpoints to checkpoint_dir:
      variable_aware_patch_best.pt  — lowest validation MSE (ore-value space)
      variable_aware_patch_last.pt  — final epoch

    Parameters
    ----------
    resources        : shared resources (norm_stats for borehole standardisation)
    cfg              : training hyperparameters
    device           : torch device string
    checkpoint_dir   : directory for saved checkpoints
    plot_dir         : if given, save validation plots here after training
    verbose          : print per-epoch metrics
    train_ds/val_ds  : pre-built E2EMapDataset (required)

    Returns
    -------
    (trained model with best weights, fitted TargetNormalizer)
    """
    checkpoint_dir = Path(checkpoint_dir)
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
    model = VariableAwarePatchBoreholeEndToEndMapBeliefTransformer(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        n_bh_tokens = cfg.n_variables * n_patches
        print(f"  model params  : {n_params:,}")
        print(f"  variable_aware: True")
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
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)  # (B, 1, n_x, n_y) normalised

            pred = model(bh, ov, pos, pm)  # (B, 1, n_x, n_y)
            loss = F.mse_loss(pred, tgt)

            if cfg.use_false_positive_penalty:
                loss = loss + cfg.false_positive_weight * false_positive_loss(
                    pred, tgt, normalizer
                )

            optimiser.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimiser.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_mse_norm = train_loss_sum / n_batches
        val_metrics = validate_e2e_map(model, val_loader, device, normalizer)

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
                checkpoint_dir / "variable_aware_patch_best.pt",
                model,
                cfg,
                epoch,
                history,
                normalizer,
                model_cfg=model_cfg,
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

    model_validation(
        model, checkpoint_dir, "variable_aware_patch_best.pt", history,
        val_loader, device, normalizer,
        false_positive_threshold=cfg.false_positive_threshold,
        use_sequential_dataset=cfg.use_sequential_dataset,
        verbose=verbose,
        best_val_mse=best_val_mse,
        best_epoch=best_epoch,
        n_epochs=cfg.n_epochs,
        plot_dir=plot_dir,
        val_ds=val_ds,
        n_val_plots=cfg.n_val_plots,
    )
    return model, normalizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_variable_aware_patch_borehole_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[
    VariableAwarePatchBoreholeEndToEndMapBeliefTransformer,
    VariableAwarePatchBoreholeConfig,
    TargetNormalizer,
    list[dict],
]:
    """Load a VariableAwarePatchBoreholeEndToEndMapBeliefTransformer checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(ckpt: dict) -> VariableAwarePatchBoreholeEndToEndMapBeliefTransformer:
        if "model_cfg" in ckpt:
            return VariableAwarePatchBoreholeEndToEndMapBeliefTransformer(ckpt["model_cfg"])
        return VariableAwarePatchBoreholeEndToEndMapBeliefTransformer(
            ckpt["cfg"].to_model_config()
        )

    return load_model_encoder_checkpoint(
        path, _model_fn, VariableAwarePatchBoreholeConfig, device
    )
