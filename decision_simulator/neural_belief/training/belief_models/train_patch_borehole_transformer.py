"""Training pipeline for PatchBoreholeEndToEndMapBeliefTransformer.

Patch-based experiment: replaces the CNN front-end of EndToEndMapBeliefTransformer
with a pure depth-patch tokeniser.  The depth axis is split into fixed-size
intervals; each patch is linearly projected to a token and passed to a
transformer.  All downstream map components are identical to the CNN baseline.

This file is intentionally parallel to train_end_to_end_map_belief.py.
Dataset, collate, and loss logic are reused directly; only the model class and
its config differ.

Checkpoints
-----------
patch_borehole_best.pt  — lowest validation MSE (ore-value space)
patch_borehole_last.pt  — final epoch
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from decision_simulator.resources import DecisionSimulationResources
from ..utils import TargetNormalizer
from ..training_utils import (
    DRILL_BINS,
    export_history,
    false_positive_loss,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
    save_no_ore_metrics,
)
from ..models.end_to_end.patch_borehole_transformer import (
    PatchBoreholeEndToEndConfig,
    PatchBoreholeEndToEndMapBeliefTransformer,
)
from .train_end_to_end_map_belief import E2EMapDataset, collate_e2e_map
from .end_to_end_helpers import (
    _validate_e2e_map,
    _validate_e2e_map_by_drill_bins,
    _validate_no_ore_e2e_map,
    _validate_e2e_map_by_step,
    _save_e2e_map_val_plots,
    _save_e2e_map_sequential_val_plots,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class PatchBoreholeE2ETrainingConfig:
    """Training hyperparameters for PatchBoreholeEndToEndMapBeliefTransformer.

    Identical to E2EMapBeliefTrainingConfig except that ``bh_channels`` is
    replaced by ``bh_patch_size`` for the patch-based borehole encoder.
    """

    # Dataset
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # Sequential dataset mode: ordered drill sequences at fixed prefix lengths
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])

    # Borehole dimensions — resolved from resources at training time
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder: depth patch size
    bh_patch_size: int = 20

    # Borehole encoder: transformer over patch tokens
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2

    # Borehole embedding output dimension
    latent_dim: int = 128

    # Spatial grid — set automatically from cache in train_patch_borehole_transformer()
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

    # False-positive penalty
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    false_positive_threshold: float = 0.05

    # Misc
    seed: int = 42
    borehole_encoder: str = "patch_borehole"
    n_val_plots: int = 20

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    def to_model_config(self) -> PatchBoreholeEndToEndConfig:
        """Build a PatchBoreholeEndToEndConfig for model construction."""
        return PatchBoreholeEndToEndConfig(
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
# Main training function
# ---------------------------------------------------------------------------

def train_patch_borehole_transformer(
    resources: DecisionSimulationResources,
    cfg: PatchBoreholeE2ETrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: E2EMapDataset | None = None,
    val_ds: E2EMapDataset | None = None,
) -> tuple[PatchBoreholeEndToEndMapBeliefTransformer, TargetNormalizer]:
    """Train PatchBoreholeEndToEndMapBeliefTransformer with a full-map reconstruction objective.

    Saves two checkpoints to checkpoint_dir:
      patch_borehole_best.pt  — lowest validation MSE (ore-value space)
      patch_borehole_last.pt  — final epoch

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
    (trained PatchBoreholeEndToEndMapBeliefTransformer with best weights, fitted TargetNormalizer)
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
    model = PatchBoreholeEndToEndMapBeliefTransformer(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params  : {n_params:,}")
        print(f"  patch_size    : {cfg.bh_patch_size}")
        import math
        n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        print(f"  n_patches     : {n_patches}")
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
        val_metrics = _validate_e2e_map(model, val_loader, device, normalizer)

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
                checkpoint_dir / "patch_borehole_best.pt",
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
        checkpoint_dir / "patch_borehole_last.pt",
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
        checkpoint_dir / "patch_borehole_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    # ---- drill-bin metrics (best model) --------------------------------------
    bin_metrics = _validate_e2e_map_by_drill_bins(model, val_loader, device, normalizer)
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
    step_metrics = _validate_e2e_map_by_step(model, val_loader, device, normalizer)
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
    no_ore_metrics = _validate_no_ore_e2e_map(
        model,
        val_loader,
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    # ---- optional validation plots -------------------------------------------
    if plot_dir is not None:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        if cfg.use_sequential_dataset:
            _save_e2e_map_sequential_val_plots(
                model, val_ds, normalizer, plot_dir, device, n_sequences=cfg.n_val_plots
            )
        else:
            _save_e2e_map_val_plots(
                model, val_ds, normalizer, plot_dir, device, n_plots=cfg.n_val_plots
            )

    if verbose:
        print(
            f"\nTraining complete.  Best val MSE: {best_val_mse:.4f}"
            f"  (epoch {best_epoch}/{cfg.n_epochs})"
        )
        print(f"  checkpoints -> {checkpoint_dir}")

    return model, normalizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_patch_borehole_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[PatchBoreholeEndToEndMapBeliefTransformer, PatchBoreholeE2ETrainingConfig, TargetNormalizer, list[dict]]:
    """Load a PatchBoreholeEndToEndMapBeliefTransformer checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(ckpt: dict) -> PatchBoreholeEndToEndMapBeliefTransformer:
        if "model_cfg" in ckpt:
            return PatchBoreholeEndToEndMapBeliefTransformer(ckpt["model_cfg"])
        return PatchBoreholeEndToEndMapBeliefTransformer(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(
        path, _model_fn, PatchBoreholeE2ETrainingConfig, device
    )
