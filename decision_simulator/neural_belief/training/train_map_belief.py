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

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulator.map_generator import SimConfig
from decision_simulator.resources import DecisionSimulationResources
from ..dataset import BeliefDatasetConfig, GeologicalBeliefDataset
from ..utils import TargetNormalizer
from ..baselines import evaluate_baselines
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    save_val_plots,
)
from ..models.map_belief_transformer import MapBeliefConfig, MapBeliefTransformer


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class MapBeliefTrainingConfig:
    """Hyperparameters for training the MapBeliefTransformer.

    Dataset and normalisation fields are identical to NeuralBeliefTrainingConfig
    so the two training functions can be called with the same preparation code.
    Architectural fields replace the UNet-specific ``in_channels`` / ``base_channels``.
    """

    # --- Dataset (same as NeuralBeliefTrainingConfig) ---
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # --- Input / grid ---
    latent_dim: int = 128           # borehole encoder latent dim (sets in_channels)
    n_x: int = 32
    n_y: int = 32

    # --- Model architecture ---
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 1024                # feedforward dim (4 × d_model)
    dropout: float = 0.1
    head_hidden_dim: int = 128

    # --- Target normalisation (same as NeuralBeliefTrainingConfig) ---
    norm_mode: str = "log1p"        # "log1p" | "zscore" | "none"

    # --- Optimisation ---
    batch_size: int = 16            # smaller than UNet due to transformer attention memory
    lr: float = 1e-4               # lower than UNet; transformers train more stably at low lr
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- Gradient clipping (important for transformer stability) ---
    grad_clip_norm: float = 1.0     # 0.0 = disabled

    # --- Misc ---
    seed: int = 42
    borehole_encoder: str = "unknown"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    @property
    def in_channels(self) -> int:
        """Total input channels: ore + mask + latent."""
        return 2 + self.latent_dim

    def to_model_config(self) -> MapBeliefConfig:
        """Construct a MapBeliefConfig from the architectural fields of this dataclass."""
        return MapBeliefConfig(
            latent_dim=self.latent_dim,
            n_x=self.n_x,
            n_y=self.n_y,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_hidden_dim=self.head_hidden_dim,
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_map_belief(
    resources: DecisionSimulationResources,
    cfg: MapBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    sim_cfg: SimConfig | None = None,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: GeologicalBeliefDataset | None = None,
    val_ds: GeologicalBeliefDataset | None = None,
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
    sim_cfg        : map dimensions / ore parameters (defaults to SimConfig())
    plot_dir       : if given, save 4 validation plots here after training
    verbose        : print per-epoch metrics
    train_ds / val_ds : optional pre-built datasets (skip generation if provided)

    Returns
    -------
    (trained MapBeliefTransformer with best weights, fitted TargetNormalizer)
    """
    if sim_cfg is None:
        sim_cfg = SimConfig()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- datasets ----------------------------------------------------------------
    if train_ds is None or val_ds is None:
        if verbose:
            print("Generating training dataset ...")
        train_ds = GeologicalBeliefDataset.generate(
            resources,
            BeliefDatasetConfig(
                n_maps=cfg.n_train_maps,
                samples_per_map=cfg.samples_per_map,
                min_drills=cfg.min_drills,
                max_drills=cfg.max_drills,
                seed=cfg.seed,
            ),
            device=device,
            sim_cfg=sim_cfg,
            verbose=verbose,
        )

        if verbose:
            print("Generating validation dataset ...")
        val_ds = GeologicalBeliefDataset.generate(
            resources,
            BeliefDatasetConfig(
                n_maps=cfg.n_val_maps,
                samples_per_map=cfg.val_samples_per_map,
                min_drills=cfg.min_drills,
                max_drills=cfg.max_drills,
                seed=cfg.seed + 1,
            ),
            device=device,
            sim_cfg=sim_cfg,
            verbose=verbose,
        )
    else:
        if verbose:
            print("Using pre-built datasets.")
        # Wrap in new objects so attribute reassignments (normalisation) do not
        # mutate the caller's datasets.
        train_ds = GeologicalBeliefDataset(train_ds.inputs, train_ds.targets, train_ds.drill_counts)
        val_ds   = GeologicalBeliefDataset(val_ds.inputs,   val_ds.targets,   val_ds.drill_counts)

    # ---- target normalisation ---------------------------------------------------
    normalizer = TargetNormalizer(mode=cfg.norm_mode)
    normalizer.fit(train_ds.targets.numpy())
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

    # ---- data loaders -----------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ------------------------------------------------------------------
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

    # ---- checkpoint helper ------------------------------------------------------
    history: list[dict] = []

    def _save_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {
                "state_dict":        model.state_dict(),
                "cfg":               cfg,            # MapBeliefTrainingConfig
                "model_cfg":         model_cfg,      # MapBeliefConfig (for loading without training cfg)
                "epoch":             epoch,
                "history":           history,
                "normalizer":        normalizer,
                "sim_cfg":           sim_cfg,
                "n_x":               sim_cfg.n_x,
                "n_y":               sim_cfg.n_y,
                "latent_dim":        cfg.latent_dim,
            },
            path,
        )

    # ---- training loop ----------------------------------------------------------
    best_val_mse = float("inf")

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)  # MSE in normalised space

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

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            _save_ckpt(checkpoint_dir / "map_belief_best.pt", epoch)

    _save_ckpt(checkpoint_dir / "map_belief_last.pt", cfg.n_epochs)

    # ---- export training history ------------------------------------------------
    _export_history(history, checkpoint_dir)

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
        save_val_plots(model, val_ds, normalizer, plot_dir, device)

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
                n    = bin_metrics.get(f"n_{key}", 0)
                mse  = bin_metrics.get(f"mse_{key}", float("nan"))
                mae  = bin_metrics.get(f"mae_{key}", float("nan"))
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
    """Load a saved MapBeliefTransformer checkpoint.

    Returns
    -------
    (model, training_config, normalizer, training_history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Prefer the dedicated MapBeliefConfig stored in the checkpoint; fall back
    # to reconstructing it from the training config for older checkpoints.
    if "model_cfg" in ckpt:
        model_cfg: MapBeliefConfig = ckpt["model_cfg"]
    else:
        train_cfg: MapBeliefTrainingConfig = ckpt["cfg"]
        model_cfg = train_cfg.to_model_config()

    model = MapBeliefTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normalizer: TargetNormalizer = ckpt.get("normalizer", TargetNormalizer(mode="none"))
    train_cfg = ckpt.get("cfg", MapBeliefTrainingConfig())

    return model, train_cfg, normalizer, ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_map_belief_training_config(
    debug: bool = False, overrides: dict | None = None
) -> MapBeliefTrainingConfig:
    """Construct a MapBeliefTrainingConfig with optional field overrides.

    Parameters
    ----------
    debug     : if True, use minimal dataset / model for fast smoke testing
    overrides : dict of field names → values to set after defaults

    Raises
    ------
    ValueError if any override key is not a valid MapBeliefTrainingConfig field.
    """
    overrides = overrides or {}
    valid_fields = {f.name for f in dataclasses.fields(MapBeliefTrainingConfig)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise ValueError(
            f"Unknown MapBeliefTrainingConfig field(s): {sorted(invalid)}.\n"
            f"Valid fields: {sorted(valid_fields)}"
        )

    if debug:
        cfg = MapBeliefTrainingConfig(
            n_train_maps=2,
            samples_per_map=2,
            n_val_maps=1,
            val_samples_per_map=2,
            n_epochs=2,
            batch_size=2,
            d_model=64,
            n_heads=4,
            n_encoder_layers=1,
            d_ff=128,
            head_hidden_dim=32,
        )
    else:
        cfg = MapBeliefTrainingConfig()

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _export_history(history: list[dict], directory: Path) -> None:
    """Write training history to JSON and CSV files."""
    if not history:
        return
    json_path = directory / "training_history.json"
    csv_path  = directory / "training_history.csv"
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
