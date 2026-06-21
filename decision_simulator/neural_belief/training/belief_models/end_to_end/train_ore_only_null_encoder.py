"""Training pipeline for OreOnlyNullEncoder.

Null-test companion to train_cat_var_encoder.py.

This script trains OreOnlyNullEncoder, which uses only observed ore values and
drill positions — no continuous borehole logs, no rock labels, no formation
labels.  It is structurally identical to CatVarEncoder's map-level pipeline but
replaces all borehole latents with zeros.

If CatVarEncoder does not outperform this model, the borehole encoder is probably
not adding useful predictive information beyond the observed ore signal.

Differences from train_cat_var_encoder.py
------------------------------------------
* Model is OreOnlyNullEncoder, not CatVarEncoder.
* Training config is OreOnlyNullConfig.
* The model forward call passes zero formation_ids (shape B, K, D) alongside
  boreholes and rock_ids for API compatibility; all three are ignored by the model.
* Checkpoints are saved as ore_only_null_best.pt / ore_only_null_last.pt.
* The _OreWrapper does not need to inject rock_ids from outer scope because the
  null encoder ignores all borehole inputs internally.

Reuse
-----
* Dataset class and collate function are imported from train_cat_var_encoder.py:
    CatVarE2EMapDataset, collate_cat_var_e2e_map
* Validation helpers are imported from helpers/:
    validate_e2e_map_by_drill_bins, validate_e2e_map_by_step, validate_no_ore_e2e_map

Training objective
------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + uncertainty_weight * uncertainty_loss

Checkpoints
-----------
ore_only_null_best.pt  — lowest validation MSE (ore-value space)
ore_only_null_last.pt  — final epoch
"""

from __future__ import annotations

import datetime
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ....map_hdf5 import MapPool
from ....training_utils import TargetNormalizer
from ....training_utils import (
    DRILL_BINS,
    export_history,
    false_positive_loss,
    load_model_encoder_checkpoint,
    pearson_correlation,
    save_checkpoint_model,
    save_no_ore_metrics,
)
from ....models.belief_models.end_to_end.ore_only_null_encoder import OreOnlyNullEncoder
from .train_cat_var_encoder import CatVarE2EMapDataset, collate_cat_var_e2e_map
from .helpers import (
    validate_e2e_map_by_drill_bins,
    validate_e2e_map_by_step,
    validate_no_ore_e2e_map,
    save_e2e_map_val_plots,
    save_sequential_val_plots,
)
from ..training_configs import OreOnlyNullConfig


# ---------------------------------------------------------------------------
# Ore-only wrapper (reuse existing drill-bin / step / no-ore helpers)
# ---------------------------------------------------------------------------


class _OreWrapperNull(nn.Module):
    """Wraps OreOnlyNullEncoder so that forward() returns only the ore prediction.

    Allows reusing validation helpers from helpers/ that expect a model whose
    forward() returns a single (B, 1, n_x, n_y) tensor and accepts the standard
    borehole batch signature (boreholes, ore_vals, positions, padding_mask).

    Zero tensors are created on-the-fly for rock_ids and formation_ids because
    OreOnlyNullEncoder ignores them.
    """

    def __init__(self, model: OreOnlyNullEncoder) -> None:
        super().__init__()
        self._model = model

    def forward(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, _, D = boreholes.shape
        rock_ids      = torch.zeros(B, K, D, dtype=torch.long, device=boreholes.device)
        formation_ids = torch.zeros(B, K, D, dtype=torch.long, device=boreholes.device)
        pred_ore, _ = self._model(
            boreholes, rock_ids, formation_ids, ore_vals, positions, padding_mask
        )
        return pred_ore


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_null_encoder(
    model: OreOnlyNullEncoder,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Ore reconstruction + uncertainty calibration metrics."""
    model.eval()
    mse_total = mae_total = corr_total = 0.0
    unc_mse_total = unc_mae_total = unc_corr_total = unc_top10_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            bh  = batch["boreholes"].to(device)
            rid = batch["rock_ids"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm  = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            B, K, _, D = bh.shape
            formation_ids = torch.zeros(B, K, D, dtype=torch.long, device=device)

            pred_ore_norm, pred_unc_norm = model(bh, rid, formation_ids, ov, pos, pm)

            pred    = normalizer.inverse_tensor(pred_ore_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)
            mse_total  += F.mse_loss(pred, tgt_raw).item()
            mae_total  += (pred - tgt_raw).abs().mean().item()
            corr_total += pearson_correlation(pred, tgt_raw)

            abs_err_norm = (pred_ore_norm - tgt).abs()
            unc_mse_total  += F.mse_loss(pred_unc_norm, abs_err_norm).item()
            unc_mae_total  += (pred_unc_norm - abs_err_norm).abs().mean().item()
            unc_corr_total += pearson_correlation(pred_unc_norm, abs_err_norm)

            B_b = pred_unc_norm.shape[0]
            pu = pred_unc_norm.view(B_b, -1)
            ae = abs_err_norm.view(B_b, -1)
            k = max(1, int(pu.shape[1] * 0.10))
            top_idx = pu.topk(k, dim=1).indices
            top10_err = ae.gather(1, top_idx).mean(dim=1)
            mean_err  = ae.mean(dim=1).clamp(min=1e-8)
            unc_top10_total += (top10_err / mean_err).mean().item()

            n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_mse":             mse_total  / n,
        "val_mae":             mae_total  / n,
        "val_corr":            corr_total / n,
        "val_unc_mse":         unc_mse_total  / n,
        "val_unc_mae":         unc_mae_total  / n,
        "val_unc_corr":        unc_corr_total / n,
        "val_unc_top10_ratio": unc_top10_total / n,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_ore_only_null_encoder(
    resources: DecisionSimulationResources,
    cfg: OreOnlyNullConfig,
    device: str,
    checkpoint_dir: Path,
    labels_dir: Path | None = None,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: CatVarE2EMapDataset | None = None,
    val_ds: CatVarE2EMapDataset | None = None,
    train_cache: MapPool | None = None,
    val_cache: MapPool | None = None,
) -> tuple[OreOnlyNullEncoder, TargetNormalizer, Path]:
    """Train OreOnlyNullEncoder.

    Creates a timestamped run directory inside checkpoint_dir and saves all
    outputs there:
      <checkpoint_dir>/<timestamp>/ore_only_null_best.pt
      <checkpoint_dir>/<timestamp>/ore_only_null_last.pt
      <checkpoint_dir>/<timestamp>/training_history.csv
      <checkpoint_dir>/<timestamp>/val_metrics_*.json

    Parameters
    ----------
    resources      : shared resources (norm_stats for borehole standardisation)
    cfg            : OreOnlyNullConfig training hyperparameters
    device         : torch device string
    checkpoint_dir : parent directory; a timestamped sub-directory is created here
    labels_dir     : directory with labels_vocab.pkl and labels_NNNNN.npz files
                     (used for dataset building only; n_rock_types is read here
                     to keep config parity with CatVarEncoder runs).
                     Pass None to run without categorical labels.
    plot_dir       : if given, validation plots are saved to the run directory
    verbose        : print per-epoch metrics
    train_ds / val_ds : pre-built CatVarE2EMapDataset instances (preferred).
                     If None, train_cache / val_cache must be provided instead.
    train_cache / val_cache : MapPool caches used to build datasets when train_ds
                     / val_ds are not supplied.

    Returns
    -------
    (trained model with best weights, fitted TargetNormalizer, run_dir)
    """
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(checkpoint_dir) / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- resolve borehole dimensions from resources --------------------------
    if resources.variable_names:
        cfg.n_variables = len(resources.variable_names)

    # ---- build datasets if not pre-supplied ----------------------------------
    if train_ds is None:
        if train_cache is None:
            raise ValueError("Provide either train_ds or train_cache.")
        train_ds = CatVarE2EMapDataset.from_cache(
            train_cache, resources, cfg, labels_dir=labels_dir, verbose=verbose, is_val=False
        )
    if val_ds is None:
        if val_cache is None:
            raise ValueError("Provide either val_ds or val_cache.")
        val_ds = CatVarE2EMapDataset.from_cache(
            val_cache, resources, cfg, labels_dir=labels_dir, verbose=verbose, is_val=True
        )

    if len(train_ds) > 0:
        cfg.n_depth = train_ds.samples[0]["boreholes"].shape[2]
        sample_map = train_ds.samples[0]["target_map"]
        cfg.n_x, cfg.n_y = int(sample_map.shape[0]), int(sample_map.shape[1])

    # ---- read vocab to set n_rock_types (config parity only) -----------------
    if labels_dir is not None:
        vocab_path = Path(labels_dir) / "labels_vocab.pkl"
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                vocabs = pickle.load(f)
            cfg.n_rock_types = len(vocabs.get("rocks", {})) or cfg.n_rock_types
            if verbose:
                print(
                    f"  label vocab   : {cfg.n_rock_types} rock types  ({vocab_path})"
                    "  [config parity only — null encoder ignores labels]"
                )

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
        collate_fn=collate_cat_var_e2e_map,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_cat_var_e2e_map,
    )

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ---------------------------------------------------------------
    model_cfg = cfg.to_model_config()
    model = OreOnlyNullEncoder(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model         : OreOnlyNullEncoder (null-test — no borehole encoder)")
        print(f"  model params  : {n_params:,}")
        print(f"  grid          : {cfg.n_x} × {cfg.n_y}")
        print(f"  latent_dim    : {cfg.latent_dim}  (zero-filled)")
        print(f"  uncertainty   : True (weight={cfg.uncertainty_weight})")

    # ---- training loop -------------------------------------------------------
    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        total_loss_sum = ore_loss_sum = unc_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            bh  = batch["boreholes"].to(device)
            rid = batch["rock_ids"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm  = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            B, K, _, D = bh.shape
            formation_ids = torch.zeros(B, K, D, dtype=torch.long, device=device)

            pred_ore, pred_uncertainty = model(bh, rid, formation_ids, ov, pos, pm)

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
            ore_loss_sum   += ore_loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        val_metrics = _validate_null_encoder(model, val_loader, device, normalizer)

        row = {
            "epoch":           epoch,
            "train_loss":      total_loss_sum / n,
            "train_ore_loss":  ore_loss_sum   / n,
            "train_unc_loss":  unc_loss_sum   / n if cfg.use_uncertainty_head else float("nan"),
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
                f"  unc_corr={val_metrics['val_unc_corr']:.4f}"
            )

        if val_metrics["val_mse"] < best_val_mse - cfg.min_delta:
            best_val_mse = val_metrics["val_mse"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint_model(
                run_dir / "ore_only_null_best.pt",
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

    save_checkpoint_model(
        run_dir / "ore_only_null_last.pt",
        model, cfg, epoch, history, normalizer, model_cfg=model_cfg,
    )
    export_history(history, run_dir)

    # ---- reload best weights -------------------------------------------------
    best_ckpt = torch.load(
        run_dir / "ore_only_null_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\n  best val epoch: {best_epoch}/{cfg.n_epochs}")

    # ---- ore-only adapter for existing drill-bin / step / no-ore helpers -----
    ore_wrapper = _OreWrapperNull(model)

    def _null_val_loader():
        for batch in DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_cat_var_e2e_map,
        ):
            yield {k: v for k, v in batch.items() if k != "rock_ids"}

    # ---- drill-bin metrics ---------------------------------------------------
    bin_metrics = validate_e2e_map_by_drill_bins(
        ore_wrapper, _null_val_loader(), device, normalizer
    )
    if bin_metrics:
        if verbose:
            print("\nValidation metrics by drill count (best model, ore-value space):")
            for lo, hi in DRILL_BINS:
                key = f"{lo}_{hi}"
                n_b  = bin_metrics.get(f"n_{key}", 0)
                mse  = bin_metrics.get(f"mse_{key}", float("nan"))
                corr = bin_metrics.get(f"corr_{key}", float("nan"))
                print(f"  drills {lo:2d}-{hi:2d}  n={n_b:5d}  mse={mse:.4f}  corr={corr:.4f}")
        with open(run_dir / "val_metrics_by_drills.json", "w") as f:
            json.dump(bin_metrics, f, indent=2)

    # ---- per-step metrics ----------------------------------------------------
    step_metrics = validate_e2e_map_by_step(
        ore_wrapper, _null_val_loader(), device, normalizer
    )
    if step_metrics:
        with open(run_dir / "val_metrics_by_step.json", "w") as f:
            json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)

    # ---- no-ore false-positive metrics ---------------------------------------
    no_ore_metrics = validate_no_ore_e2e_map(
        ore_wrapper,
        _null_val_loader(),
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, run_dir, verbose=verbose)

    # ---- uncertainty metrics (final) -----------------------------------------
    final_val = _validate_null_encoder(model, val_loader, device, normalizer)
    with open(run_dir / "val_metrics_uncertainty.json", "w") as f:
        json.dump(final_val, f, indent=2)
    if verbose:
        print(
            f"\nUncertainty calibration (best model, normalised space):"
            f"\n  unc_mse      : {final_val['val_unc_mse']:.4f}"
            f"\n  unc_corr     : {final_val['val_unc_corr']:.4f}"
            f"\n  top10_ratio  : {final_val['val_unc_top10_ratio']:.4f}"
        )

    if plot_dir is not None:
        if cfg.use_sequential_dataset:
            save_sequential_val_plots(
                ore_wrapper, val_ds, normalizer, run_dir, device,
                n_sequences=cfg.n_val_plots,
            )
        else:
            save_e2e_map_val_plots(
                ore_wrapper, val_ds, normalizer, run_dir, device,
                n_plots=cfg.n_val_plots,
            )

    if verbose:
        print(
            f"\nTraining complete.  Best val MSE: {best_val_mse:.4f}"
            f"  (epoch {best_epoch}/{cfg.n_epochs})"
        )
        print(f"  run directory -> {run_dir}")

    return model, normalizer, run_dir


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_ore_only_null_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[OreOnlyNullEncoder, OreOnlyNullConfig, TargetNormalizer, list[dict]]:
    """Load an OreOnlyNullEncoder checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(ckpt: dict) -> OreOnlyNullEncoder:
        if "model_cfg" in ckpt:
            return OreOnlyNullEncoder(ckpt["model_cfg"])
        return OreOnlyNullEncoder(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(path, _model_fn, OreOnlyNullConfig, device)
