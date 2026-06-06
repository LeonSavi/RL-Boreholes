"""Training pipeline for CatVarEncoder.

Parallel companion to train_variable_aware_patch_borehole_uncertainty_transformer.py.
Extends that pipeline by loading per-depth rock-type labels from labels_NNNNN.npz
files (produced by generate_training_maps.py) and passing them into the model
alongside the continuous borehole variables.

Differences from the uncertainty variant
-----------------------------------------
* Dataset class is CatVarE2EMapDataset, which stores rock_ids per sample in addition
  to the standard continuous borehole fields.
* Collate function is collate_cat_var_e2e_map, which pads rock_ids to max_K.
* Model is CatVarEncoder, which accepts (boreholes, rock_ids, ...).
* Training function accepts a labels_dir argument pointing to the directory that
  contains labels_vocab.pkl and labels_NNNNN.npz files.
  When labels_dir is None the dataset falls back to zero-label tensors (all 'other').
  n_rock_types is inferred automatically from labels_vocab.pkl.

Label loading
-------------
labels_NNNNN.npz stores:
  rocks : (n_x*n_y, D)  int8  — rock-type vocab indices

Flat borehole-location indexing: flat_idx = i * n_y + j,
matching the borehole_arrays layout in NpzMap.

The vocab is in labels_vocab.pkl: {"rocks": {name: int, ...}}.
n_rock_types is read from len(vocab["rocks"]) automatically.

Training objective
------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + cfg.uncertainty_weight * uncertainty_loss

Checkpoints
-----------
cat_var_best.pt  — lowest validation MSE (ore-value space)
cat_var_last.pt  — final epoch
"""

from __future__ import annotations

import datetime
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ....map_cache import NpzMap
from ....models.belief_models.borehole_encoders.autoencoder import standardise
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
from ....models.belief_models.end_to_end.cat_var_encoder import CatVarEncoder
from .train_end_to_end_map_belief import E2EMapDataset
from .helpers import (
    validate_e2e_map_by_drill_bins,
    validate_e2e_map_by_step,
    validate_no_ore_e2e_map,
)
from ..training_configs import CatVarConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CatVarE2EMapDataset(Dataset):
    """Pre-generated (partial observation → full map) pairs with rock-type labels.

    Extends E2EMapDataset by storing per-borehole rock_ids alongside the continuous
    borehole fields.  Each sample dict gains:

        rock_ids  (K, D)  np.ndarray int64 — rock-type vocab indices

    When no labels_dir is provided (or the npz file is absent), rock_ids defaults to
    zeros so the model sees only the 'other'/unknown category.
    """

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def raw_targets(self) -> np.ndarray:
        return np.concatenate([s["target_map"].ravel() for s in self.samples])

    def apply_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        for s in self.samples:
            s["target_map"] = normalizer.transform(
                s["target_map"].ravel()
            ).reshape(s["target_map"].shape).astype(np.float32)

    @classmethod
    def from_cache(
        cls,
        cache: NpzMap,
        resources: DecisionSimulationResources,
        cfg: CatVarConfig,
        labels_dir: Path | None = None,
        verbose: bool = True,
        is_val: bool = False,
    ) -> "CatVarE2EMapDataset":
        """Build dataset from a pre-loaded NpzMap, optionally loading label files.

        Parameters
        ----------
        cache      : pre-loaded map pool (train or val subset)
        resources  : shared experiment resources (norm stats, variable names)
        cfg        : CatVarConfig training configuration
        labels_dir : directory containing labels_vocab.pkl and labels_NNNNN.npz.
                     If None or the file does not exist for a given map index,
                     rock_ids defaults to zeros (all 'other').
        verbose    : print progress every 10 maps
        is_val     : use val-specific sample counts and a shifted RNG seed
        """
        spm = cfg.val_samples_per_map if is_val else cfg.samples_per_map
        seed = cfg.seed + (1 if is_val else 0)
        rng = np.random.default_rng(seed)

        n_x, n_y = cache.n_x, cache.n_y
        n_depth = cache.borehole_arrays[0].shape[-1] if cache.pool_size > 0 else cfg.n_depth
        variables = resources.variable_names

        xs = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys, indexing="ij")
        grid_pos = np.stack([xg, yg], axis=-1)  # (n_x, n_y, 2)

        all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]
        all_idx = np.arange(len(all_locations))

        samples: list[dict] = []

        for map_idx in range(cache.pool_size):
            bh_arr = cache.borehole_arrays[map_idx].copy()
            if resources.norm_stats:
                bh_arr = standardise(bh_arr, resources.norm_stats, variables)
            bh_arr = np.nan_to_num(bh_arr, nan=0.0).astype(np.float32)

            target_ore = cache.targets[map_idx]  # (n_x, n_y)

            # Load rock labels — prefer rocks already in the NpzMap cache (loaded
            # from the HDF5 rocks dataset), fall back to separate labels_NNNNN.npz.
            rock_labels: np.ndarray | None = None
            if cache.rocks_arrays is not None and map_idx < len(cache.rocks_arrays):
                rock_labels = cache.rocks_arrays[map_idx].astype(np.int64)  # (n_x*n_y, D)
            elif labels_dir is not None:
                npz_path = Path(labels_dir) / f"labels_{map_idx:05d}.npz"
                if npz_path.exists():
                    lbl = np.load(npz_path)
                    rock_labels = lbl["rocks"].astype(np.int64)    # (n_x*n_y, D)
                elif verbose and map_idx == 0:
                    print(
                        f"  [CatVarE2EMapDataset] labels file not found: {npz_path}"
                        "  — rock_ids will be zero (all 'other')."
                    )

            def _append(chosen_idx: np.ndarray, sequence_id: int = 0) -> None:
                drill_locs = [all_locations[k] for k in chosen_idx]
                K = len(drill_locs)
                flat_ids = [i * n_y + j for i, j in drill_locs]

                drill_bhs = np.stack([bh_arr[fid] for fid in flat_ids])  # (K, V, D)
                ore_vals_k = np.array(
                    [target_ore[i, j] for i, j in drill_locs], dtype=np.float32
                )
                positions_k = np.stack([grid_pos[i, j] for i, j in drill_locs])

                rock_ids_k = (
                    np.stack([rock_labels[fid] for fid in flat_ids])  # (K, D)
                    if rock_labels is not None
                    else np.zeros((K, n_depth), dtype=np.int64)
                )

                samples.append(
                    {
                        "boreholes":  drill_bhs,
                        "rock_ids":   rock_ids_k,
                        "ore_vals":   ore_vals_k,
                        "positions":  positions_k,
                        "target_map": target_ore.copy(),
                        "drill_count": K,
                        "map_idx":    map_idx,
                        "sequence_id": sequence_id,
                    }
                )

            if cfg.use_sequential_dataset:
                for seq_id in range(cfg.n_sequences_per_map):
                    sequence = rng.permutation(all_idx)
                    for step in cfg.prefix_steps:
                        if step >= len(all_locations):
                            continue
                        _append(sequence[:step], sequence_id=seq_id)
            else:
                for _ in range(spm):
                    n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                    chosen_idx = rng.choice(all_idx, size=n_drills, replace=False)
                    _append(chosen_idx)

            if verbose and (map_idx + 1) % 10 == 0:
                print(
                    f"  [CatVarE2EMapDataset] {map_idx + 1}/{cache.pool_size} maps processed"
                )

        return cls(samples)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_cat_var_e2e_map(batch: list[dict]) -> dict:
    """Pad variable-length K sequences to max_K, including rock_ids."""
    max_K = max(s["boreholes"].shape[0] for s in batch)
    B = len(batch)
    V, D = batch[0]["boreholes"].shape[1], batch[0]["boreholes"].shape[2]
    n_x, n_y = batch[0]["target_map"].shape

    boreholes_pad  = np.zeros((B, max_K, V, D),   dtype=np.float32)
    rock_pad       = np.zeros((B, max_K, D),       dtype=np.int64)
    ore_pad        = np.zeros((B, max_K),          dtype=np.float32)
    pos_pad        = np.zeros((B, max_K, 2),       dtype=np.float32)
    padding_mask   = np.ones((B, max_K),           dtype=bool)   # True = padded
    target_maps    = np.zeros((B, 1, n_x, n_y),   dtype=np.float32)
    drill_counts   = np.zeros(B,                   dtype=np.int64)

    for i, s in enumerate(batch):
        K = s["boreholes"].shape[0]
        boreholes_pad[i, :K] = s["boreholes"]
        rock_pad[i, :K]      = s["rock_ids"]
        ore_pad[i, :K]       = s["ore_vals"]
        pos_pad[i, :K]       = s["positions"]
        padding_mask[i, :K]  = False
        target_maps[i, 0]    = s["target_map"]
        drill_counts[i]      = s.get("drill_count", K)

    return {
        "boreholes":    torch.from_numpy(boreholes_pad),
        "rock_ids":     torch.from_numpy(rock_pad),
        "ore_vals":     torch.from_numpy(ore_pad),
        "positions":    torch.from_numpy(pos_pad),
        "padding_mask": torch.from_numpy(padding_mask),
        "target_map":   torch.from_numpy(target_maps),
        "drill_counts": torch.from_numpy(drill_counts),
    }


# ---------------------------------------------------------------------------
# Ore-only wrapper (reuse existing ore-only validation helpers)
# ---------------------------------------------------------------------------


class _OreWrapper(nn.Module):
    """Wraps CatVarEncoder so that forward() returns only the ore prediction.

    Allows reusing validation helpers from helpers/ that expect a model whose
    forward() returns a single (B, 1, n_x, n_y) tensor.  The helpers receive
    only the standard borehole batch dict, so rock_ids must be captured
    from outer scope via the closure.
    """

    def __init__(self, model: CatVarEncoder) -> None:
        super().__init__()
        self._model = model
        self._rock_ids: torch.Tensor | None = None

    def forward(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rock_ids = self._rock_ids
        if rock_ids is None:
            B, K, D = boreholes.shape[0], boreholes.shape[1], boreholes.shape[3]
            rock_ids = torch.zeros(B, K, D, dtype=torch.long, device=boreholes.device)
        pred_ore, _ = self._model(boreholes, rock_ids, ore_vals, positions, padding_mask)
        return pred_ore


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_cat_var(
    model: CatVarEncoder,
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

            pred_ore_norm, pred_unc_norm = model(bh, rid, ov, pos, pm)

            pred = normalizer.inverse_tensor(pred_ore_norm)
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
            mean_err = ae.mean(dim=1).clamp(min=1e-8)
            unc_top10_total += (top10_err / mean_err).mean().item()

            n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_mse":            mse_total  / n,
        "val_mae":            mae_total  / n,
        "val_corr":           corr_total / n,
        "val_unc_mse":        unc_mse_total  / n,
        "val_unc_mae":        unc_mae_total  / n,
        "val_unc_corr":       unc_corr_total / n,
        "val_unc_top10_ratio": unc_top10_total / n,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_cat_var_encoder(
    resources: DecisionSimulationResources,
    cfg: CatVarConfig,
    device: str,
    checkpoint_dir: Path,
    labels_dir: Path | None = None,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: CatVarE2EMapDataset | None = None,
    val_ds: CatVarE2EMapDataset | None = None,
    train_cache: NpzMap | None = None,
    val_cache: NpzMap | None = None,
) -> tuple[CatVarEncoder, TargetNormalizer, Path]:
    """Train CatVarEncoder.

    Creates a timestamped run directory inside checkpoint_dir and saves all
    outputs there:
      <checkpoint_dir>/<timestamp>/cat_var_best.pt
      <checkpoint_dir>/<timestamp>/cat_var_last.pt
      <checkpoint_dir>/<timestamp>/training_history.csv
      <checkpoint_dir>/<timestamp>/val_metrics_*.json

    Parameters
    ----------
    resources      : shared resources (norm_stats for borehole standardisation)
    cfg            : training hyperparameters
    device         : torch device string
    checkpoint_dir : parent directory; a timestamped sub-directory is created here
    labels_dir     : directory with labels_vocab.pkl and labels_NNNNN.npz files.
                     Pass None to run without categorical labels (all-zero → 'other').
    plot_dir       : if given, validation plots are saved to the run directory
    verbose        : print per-epoch metrics
    train_ds / val_ds : pre-built CatVarE2EMapDataset instances (preferred).
                     If None, train_cache / val_cache must be provided instead.
    train_cache / val_cache : NpzMap caches used to build datasets when train_ds
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

    # ---- read vocab to set n_rock_types --------------------------------------
    if labels_dir is not None:
        vocab_path = Path(labels_dir) / "labels_vocab.pkl"
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                vocabs = pickle.load(f)
            cfg.n_rock_types = len(vocabs.get("rocks", {})) or cfg.n_rock_types
            if verbose:
                print(f"  label vocab   : {cfg.n_rock_types} rock types  ({vocab_path})")

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
    model = CatVarEncoder(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        n_bh_tokens = cfg.n_variables * n_patches
        print(f"  model params  : {n_params:,}")
        print(f"  n_variables   : {cfg.n_variables}")
        print(f"  n_rock_types  : {cfg.n_rock_types}")
        print(f"  patch_size    : {cfg.bh_patch_size}")
        print(f"  n_patches     : {n_patches}")
        print(f"  bh_tokens/bh  : {n_bh_tokens} + 1 CLS = {n_bh_tokens + 1}")
        print(f"  uncertainty   : True (weight={cfg.uncertainty_weight})")
        print(f"  grid          : {cfg.n_x} × {cfg.n_y}")

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

            pred_ore, pred_uncertainty = model(bh, rid, ov, pos, pm)

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
        val_metrics = _validate_cat_var(model, val_loader, device, normalizer)

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
                run_dir / "cat_var_best.pt",
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
        run_dir / "cat_var_last.pt",
        model, cfg, epoch, history, normalizer, model_cfg=model_cfg,
    )
    export_history(history, run_dir)

    # ---- reload best weights -------------------------------------------------
    best_ckpt = torch.load(
        run_dir / "cat_var_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\n  best val epoch: {best_epoch}/{cfg.n_epochs}")

    # ---- ore-only adapter for existing drill-bin / step / no-ore helpers -----
    ore_wrapper = _OreWrapper(model)

    # Build a val_loader that also sets rock IDs on the wrapper
    def _ore_only_val_loader():
        for batch in DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_cat_var_e2e_map,
        ):
            ore_wrapper._rock_ids = batch["rock_ids"].to(device)
            yield {k: v for k, v in batch.items() if k != "rock_ids"}

    # ---- drill-bin metrics ---------------------------------------------------
    bin_metrics = validate_e2e_map_by_drill_bins(
        ore_wrapper, _ore_only_val_loader(), device, normalizer
    )
    if bin_metrics:
        if verbose:
            print("\nValidation metrics by drill count (best model, ore-value space):")
            for lo, hi in DRILL_BINS:
                key = f"{lo}_{hi}"
                n_b = bin_metrics.get(f"n_{key}", 0)
                mse = bin_metrics.get(f"mse_{key}", float("nan"))
                corr = bin_metrics.get(f"corr_{key}", float("nan"))
                print(f"  drills {lo:2d}-{hi:2d}  n={n_b:5d}  mse={mse:.4f}  corr={corr:.4f}")
        with open(run_dir / "val_metrics_by_drills.json", "w") as f:
            json.dump(bin_metrics, f, indent=2)

    # ---- per-step metrics ----------------------------------------------------
    step_metrics = validate_e2e_map_by_step(
        ore_wrapper, _ore_only_val_loader(), device, normalizer
    )
    if step_metrics:
        with open(run_dir / "val_metrics_by_step.json", "w") as f:
            json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)

    # ---- no-ore false-positive metrics ---------------------------------------
    no_ore_metrics = validate_no_ore_e2e_map(
        ore_wrapper,
        _ore_only_val_loader(),
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, run_dir, verbose=verbose)

    # ---- uncertainty metrics (final) -----------------------------------------
    final_val = _validate_cat_var(model, val_loader, device, normalizer)
    with open(run_dir / "val_metrics_uncertainty.json", "w") as f:
        json.dump(final_val, f, indent=2)
    if verbose:
        print(
            f"\nUncertainty calibration (best model, normalised space):"
            f"\n  unc_mse      : {final_val['val_unc_mse']:.4f}"
            f"\n  unc_corr     : {final_val['val_unc_corr']:.4f}"
            f"\n  top10_ratio  : {final_val['val_unc_top10_ratio']:.4f}"
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


def load_cat_var_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[CatVarEncoder, CatVarConfig, TargetNormalizer, list[dict]]:
    """Load a CatVarEncoder checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(ckpt: dict) -> CatVarEncoder:
        if "model_cfg" in ckpt:
            return CatVarEncoder(ckpt["model_cfg"])
        return CatVarEncoder(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(path, _model_fn, CatVarConfig, device)
