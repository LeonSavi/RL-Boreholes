"""Training pipeline for EndToEndMapBeliefTransformer.

Trains the borehole encoder and map belief transformer jointly to reconstruct
the full ore map from raw drilled boreholes.  Unlike train_end_to_end.py (which
predicts a scalar ore value at one candidate), this pipeline trains on a full-map
reconstruction loss so that borehole embeddings learn features useful for the
global geological belief state.

Key differences from train_map_belief.py
-----------------------------------------
* Input is raw boreholes (K, V, D) per sample, not pre-computed latent maps.
* Variable-length observation sequences are handled by a custom collate_fn that
  pads borehole tensors and builds an attention key-padding mask.
* The borehole encoder is part of the model graph and receives gradients.
* The normalizer is fitted internally from training-map ore values.

Key differences from train_end_to_end.py
-----------------------------------------
* Target is a full ore map (n_x, n_y) per sample, not a scalar candidate value.
* Loss is MSE over the full reconstructed map.
* No candidate sampling is needed — the full map is the reconstruction target.

Dataset structure
-----------------
Each sample contains:
    boreholes    (K, V, D)         standardised raw boreholes at drilled cells
    ore_vals     (K,)              observed ore at drilled cells
    positions    (K, 2)            normalised [0,1] (x, y) of drilled cells
    target_map   (n_x, n_y)       full ore map target (normalised after fitting)
    drill_count  int               K — used for drill-bin metrics
    map_idx      int               index into the cache (for plotting)

K varies per sample, so collate_e2e_map pads boreholes to max_K in each batch
and creates a boolean padding mask (True = padded).

Checkpoints
-----------
e2e_map_belief_best.pt  — lowest validation MSE (ore-value space)
e2e_map_belief_last.pt  — final epoch
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ..map_cache import NpzMap
from ..models.borehole_encoders.autoencoder import standardise
from ..models.end_to_end.end_to_end_map_belief_transformer import (
    EndToEndMapBeliefTransformer,
)
from ..utils import TargetNormalizer
from ..training_utils import (
    DRILL_BINS,
    export_history,
    false_positive_loss,
    load_model_encoder_checkpoint,
    pearson_correlation,
    save_checkpoint_model,
    save_no_ore_metrics,
)
from .training_configs import E2EMapBeliefTrainingConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class E2EMapDataset(Dataset):
    """Pre-generated (partial observation → full map) pairs for reconstruction training.

    Each sample stores raw standardised boreholes at the drilled cells together
    with the full target ore map.  This enables end-to-end training of the
    borehole encoder and map belief transformer with a full-map reconstruction
    objective.

    Attributes
    ----------
    samples : list of dicts, each with keys
        boreholes    (K, V, D)  np.ndarray float32 — standardised raw boreholes
        ore_vals     (K,)       np.ndarray float32 — observed ore at drilled cells
        positions    (K, 2)     np.ndarray float32 — normalised [0,1] (x, y)
        target_map   (n_x, n_y) np.ndarray float32 — full ore map (normalised after fitting)
        drill_count  int
        map_idx      int
    """

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def raw_targets(self) -> np.ndarray:
        """Return all ore values from target maps for normalizer fitting."""
        return np.concatenate([s["target_map"].ravel() for s in self.samples])

    def apply_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Normalise stored target_map tensors in-place."""
        for s in self.samples:
            s["target_map"] = normalizer.transform(
                s["target_map"].ravel()
            ).reshape(s["target_map"].shape).astype(np.float32)

    @classmethod
    def from_cache(
        cls,
        cache: NpzMap,
        resources: DecisionSimulationResources,
        cfg: E2EMapBeliefTrainingConfig,
        verbose: bool = True,
        is_val: bool = False,
    ) -> "E2EMapDataset":
        """Build the dataset from a pre-loaded NpzMap.

        For each map, samples ``samples_per_map`` random drill configurations
        (or sequential prefix slices when ``use_sequential_dataset=True``).
        Each configuration stores the raw boreholes at the drilled cells and
        the full target ore map — no candidate sampling is required.

        Parameters
        ----------
        cache     : pre-loaded map pool (train or val subset)
        resources : shared experiment resources (norm stats, variable names, …)
        cfg       : training configuration
        verbose   : print progress every 10 maps
        is_val    : use val-specific sample counts and a shifted RNG seed
        """
        spm = cfg.val_samples_per_map if is_val else cfg.samples_per_map
        seed = cfg.seed + (1 if is_val else 0)
        rng = np.random.default_rng(seed)

        n_x, n_y = cache.n_x, cache.n_y
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

            target_ore = cache.targets[map_idx]  # (n_x, n_y) float32

            def _append(chosen_idx: np.ndarray) -> None:
                drill_locs = [all_locations[k] for k in chosen_idx]
                K = len(drill_locs)
                drill_bhs = np.stack(
                    [bh_arr[i * n_y + j] for i, j in drill_locs], axis=0
                )  # (K, V, D)
                ore_vals_k = np.array(
                    [target_ore[i, j] for i, j in drill_locs], dtype=np.float32
                )  # (K,)
                positions_k = np.stack(
                    [grid_pos[i, j] for i, j in drill_locs], axis=0
                )  # (K, 2)
                samples.append(
                    {
                        "boreholes": drill_bhs,
                        "ore_vals": ore_vals_k,
                        "positions": positions_k,
                        "target_map": target_ore.copy(),
                        "drill_count": K,
                        "map_idx": map_idx,
                    }
                )

            if cfg.use_sequential_dataset:
                for _ in range(cfg.n_sequences_per_map):
                    sequence = rng.permutation(all_idx)
                    for step in cfg.prefix_steps:
                        if step >= len(all_locations):
                            continue
                        _append(sequence[:step])
            else:
                for _ in range(spm):
                    n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                    chosen_idx = rng.choice(all_idx, size=n_drills, replace=False)
                    _append(chosen_idx)

            if verbose and (map_idx + 1) % 10 == 0:
                print(
                    f"  [e2e-map dataset] {map_idx + 1}/{cache.pool_size} maps processed"
                )

        return cls(samples)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_e2e_map(
    model: EndToEndMapBeliefTransformer,
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
    model: EndToEndMapBeliefTransformer,
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

    preds = torch.cat(all_pred, dim=0)    # (N, 1, n_x, n_y)
    tgts = torch.cat(all_tgt, dim=0)     # (N, 1, n_x, n_y)
    counts = torch.cat(all_counts, dim=0) # (N,)

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
    model: EndToEndMapBeliefTransformer,
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
    model: EndToEndMapBeliefTransformer,
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
    model: EndToEndMapBeliefTransformer,
    val_ds: E2EMapDataset,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 20,
) -> None:
    """Save n_plots 4-panel belief-map figures to a timestamped subdirectory."""
    import datetime

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

        # Reconstruct sparse map and mask from sample observations
        n_x, n_y = sample["target_map"].shape
        sparse_ore_map = np.zeros((n_x, n_y), dtype=np.float32)
        observation_mask = np.zeros((n_x, n_y), dtype=np.float32)
        for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
            i = int(round(float(px) * (n_x - 1)))
            j = int(round(float(py) * (n_y - 1)))
            sparse_ore_map[i, j] = float(ov_val)
            observation_mask[i, j] = 1.0

        # target_map is already normalised — denormalise for display
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


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_end_to_end_map_belief(
    resources: DecisionSimulationResources,
    cfg: E2EMapBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: E2EMapDataset | None = None,
    val_ds: E2EMapDataset | None = None,
) -> tuple[EndToEndMapBeliefTransformer, TargetNormalizer]:
    """Train EndToEndMapBeliefTransformer with a full-map reconstruction objective.

    Saves two checkpoints to checkpoint_dir:
      e2e_map_belief_best.pt  — lowest validation MSE (ore-value space)
      e2e_map_belief_last.pt  — final epoch

    Parameters
    ----------
    resources        : shared resources (norm_stats for borehole standardisation;
                       the pre-built borehole encoder in resources is NOT used)
    cfg              : training hyperparameters
    device           : torch device string
    checkpoint_dir   : directory for saved checkpoints
    plot_dir         : if given, save validation plots here after training
    verbose          : print per-epoch metrics
    train_ds/val_ds  : pre-built E2EMapDataset (required)

    Returns
    -------
    (trained EndToEndMapBeliefTransformer with best weights, fitted TargetNormalizer)
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
    model = EndToEndMapBeliefTransformer(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params  : {n_params:,}")
        print(f"  d_model       : {cfg.d_model}")
        print(f"  n_enc_layers  : {cfg.n_encoder_layers}")
        print(f"  n_heads       : {cfg.n_heads}")
        print(f"  latent_dim    : {cfg.latent_dim}")
        print(f"  grid          : {cfg.n_x} × {cfg.n_y}")

    # ---- training loop -------------------------------------------------------
    history: list[dict] = []
    best_val_mse = float("inf")

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

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            save_checkpoint_model(
                checkpoint_dir / "e2e_map_belief_best.pt",
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

    save_checkpoint_model(
        checkpoint_dir / "e2e_map_belief_last.pt",
        model,
        cfg,
        cfg.n_epochs,
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
        checkpoint_dir / "e2e_map_belief_best.pt",
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

    # ---- per-step metrics ---------------------------------------------------
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

    # ---- no-ore false-positive metrics --------------------------------------
    no_ore_metrics = _validate_no_ore_e2e_map(
        model,
        val_loader,
        device,
        normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    # ---- optional validation plots ------------------------------------------
    if plot_dir is not None:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        _save_e2e_map_val_plots(
            model, val_ds, normalizer, plot_dir, device, n_plots=cfg.n_val_plots
        )

    if verbose:
        best_row = min(history, key=lambda r: r["val_mse"])
        print(
            f"\nTraining complete.  Best val MSE: {best_val_mse:.4f}"
            f"  (epoch {best_row['epoch']})"
        )
        print(f"  checkpoints -> {checkpoint_dir}")

    return model, normalizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_e2e_map_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[EndToEndMapBeliefTransformer, E2EMapBeliefTrainingConfig, TargetNormalizer, list[dict]]:
    """Load an EndToEndMapBeliefTransformer checkpoint.

    Returns
    -------
    (model, training_cfg, normalizer, history)
    """
    def _model_fn(ckpt: dict) -> EndToEndMapBeliefTransformer:
        if "model_cfg" in ckpt:
            return EndToEndMapBeliefTransformer(ckpt["model_cfg"])
        return EndToEndMapBeliefTransformer(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(
        path, _model_fn, E2EMapBeliefTrainingConfig, device
    )
