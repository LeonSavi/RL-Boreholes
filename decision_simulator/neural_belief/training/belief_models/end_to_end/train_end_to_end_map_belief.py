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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ....map_hdf5 import MapPool
from ....models.belief_models.borehole_encoders.autoencoder import standardise
from ....models.belief_models.end_to_end.end_to_end_map_belief_transformer import (
    EndToEndMapBeliefTransformer,
)
from ....training_utils import TargetNormalizer
from ....training_utils import (
    false_positive_loss,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from ..training_configs import E2EMapBeliefConfig
from .helpers import validate_e2e_map, model_validation, collate_e2e_map


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
        cache: MapPool,
        resources: DecisionSimulationResources,
        cfg: E2EMapBeliefConfig,
        verbose: bool = True,
        is_val: bool = False,
    ) -> "E2EMapDataset":
        """Build the dataset from a pre-loaded MapPool.

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

            def _append(chosen_idx: np.ndarray, sequence_id: int = 0) -> None:
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
                    f"  [e2e-map dataset] {map_idx + 1}/{cache.pool_size} maps processed"
                )

        return cls(samples)



# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_end_to_end_map_belief(
    resources: DecisionSimulationResources,
    cfg: E2EMapBeliefConfig,
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
    best_epoch = 0
    patience_counter = 0
    epoch = 0  # defined here so it is accessible after a potential early-stop break

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
                checkpoint_dir / "e2e_map_belief_best.pt",
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
        model, checkpoint_dir, "e2e_map_belief_best.pt", history,
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

def load_e2e_map_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[EndToEndMapBeliefTransformer, E2EMapBeliefConfig, TargetNormalizer, list[dict]]:
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
        path, _model_fn, E2EMapBeliefConfig, device
    )
