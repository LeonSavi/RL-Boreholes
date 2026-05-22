"""Training pipeline for the end-to-end candidate scoring transformer.

Trains the borehole encoder and candidate scoring transformer jointly so that
borehole embeddings are optimized for spatial ore inference, not reconstruction.

Key differences from train_map_belief.py
-----------------------------------------
* Input is raw boreholes per observation, not pre-computed latent maps.
  The borehole encoder is part of the model graph and receives gradients.
* Target is a scalar ore value at a candidate location, not a full ore map.
* Loss is SmoothL1Loss (Huber) rather than MSE, matching JEPA's training loss.
* Variable-length observation sequences are handled with a custom collate_fn
  that pads borehole tensors and builds an attention key-padding mask.

Dataset structure
-----------------
Each sample contains:
    boreholes     (K, V, D)  — standardised raw boreholes at drilled cells
    ore_vals      (K,)       — observed ore values at drilled cells
    positions     (K, 2)     — normalised [0,1] (x,y) of drilled cells
    candidate_pos (2,)       — normalised [0,1] (cx,cy) of the candidate
    target_ore    scalar     — true ore at the candidate location (raw space)

K varies per sample (min_drills … max_drills), so a custom collate_fn pads
boreholes to the maximum K in each mini-batch and creates a boolean mask.

Experiment variants
-------------------
A  No borehole encoder   train with latent_dim=0 in config overrides
B  End-to-end encoder    default
C  Shuffled boreholes    shuffle_boreholes=True  (breaks spatial correspondence)
D  JEPA initialisation   pretrained_bh_encoder_path="checkpoints/jepa.pt"

Checkpoints
-----------
e2e_best.pt  — lowest validation MSE (ore-value space)
e2e_last.pt  — final epoch
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ..map_cache import NpzMapCache
from ..models.borehole_encoders.autoencoder import standardise
from ..utils import TargetNormalizer
from ..models.end_to_end.candidate_scoring_transformer import (
    E2EConfig,
    CandidateScoringTransformer,
)


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

_ORE_EPS: float = 1e-3  # ore values below this are treated as "no ore"

# Probabilities for each candidate ore-value bin when the map has ore
_CANDIDATE_TYPE_PROBS: dict[str, float] = {
    "zero": 0.40,
    "low":  0.20,
    "mid":  0.20,
    "high": 0.20,
}

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class E2ETrainingConfig:
    """Hyperparameters for training the end-to-end candidate scoring model."""

    # Dataset
    n_train_maps: int = 50
    samples_per_map: int = 20
    candidates_per_sample: int = 1   # candidates sampled per (map, drilling-history) pair
    n_val_maps: int = 30
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # Sequential dataset mode: ordered drill sequences at fixed prefix lengths.
    # Mirrors the real exploration setting where each drill informs the next.
    # When False (default), K drills are sampled randomly from [min_drills, max_drills].
    use_sequential_dataset: bool = False
    n_sequences_per_map: int = 3
    prefix_steps: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15])

    # Borehole dimensions — resolved from resources at training time
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder architecture
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))
    bh_d_model: int = 128
    bh_n_heads: int = 4
    bh_n_layers: int = 2
    latent_dim: int = 128

    # Candidate scoring transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    head_hidden_dim: int = 128

    # Positional encoding
    pe_max_freq: float = 10000.0

    # Optimisation
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50
    norm_mode: str = "log1p"         # "log1p" | "zscore" | "none"
    grad_clip_norm: float = 1.0      # 0.0 = disabled

    # False-positive penalty: penalise high predictions where true ore = 0
    use_false_positive_penalty: bool = False
    false_positive_weight: float = 0.1
    fp_threshold: float = 1e-3       # ore values below this are treated as "no ore"

    # Experiment controls
    shuffle_boreholes: bool = False        # variant C sanity check
    pretrained_bh_encoder_path: str | None = None  # variant D: JEPA init

    # Grid dimensions — set automatically from cache in train_end_to_end()
    n_x: int = 32
    n_y: int = 32

    # Misc
    seed: int = 42
    borehole_encoder: str = "end_to_end"  # informational; stored in checkpoint
    n_val_plots: int = 20

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    def to_model_config(self) -> E2EConfig:
        """Build an E2EConfig from the architectural fields of this dataclass."""
        return E2EConfig(
            n_variables=self.n_variables,
            n_depth=self.n_depth,
            bh_channels=self.bh_channels,
            bh_d_model=self.bh_d_model,
            bh_n_heads=self.bh_n_heads,
            bh_n_layers=self.bh_n_layers,
            latent_dim=self.latent_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_hidden_dim=self.head_hidden_dim,
            pe_max_freq=self.pe_max_freq,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class E2EDataset(Dataset):
    """Pre-generated (partial observation + candidate) → scalar ore pairs.

    Each sample stores raw standardised boreholes at the drilled cells so
    the borehole encoder can be trained end-to-end.  This is intentionally
    different from GeologicalBeliefDataset, which stores pre-computed latents.

    Attributes
    ----------
    samples : list of dicts, each with keys
        boreholes      (K, V, D) np.ndarray float32
        ore_vals       (K,)      np.ndarray float32
        positions      (K, 2)   np.ndarray float32  — normalised [0,1]
        candidate_pos  (2,)     np.ndarray float32  — normalised [0,1]
        target_ore     float32  — true ore at candidate (raw space)
    """

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples
        self._targets_raw: np.ndarray | None = None   # cached for normalizer fitting

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def raw_targets(self) -> np.ndarray:
        """Return all target ore values as a 1-D float32 array (raw space)."""
        if self._targets_raw is None:
            self._targets_raw = np.array(
                [s["target_ore"] for s in self.samples], dtype=np.float32
            )
        return self._targets_raw

    def apply_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Normalize stored target_ore values in-place."""
        for s in self.samples:
            s["target_ore"] = float(
                normalizer.transform(np.array([s["target_ore"]], dtype=np.float32))[0]
            )
        self._targets_raw = None  # invalidate cache

    @classmethod
    def from_cache(
        cls,
        cache: NpzMapCache,
        resources: DecisionSimulationResources,
        cfg: E2ETrainingConfig,
        verbose: bool = True,
        is_val: bool = False,
    ) -> "E2EDataset":
        """Build the dataset from a pre-loaded NpzMapCache.

        Reads boreholes and ore targets directly from the cache rather than
        generating maps on-the-fly, which avoids the simulator overhead on
        remote storage such as Google Drive.

        Parameters
        ----------
        cache     : pre-loaded map pool (train or val subset)
        resources : shared experiment resources (norm stats, variable names, …)
        cfg       : training configuration
        verbose   : print progress every 10 maps
        is_val    : if True, use val-specific sample counts and shift the RNG seed
        """
        spm = cfg.val_samples_per_map if is_val else cfg.samples_per_map
        seed = cfg.seed + (1 if is_val else 0)

        rng = np.random.default_rng(seed)

        n_x, n_y = cache.n_x, cache.n_y
        variables = resources.variable_names

        # Pre-compute normalised grid positions — (n_x, n_y, 2) float32
        xs = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys, indexing="ij")
        grid_pos = np.stack([xg, yg], axis=-1)  # (n_x, n_y, 2)

        all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]

        samples: list[dict] = []

        for map_idx in range(cache.pool_size):
            # Load and standardise boreholes — shape (n_x*n_y, V, D), raw in cache
            bh_arr = cache.borehole_arrays[map_idx].copy()
            if resources.norm_stats:
                bh_arr = standardise(bh_arr, resources.norm_stats, variables)
            bh_arr = np.nan_to_num(bh_arr, nan=0.0).astype(np.float32)

            target_ore = cache.targets[map_idx]  # (n_x, n_y) float32

            # Per-map ore statistics used by stratified candidate sampling
            flat_ore = target_ore.reshape(-1)  # (n_x * n_y,)
            has_ore = bool((flat_ore > _ORE_EPS).any())

            def _append_sample(chosen_idx: np.ndarray, cand_flat: int) -> None:
                """Build one sample dict and append it to ``samples``."""
                drill_locs = [all_locations[k] for k in chosen_idx]
                ci, cj = all_locations[cand_flat]
                n_k = len(drill_locs)

                drill_bhs = np.stack(
                    [bh_arr[i * n_y + j] for i, j in drill_locs], axis=0
                )  # (K, V, D)
                ore_vals_k = np.array(
                    [target_ore[i, j] for i, j in drill_locs], dtype=np.float32
                )  # (K,)
                positions_k = np.stack(
                    [grid_pos[i, j] for i, j in drill_locs], axis=0
                )  # (K, 2)

                if cfg.shuffle_boreholes:
                    perm = rng.permutation(n_k)
                    drill_bhs = drill_bhs[perm]

                samples.append({
                    "boreholes":     drill_bhs,
                    "ore_vals":      ore_vals_k,
                    "positions":     positions_k,
                    "candidate_pos": grid_pos[ci, cj],
                    "target_ore":    float(target_ore[ci, cj]),
                    "map_idx":       map_idx,
                })

            if cfg.use_sequential_dataset:
                # Sequential path: fixed drill orderings, prefix slices at each step.
                # Mirrors the real exploration setting where each borehole result
                # informs the choice of the next drill location.
                all_loc_idx = np.arange(len(all_locations))
                for _ in range(cfg.n_sequences_per_map):
                    sequence = rng.permutation(all_loc_idx)
                    for step in cfg.prefix_steps:
                        if step >= len(all_locations):
                            continue
                        chosen_idx = sequence[:step]
                        visited = set(chosen_idx.tolist())
                        for _ in range(cfg.candidates_per_sample):
                            pools = _build_candidate_pools(flat_ore, visited)
                            cand_idx = _sample_candidate_from_pools(pools, has_ore, rng)
                            _append_sample(chosen_idx, cand_idx)
            else:
                # Random path: K drills sampled uniformly from [min_drills, max_drills].
                all_idx = np.arange(len(all_locations))
                for _ in range(spm):
                    for _ in range(cfg.candidates_per_sample):
                        n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                        chosen_idx = rng.choice(all_idx, size=n_drills, replace=False)
                        visited = set(chosen_idx.tolist())
                        pools = _build_candidate_pools(flat_ore, visited)
                        cand_idx = _sample_candidate_from_pools(pools, has_ore, rng)
                        _append_sample(chosen_idx, cand_idx)

            if verbose and (map_idx + 1) % 10 == 0:
                print(f"  [e2e dataset] {map_idx + 1}/{cache.pool_size} maps processed")

        return cls(samples)


# ---------------------------------------------------------------------------
# Stratified candidate sampling helpers
# ---------------------------------------------------------------------------

def _build_candidate_pools(
    flat_ore: np.ndarray,
    visited: set[int],
    ore_eps: float = _ORE_EPS,
) -> dict[str, np.ndarray]:
    """Partition unvisited cell indices into ore-value bins.

    Quantiles are computed exclusively over positive-ore cells so that
    the bins reflect the ore distribution rather than the (dominant)
    empty-space distribution.

    Returns
    -------
    dict with keys 'zero', 'low', 'mid', 'high', each a 1-D int array of
    flat cell indices that are *not* in the visited set.
    """
    visited_arr = np.fromiter(visited, dtype=np.intp, count=len(visited))
    unvisited_mask = np.ones(len(flat_ore), dtype=bool)
    unvisited_mask[visited_arr] = False

    positive_vals = flat_ore[flat_ore > ore_eps]
    if positive_vals.size == 0:
        # No ore on this map — everything goes in the zero pool
        return {
            "zero": np.where(unvisited_mask)[0],
            "low":  np.empty(0, dtype=np.intp),
            "mid":  np.empty(0, dtype=np.intp),
            "high": np.empty(0, dtype=np.intp),
        }

    q50 = float(np.quantile(positive_vals, 0.50))
    q80 = float(np.quantile(positive_vals, 0.80))

    def _pool(cond: np.ndarray) -> np.ndarray:
        return np.where(cond & unvisited_mask)[0]

    return {
        "zero": _pool(flat_ore <= ore_eps),
        "low":  _pool((flat_ore > ore_eps) & (flat_ore <= q50)),
        "mid":  _pool((flat_ore > q50)     & (flat_ore <= q80)),
        "high": _pool(flat_ore > q80),
    }


def _sample_candidate_from_pools(
    pools: dict[str, np.ndarray],
    has_ore: bool,
    rng: np.random.Generator,
    type_probs: dict[str, float] = _CANDIDATE_TYPE_PROBS,
) -> int:
    """Sample one candidate flat index from the stratified pools.

    For no-ore maps samples uniformly from all unvisited cells.
    For ore maps uses type_probs, with fallback to any non-empty pool.

    Raises
    ------
    ValueError if every pool is empty (all cells are visited).
    """
    if not has_ore:
        pool = pools["zero"]
        if pool.size == 0:
            pool = np.concatenate(list(pools.values()))
        return int(rng.choice(pool))

    types = list(type_probs.keys())
    probs = np.array([type_probs[t] for t in types], dtype=np.float64)
    probs /= probs.sum()

    chosen_type = types[int(rng.choice(len(types), p=probs))]
    fallback_order = [chosen_type] + [t for t in ("high", "mid", "low", "zero") if t != chosen_type]

    for t in fallback_order:
        if pools[t].size > 0:
            return int(rng.choice(pools[t]))

    # Final fallback: any remaining unvisited cell
    all_candidates = np.concatenate(list(pools.values()))
    if all_candidates.size > 0:
        return int(rng.choice(all_candidates))

    raise ValueError("All candidate pools are empty — every cell has been visited.")


# ---------------------------------------------------------------------------
# Custom collate for variable-length observation sequences
# ---------------------------------------------------------------------------

def collate_e2e(batch: list[dict]) -> dict:
    """Pad borehole sequences to the longest K in the batch.

    Padding positions are identified by a boolean mask (True = ignore).

    Returns a dict with keys:
        boreholes      (B, max_K, V, D) float32 tensor
        ore_vals       (B, max_K)       float32 tensor  — 0 at padding
        positions      (B, max_K, 2)    float32 tensor  — 0 at padding
        candidate_pos  (B, 2)           float32 tensor
        target_ore     (B, 1)           float32 tensor
        padding_mask   (B, max_K)       bool tensor  — True at padding rows
    """
    max_K = max(s["boreholes"].shape[0] for s in batch)
    B = len(batch)

    # Infer V and D from the first sample
    V, D = batch[0]["boreholes"].shape[1], batch[0]["boreholes"].shape[2]

    boreholes_pad  = np.zeros((B, max_K, V, D), dtype=np.float32)
    ore_pad        = np.zeros((B, max_K),        dtype=np.float32)
    pos_pad        = np.zeros((B, max_K, 2),     dtype=np.float32)
    padding_mask   = np.ones((B, max_K),         dtype=bool)   # True = padded
    cand_pos_arr   = np.zeros((B, 2),            dtype=np.float32)
    targets        = np.zeros((B, 1),            dtype=np.float32)

    for i, s in enumerate(batch):
        K = s["boreholes"].shape[0]
        boreholes_pad[i, :K]  = s["boreholes"]
        ore_pad[i, :K]        = s["ore_vals"]
        pos_pad[i, :K]        = s["positions"]
        padding_mask[i, :K]   = False           # real observations → not padded
        cand_pos_arr[i]        = s["candidate_pos"]
        targets[i, 0]          = s["target_ore"]

    return {
        "boreholes":     torch.from_numpy(boreholes_pad),
        "ore_vals":      torch.from_numpy(ore_pad),
        "positions":     torch.from_numpy(pos_pad),
        "candidate_pos": torch.from_numpy(cand_pos_arr),
        "target_ore":    torch.from_numpy(targets),
        "padding_mask":  torch.from_numpy(padding_mask),
    }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _false_positive_loss(
    pred_norm: torch.Tensor,
    tgt_norm: torch.Tensor,
    normalizer: TargetNormalizer,
    threshold: float = 1e-3,
) -> torch.Tensor:
    """Penalise positive predictions where the true ore value is (near) zero.

    Adapted from training_utils.false_positive_loss for scalar targets.

    Returns a zero tensor when no no-ore samples are present in the batch.
    """
    tgt_ore = normalizer.inverse_tensor(tgt_norm)          # (B, 1) — raw space
    no_ore = tgt_ore.squeeze(-1) < threshold               # (B,) bool
    if not no_ore.any():
        return pred_norm.new_zeros(())
    return pred_norm[no_ore].clamp(min=0.0).mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_e2e(
    model: CandidateScoringTransformer,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Validation in ore-value (denormalised) space.

    Computes scalar MSE, MAE, and batch-level Pearson correlation across the
    entire validation set.  Pearson is computed over all samples jointly
    (not averaged per batch) because per-sample correlation is undefined for
    scalar predictions.
    """
    model.eval()
    all_pred: list[torch.Tensor] = []
    all_tgt:  list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            bh  = batch["boreholes"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            cp  = batch["candidate_pos"].to(device)
            tgt = batch["target_ore"].to(device)
            pm  = batch["padding_mask"].to(device)

            pred_norm = model(bh, ov, pos, cp, pm)
            pred = normalizer.inverse_tensor(pred_norm)   # (B, 1)
            tgt_raw = normalizer.inverse_tensor(tgt)      # (B, 1)

            all_pred.append(pred.cpu())
            all_tgt.append(tgt_raw.cpu())

    preds = torch.cat(all_pred).view(-1)   # (N,)
    tgts  = torch.cat(all_tgt).view(-1)    # (N,)

    mse = nn.functional.mse_loss(preds, tgts).item()
    mae = (preds - tgts).abs().mean().item()

    p_c = preds - preds.mean()
    t_c = tgts  - tgts.mean()
    corr = (
        (p_c * t_c).sum() /
        (p_c.norm() * t_c.norm()).clamp(min=1e-8)
    ).item()

    return {"val_mse": mse, "val_mae": mae, "val_corr": corr}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_end_to_end(
    resources: DecisionSimulationResources,
    cfg: E2ETrainingConfig,
    device: str,
    checkpoint_dir: Path,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: E2EDataset | None = None,
    val_ds: E2EDataset | None = None,
    train_cache: NpzMapCache | None = None,
    val_cache: NpzMapCache | None = None,
) -> tuple[CandidateScoringTransformer, TargetNormalizer]:
    """Train the end-to-end candidate scoring transformer.

    Saves two checkpoints to checkpoint_dir:
      e2e_best.pt  — lowest validation MSE (ore-value space)
      e2e_last.pt  — final epoch

    Parameters
    ----------
    resources        : shared resources (norm stats used for borehole
                       standardisation; the borehole encoder inside resources
                       is NOT used — the e2e model trains its own)
    cfg              : training hyperparameters
    device           : torch device string
    checkpoint_dir   : directory for saved checkpoints
    verbose          : print per-epoch metrics
    train_ds/val_ds  : pre-built E2EDataset (takes priority over caches)
    train_cache/val_cache : NpzMapCache to build datasets from when datasets
                            are not provided directly

    Returns
    -------
    (trained CandidateScoringTransformer with best weights, fitted TargetNormalizer)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- resolve n_variables and n_depth from resources ----------------------
    if resources.variable_names:
        cfg.n_variables = len(resources.variable_names)

    # ---- datasets ------------------------------------------------------------
    if train_ds is None:
        if train_cache is None:
            raise ValueError(
                "Provide either train_ds (a pre-built E2EDataset) or "
                "train_cache (an NpzMapCache) to build the training dataset from."
            )
        if verbose:
            print("Building training dataset from cache …")
        train_ds = E2EDataset.from_cache(train_cache, resources, cfg, verbose=verbose, is_val=False)

    if val_ds is None:
        if val_cache is None:
            raise ValueError(
                "Provide either val_ds (a pre-built E2EDataset) or "
                "val_cache (an NpzMapCache) to build the validation dataset from."
            )
        if verbose:
            print("Building validation dataset from cache …")
        val_ds = E2EDataset.from_cache(val_cache, resources, cfg, verbose=verbose, is_val=True)

    # Resolve grid dimensions and depth from cache / dataset
    if train_cache is not None:
        cfg.n_x = train_cache.n_x
        cfg.n_y = train_cache.n_y
    if cfg.n_depth == 440 and len(train_ds) > 0:
        cfg.n_depth = train_ds.samples[0]["boreholes"].shape[2]

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
        collate_fn=collate_e2e,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_e2e,
    )

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ---------------------------------------------------------------
    model_cfg = cfg.to_model_config()
    model = CandidateScoringTransformer(model_cfg).to(device)

    # Optional: initialise borehole encoder from pretrained JEPA weights
    if cfg.pretrained_bh_encoder_path and model_cfg.use_encoder:
        _load_jepa_backbone_weights(model, cfg.pretrained_bh_encoder_path, device, verbose)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params  : {n_params:,}")
        print(f"  d_model       : {cfg.d_model}")
        print(f"  n_layers      : {cfg.n_layers}")
        print(f"  n_heads       : {cfg.n_heads}")
        print(f"  latent_dim    : {cfg.latent_dim}  (0 = no encoder, variant A)")
        if cfg.shuffle_boreholes:
            print("  variant C     : borehole shuffling enabled")
        if cfg.pretrained_bh_encoder_path:
            print(f"  variant D     : JEPA init from {cfg.pretrained_bh_encoder_path}")

    # ---- checkpoint helper ---------------------------------------------------
    history: list[dict] = []

    def _save_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg":        cfg,
                "model_cfg":  model_cfg,
                "epoch":      epoch,
                "history":    history,
                "normalizer": normalizer,
            },
            path,
        )

    # ---- training loop -------------------------------------------------------
    best_val_mse = float("inf")

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            bh  = batch["boreholes"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            cp  = batch["candidate_pos"].to(device)
            tgt = batch["target_ore"].to(device)    # (B, 1) — normalised
            pm  = batch["padding_mask"].to(device)

            pred = model(bh, ov, pos, cp, pm)       # (B, 1)
            loss = loss_fn(pred, tgt)

            if cfg.use_false_positive_penalty:
                fp_loss = _false_positive_loss(pred, tgt, normalizer, cfg.fp_threshold)
                loss = loss + cfg.false_positive_weight * fp_loss

            optimiser.zero_grad()
            loss.backward()

            if cfg.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimiser.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss_avg = train_loss_sum / n_batches
        val_metrics = _validate_e2e(model, val_loader, device, normalizer)

        row = {"epoch": epoch, "train_loss": train_loss_avg, **val_metrics}
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{cfg.n_epochs}"
                f"  train_loss={train_loss_avg:.4f}"
                f"  val_mse={val_metrics['val_mse']:.4f}"
                f"  val_mae={val_metrics['val_mae']:.4f}"
                f"  val_corr={val_metrics['val_corr']:.4f}"
            )

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            _save_ckpt(checkpoint_dir / "e2e_best.pt", epoch)

    _save_ckpt(checkpoint_dir / "e2e_last.pt", cfg.n_epochs)
    _export_history(history, checkpoint_dir)

    # ---- reload best weights -------------------------------------------------
    best_ckpt = torch.load(
        checkpoint_dir / "e2e_best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    # ---- per-step / per-drill-count metrics (best model) ---------------------
    step_metrics = _validate_e2e_by_step(model, val_loader, device, normalizer)
    if step_metrics:
        if verbose:
            label = "step" if cfg.use_sequential_dataset else "drill count"
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

    # ---- optional validation plots -------------------------------------------
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _save_e2e_val_plots(model, val_ds, val_loader, normalizer, plot_dir, device, cfg, val_cache=val_cache)

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

def load_e2e_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[CandidateScoringTransformer, E2ETrainingConfig, TargetNormalizer, list[dict]]:
    """Load a saved CandidateScoringTransformer checkpoint.

    Returns
    -------
    (model, training_config, normalizer, training_history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "model_cfg" in ckpt:
        model_cfg: E2EConfig = ckpt["model_cfg"]
    else:
        train_cfg: E2ETrainingConfig = ckpt["cfg"]
        model_cfg = train_cfg.to_model_config()

    model = CandidateScoringTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normalizer: TargetNormalizer = ckpt.get("normalizer", TargetNormalizer(mode="none"))
    train_cfg = ckpt.get("cfg", E2ETrainingConfig())

    return model, train_cfg, normalizer, ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_e2e_training_config(
    debug: bool = False,
    overrides: dict | None = None,
) -> E2ETrainingConfig:
    """Build an E2ETrainingConfig with optional field overrides.

    Parameters
    ----------
    debug     : if True, use minimal dataset / model for fast smoke testing
    overrides : dict of field names → values applied after defaults

    Raises
    ------
    ValueError if any override key is not a valid E2ETrainingConfig field.
    """
    overrides = overrides or {}
    valid_fields = {f.name for f in dataclasses.fields(E2ETrainingConfig)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise ValueError(
            f"Unknown E2ETrainingConfig field(s): {sorted(invalid)}.\n"
            f"Valid fields: {sorted(valid_fields)}"
        )

    if debug:
        cfg = E2ETrainingConfig(
            n_train_maps=2,
            samples_per_map=4,
            candidates_per_sample=1,
            n_val_maps=1,
            val_samples_per_map=4,
            n_epochs=2,
            batch_size=4,
            d_model=64,
            n_heads=4,
            n_layers=1,
            d_ff=128,
            head_hidden_dim=32,
            bh_d_model=32,
            bh_n_heads=4,
            bh_n_layers=1,
            latent_dim=32,
        )
    else:
        cfg = E2ETrainingConfig()

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Validation metrics helpers (by drill count / step)
# ---------------------------------------------------------------------------

def _validate_e2e_by_step(
    model: CandidateScoringTransformer,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[int, dict[str, float]]:
    """Compute MSE/MAE/Pearson grouped by exact K (number of observed boreholes).

    In sequential mode K equals the prefix step, so this gives per-step metrics.
    In random mode K is sampled from [min_drills, max_drills].

    Returns
    -------
    dict mapping K → {"n": int, "mse": float, "mae": float, "corr": float}
    """
    from collections import defaultdict

    model.eval()
    groups: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))

    with torch.no_grad():
        for batch in val_loader:
            bh  = batch["boreholes"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            cp  = batch["candidate_pos"].to(device)
            tgt = batch["target_ore"].to(device)
            pm  = batch["padding_mask"].to(device)

            pred_norm = model(bh, ov, pos, cp, pm)
            preds = normalizer.inverse_tensor(pred_norm).cpu().view(-1).tolist()
            tgts  = normalizer.inverse_tensor(tgt).cpu().view(-1).tolist()
            ks    = (~pm).sum(dim=1).cpu().tolist()

            for k, p, t in zip(ks, preds, tgts):
                groups[int(k)][0].append(p)
                groups[int(k)][1].append(t)

    metrics: dict[int, dict[str, float]] = {}
    for k in sorted(groups):
        ps = torch.tensor(groups[k][0])
        ts = torch.tensor(groups[k][1])
        mse  = nn.functional.mse_loss(ps, ts).item()
        mae  = (ps - ts).abs().mean().item()
        pc   = ps - ps.mean()
        tc   = ts - ts.mean()
        corr = ((pc * tc).sum() / (pc.norm() * tc.norm()).clamp(min=1e-8)).item()
        metrics[k] = {"n": len(ps), "mse": mse, "mae": mae, "corr": corr}

    return metrics


def _save_e2e_val_plots(
    model: CandidateScoringTransformer,
    val_ds: "E2EDataset",
    val_loader: DataLoader,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    cfg: E2ETrainingConfig,
    val_cache: NpzMapCache | None = None,
) -> None:
    """Save validation plots to a timestamped subdir under plot_dir.

    Produces (when val_cache is available):
      val_sample_XX.png  — 4-panel belief map per sample (observations / true /
                           predicted / error), matching the other training scripts.

    Always produces:
      scatter_overall.png  — predicted vs true ore across all validation samples
      scatter_by_step.png  — one panel per unique drill count / prefix step
    """
    import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..visualize import plot_belief_sample

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(plot_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    n_x, n_y = cfg.n_x, cfg.n_y

    # Pre-compute the full normalised grid — (n_x*n_y, 2) — used as candidates
    xs = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    all_positions = np.stack([xg.ravel(), yg.ravel()], axis=-1)  # (n_x*n_y, 2)
    all_positions_t = torch.from_numpy(all_positions).to(device)

    model.eval()

    # ---- 4-panel belief maps (matches plot_belief_sample output) ----------------
    if val_cache is not None:
        indices = np.linspace(0, len(val_ds) - 1, cfg.n_val_plots, dtype=int)
        for plot_k, idx in enumerate(indices):
            sample = val_ds.samples[int(idx)]
            map_idx = sample.get("map_idx")
            if map_idx is None:
                continue

            bh  = torch.from_numpy(sample["boreholes"]).to(device)   # (K, V, D)
            ov  = torch.from_numpy(sample["ore_vals"]).to(device)    # (K,)
            pos = torch.from_numpy(sample["positions"]).to(device)   # (K, 2)

            with torch.no_grad():
                scores_norm = model.score_candidates(
                    bh, ov, pos, all_positions_t
                )  # (n_x*n_y,) normalised

            predicted_ore_map = (
                normalizer.inverse(scores_norm.cpu().numpy())
                .reshape(n_x, n_y)
                .astype(np.float32)
            )

            # Sparse ore map and observation mask from drilled positions
            sparse_ore_map  = np.zeros((n_x, n_y), dtype=np.float32)
            observation_mask = np.zeros((n_x, n_y), dtype=np.float32)
            pos_np = sample["positions"]  # (K, 2) normalised
            ore_np = sample["ore_vals"]   # (K,)
            for (px, py), ov_val in zip(pos_np, ore_np):
                i = int(round(float(px) * (n_x - 1)))
                j = int(round(float(py) * (n_y - 1)))
                sparse_ore_map[i, j]   = float(ov_val)
                observation_mask[i, j] = 1.0

            true_ore_map = val_cache.targets[map_idx].astype(np.float32)
            n_drills = int(observation_mask.sum())

            plot_belief_sample(
                sparse_ore_map=sparse_ore_map,
                observation_mask=observation_mask,
                true_ore_map=true_ore_map,
                predicted_ore_map=predicted_ore_map,
                save_path=out_dir / f"val_sample_{plot_k:02d}.png",
                title=f"Val sample {plot_k}  ({n_drills} drills)",
                timestamp=timestamp,
            )

    # ---- scatter plots (predicted vs true) -------------------------------------
    all_ks:    list[int]   = []
    all_preds: list[float] = []
    all_tgts:  list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            bh  = batch["boreholes"].to(device)
            ov  = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            cp  = batch["candidate_pos"].to(device)
            tgt = batch["target_ore"].to(device)
            pm  = batch["padding_mask"].to(device)

            pred_norm = model(bh, ov, pos, cp, pm)
            preds = normalizer.inverse_tensor(pred_norm).cpu().view(-1).tolist()
            tgts  = normalizer.inverse_tensor(tgt).cpu().view(-1).tolist()
            ks    = (~pm).sum(dim=1).cpu().tolist()

            all_ks.extend(int(k) for k in ks)
            all_preds.extend(preds)
            all_tgts.extend(tgts)

    preds_arr = np.array(all_preds)
    tgts_arr  = np.array(all_tgts)
    ks_arr    = np.array(all_ks)

    def _scatter_ax(ax: "plt.Axes", p: np.ndarray, t: np.ndarray, title: str) -> None:
        ax.scatter(t, p, alpha=0.3, s=8, color="steelblue", rasterized=True)
        vmax = float(max(t.max(), p.max(), 1e-6))
        ax.plot([0, vmax], [0, vmax], "r--", lw=1)
        ax.set_xlabel("True ore")
        ax.set_ylabel("Predicted ore")
        ax.set_title(title)

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_ax(ax, preds_arr, tgts_arr, f"E2E: pred vs true ore (n={len(preds_arr)})")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_overall.png", dpi=120)
    plt.close(fig)

    unique_ks = sorted(set(all_ks))
    if len(unique_ks) > 1:
        ncols = min(4, len(unique_ks))
        nrows = (len(unique_ks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        for i, k in enumerate(unique_ks):
            row, col = divmod(i, ncols)
            mask = ks_arr == k
            _scatter_ax(axes[row][col], preds_arr[mask], tgts_arr[mask],
                        f"K={k} drills (n={mask.sum()})")
        for i in range(len(unique_ks), nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row][col].set_visible(False)
        label = "step" if cfg.use_sequential_dataset else "drill count"
        fig.suptitle(f"E2E: pred vs true ore by {label}", y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / "scatter_by_step.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  plots saved -> {out_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_jepa_backbone_weights(
    model: CandidateScoringTransformer,
    jepa_path: str,
    device: str,
    verbose: bool,
) -> None:
    """Copy CNN backbone weights from a JEPA checkpoint into the borehole encoder.

    Only the convolutional layers are transferred; the transformer and
    projection layers of the new encoder are left randomly initialised since
    the JEPA predictor architecture differs.
    """
    from ..models.borehole_encoders.jepa_encoder import JEPAModel

    try:
        ckpt = torch.load(jepa_path, map_location=device, weights_only=False)
        jepa_state = ckpt.get("context_encoder", ckpt.get("state_dict", ckpt))

        target_state = model.bh_encoder.state_dict()
        transferred = 0
        for name, param in jepa_state.items():
            # JEPA stores backbone weights under "backbone.conv.*"
            if name.startswith("backbone.conv."):
                new_name = "conv." + name[len("backbone.conv."):]
                if new_name in target_state and target_state[new_name].shape == param.shape:
                    target_state[new_name].copy_(param)
                    transferred += 1

        model.bh_encoder.load_state_dict(target_state)
        if verbose:
            print(f"  JEPA init     : transferred {transferred} conv layer weight tensors")
    except Exception as exc:
        if verbose:
            print(f"  JEPA init     : FAILED ({exc}) — continuing with random init")


def _export_history(history: list[dict], directory: Path) -> None:
    """Write training history to JSON and CSV."""
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
