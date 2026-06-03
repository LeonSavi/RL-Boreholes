"""Guided-exploration curriculum training for EndToEndMapBeliefTransformer.

Hypothesis
----------
Random drilling underrepresents rare but decision-critical high-ore observations.
By exposing the belief model to a mixture of random trajectories (75 %) and
guided trajectories (25 %), the model may learn stronger ore-body reconstruction,
better peak-ore prediction, and improved use of informative boreholes -- without
changing the model architecture at all.

Guided drilling modes
---------------------
Mode A  (uncertainty-only)
    After ``phase1_drills`` random drills, every subsequent drill is chosen at the
    cell with the highest predicted uncertainty according to the guide model.
    When no guide model is loaded, falls back to max-distance-from-drills heuristic.

Mode AB (uncertainty + ore-assisted)
    Phase-2 steps alternate: uncertainty-select / ore-select.
    Ore-select picks a random cell from the top-N% ore cells (oracle, training only).

Mode AC (uncertainty + boundary-assisted)
    Phase-2 steps alternate: uncertainty-select / boundary-select.
    Boundary-select picks a random cell near the ore-body boundary (oracle, training only).

Mixing
------
  75 % random sequences (identical to train_end_to_end_map_belief.py)
  25 % guided sequences:
      50 % Mode A
      25 % Mode AB
      25 % Mode AC

Oracle isolation
----------------
``yield_target`` (the true ore map) is accessed ONLY during sequence generation, inside
``_derive_ore_features`` and ``_build_guided_sequence``.  The model-facing sample dict
contains only the fields that the existing training loop produces: ``boreholes``,
``ore_vals``, ``positions``, ``target_map``, ``drill_count``.
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from torch.utils.data import DataLoader, Dataset

from decision_simulator.resources import DecisionSimulationResources
from ...map_cache import NpzMap
from ...models.belief_models.borehole_encoders.autoencoder import standardise
from ...models.belief_models.end_to_end.end_to_end_map_belief_transformer import (
    EndToEndMapBeliefTransformer,
)
from ...training_utils import TargetNormalizer
from ...training_utils import (
    false_positive_loss,
    load_model_encoder_checkpoint,
    save_checkpoint_model,
)
from ..belief_models.training_configs import GuidedExplorationConfig
from .train_end_to_end_map_belief import E2EMapDataset
from .helpers import validate_e2e_map, model_validation, collate_e2e_map


# ---------------------------------------------------------------------------
# Ore-feature extraction
# ---------------------------------------------------------------------------


class OreFeatures(NamedTuple):
    """Per-map oracle features derived from yield_target at generation time."""

    high_ore_idx: np.ndarray    # 1-D flat indices of top-N% cells
    boundary_idx: np.ndarray    # 1-D flat indices of ore-boundary cells
    has_ore: bool               # False for maps with no ore body


def _derive_ore_features(
    yield_target: np.ndarray,
    ore_top_pct: float,
    boundary_ore_pct: float,
) -> OreFeatures:
    """Derive oracle drilling-target features from the 2-D ore map.

    All computation is done on the raw yield_target at sequence-generation time.
    The model never sees these features.
    """
    nonzero = yield_target[yield_target > 0]
    if nonzero.size == 0:
        n = yield_target.size
        return OreFeatures(
            high_ore_idx=np.arange(n),
            boundary_idx=np.arange(n),
            has_ore=False,
        )

    # High-ore cells: top ore_top_pct of nonzero cells
    high_thresh = np.percentile(nonzero, (1.0 - ore_top_pct) * 100.0)
    high_mask = yield_target >= high_thresh
    high_ore_idx = np.flatnonzero(high_mask)

    # Ore-body boundary: cells on the edge of the thresholded ore region
    boundary_thresh = np.percentile(nonzero, boundary_ore_pct * 100.0)
    ore_mask = yield_target >= boundary_thresh
    eroded = binary_erosion(ore_mask, iterations=1)
    boundary_mask = ore_mask & ~eroded
    boundary_idx = np.flatnonzero(boundary_mask)
    if boundary_idx.size == 0:
        # Degenerate case: ore region is a single cell or fully eroded
        boundary_idx = np.flatnonzero(ore_mask)

    return OreFeatures(
        high_ore_idx=high_ore_idx,
        boundary_idx=boundary_idx,
        has_ore=True,
    )


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------


@dataclass
class GuidedTrainingStats:
    n_random: int = 0
    n_guided_a: int = 0
    n_guided_ab: int = 0
    n_guided_ac: int = 0
    n_ore_selected: int = 0
    n_boundary_selected: int = 0
    n_guided_fallback: int = 0   # guided mode fell back to random (no-ore map)

    @property
    def n_total(self) -> int:
        return self.n_random + self.n_guided_a + self.n_guided_ab + self.n_guided_ac

    def fractions(self) -> dict[str, float]:
        n = max(self.n_total, 1)
        return {
            "frac_random": self.n_random / n,
            "frac_guided_a": self.n_guided_a / n,
            "frac_guided_ab": self.n_guided_ab / n,
            "frac_guided_ac": self.n_guided_ac / n,
            "frac_ore_steps": self.n_ore_selected / max(self.n_guided_ab + self.n_guided_ac, 1),
            "frac_boundary_steps": self.n_boundary_selected / max(self.n_guided_ac, 1),
            "n_guided_fallback": float(self.n_guided_fallback),
        }

    def print_summary(self, prefix: str = "") -> None:
        f = self.fractions()
        print(
            f"{prefix}dataset stats:"
            f"  random={f['frac_random']:.2f}"
            f"  guided_A={f['frac_guided_a']:.2f}"
            f"  guided_AB={f['frac_guided_ab']:.2f}"
            f"  guided_AC={f['frac_guided_ac']:.2f}"
            f"  ore_steps={f['frac_ore_steps']:.2f}"
            f"  boundary_steps={f['frac_boundary_steps']:.2f}"
            f"  fallbacks={self.n_guided_fallback}"
        )


# ---------------------------------------------------------------------------
# Guide model loading
# ---------------------------------------------------------------------------


def _load_guide_model(
    guide_ckpt_path: str | Path,
    device: str,
) -> nn.Module:
    """Load a pretrained guide model that produces uncertainty estimates.

    Supports:
    * ``VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer``
      (forward returns ``(pred_ore, pred_uncertainty)``)
    * ``EndToEndMapBeliefTransformer``
      (forward returns ``pred_ore``; uncertainty falls back to heuristic downstream)

    The loaded model is placed on *device*, frozen (no grad), and set to eval().
    """
    from ...models.belief_models.end_to_end.variable_aware_patch_borehole_uncertainty_transformer import (
        VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer,
    )
    from ...models.belief_models.end_to_end.variable_aware_patch_borehole_transformer import (
        VariableAwarePatchBoreholeEndToEndConfig,
    )

    ckpt = torch.load(guide_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", None)

    # Try to detect model type from the checkpoint config class name
    cfg_class_name = type(cfg).__name__ if cfg is not None else ""

    if "Uncertainty" in cfg_class_name or "model_cfg" in ckpt:
        # Uncertainty transformer checkpoint
        if "model_cfg" in ckpt:
            model_cfg = ckpt["model_cfg"]
        elif hasattr(cfg, "to_model_config"):
            model_cfg = cfg.to_model_config()
        else:
            raise ValueError(
                "Cannot reconstruct model config from guide checkpoint. "
                "Expected 'model_cfg' key or config with to_model_config()."
            )
        model = VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer(
            model_cfg
        )
    else:
        # Fall back to EndToEndMapBeliefTransformer
        if "model_cfg" in ckpt:
            model_cfg = ckpt["model_cfg"]
        elif hasattr(cfg, "to_model_config"):
            model_cfg = cfg.to_model_config()
        else:
            raise ValueError(
                "Cannot reconstruct model config from guide checkpoint."
            )
        model = EndToEndMapBeliefTransformer(model_cfg)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Uncertainty-based cell selection
# ---------------------------------------------------------------------------


def _distance_heuristic_select(
    drilled_flat_idx: np.ndarray,
    undrilled_flat_idx: np.ndarray,
    n_x: int,
    n_y: int,
    rng: np.random.Generator,
) -> int:
    """Select undrilled cell maximising min-distance to any drilled cell.

    Used as the fallback when no guide model is available.
    """
    if drilled_flat_idx.size == 0:
        return int(rng.choice(undrilled_flat_idx))

    dr = drilled_flat_idx // n_y
    dc = drilled_flat_idx % n_y
    ur = undrilled_flat_idx // n_y
    uc = undrilled_flat_idx % n_y

    # Manhattan distance from each undrilled cell to its nearest drilled cell
    dists = np.abs(ur[:, None] - dr[None, :]) + np.abs(uc[:, None] - dc[None, :])
    min_dist = dists.min(axis=1)
    best = np.flatnonzero(min_dist == min_dist.max())
    return int(undrilled_flat_idx[rng.choice(best)])


def _model_uncertainty_select(
    guide_model: nn.Module,
    bh_arr: np.ndarray,
    drilled_flat_idx: np.ndarray,
    ore_vals_so_far: np.ndarray,
    grid_pos: np.ndarray,
    undrilled_flat_idx: np.ndarray,
    n_x: int,
    n_y: int,
    normalizer: TargetNormalizer,
    device: str,
    rng: np.random.Generator,
) -> int:
    """Select next drill location by highest predicted uncertainty.

    Parameters
    ----------
    guide_model     : loaded guide model (eval mode, frozen)
    bh_arr          : (n_x*n_y, V, D) standardised borehole array for this map
    drilled_flat_idx: 1-D array of already-drilled flat indices
    ore_vals_so_far : ore values at drilled locations (same order)
    grid_pos        : (n_x, n_y, 2) normalised positions
    undrilled_flat_idx: 1-D remaining flat indices
    normalizer      : TargetNormalizer (used for guide model inference normalisation)
    """
    K = drilled_flat_idx.size
    if K == 0:
        return int(rng.choice(undrilled_flat_idx))

    # Build a batch-1 input from current observations
    all_locs = [(i, j) for i in range(n_x) for j in range(n_y)]
    drill_locs = [all_locs[k] for k in drilled_flat_idx]

    bhs_k = np.stack([bh_arr[i * n_y + j] for i, j in drill_locs], axis=0)  # (K, V, D)
    pos_k = np.stack([grid_pos[i, j] for i, j in drill_locs], axis=0)        # (K, 2)

    bh_t = torch.from_numpy(bhs_k).unsqueeze(0).to(device)      # (1, K, V, D)
    ov_t = torch.from_numpy(ore_vals_so_far).unsqueeze(0).to(device)  # (1, K)
    pos_t = torch.from_numpy(pos_k).unsqueeze(0).to(device)      # (1, K, 2)

    with torch.no_grad():
        out = guide_model(bh_t, ov_t, pos_t)

    # Model may return (pred_ore, pred_uncertainty) or just pred_ore
    if isinstance(out, (tuple, list)):
        unc_map = out[1].squeeze().cpu().numpy()  # (n_x, n_y)
    else:
        # No uncertainty head — use absolute prediction error proxy is not available here.
        # Fall back to distance heuristic.
        return _distance_heuristic_select(
            drilled_flat_idx, undrilled_flat_idx, n_x, n_y, rng
        )

    # Pick argmax uncertainty among undrilled cells
    flat_unc = unc_map.ravel()
    best_local = int(np.argmax(flat_unc[undrilled_flat_idx]))
    return int(undrilled_flat_idx[best_local])


# ---------------------------------------------------------------------------
# Guided sequence builder
# ---------------------------------------------------------------------------


def _build_guided_sequence(
    bh_arr: np.ndarray,
    yield_target: np.ndarray,
    ore_features: OreFeatures,
    grid_pos: np.ndarray,
    all_idx: np.ndarray,
    rng: np.random.Generator,
    guide_model: nn.Module | None,
    normalizer: TargetNormalizer | None,
    device: str,
    cfg: GuidedExplorationConfig,
    n_drills: int,
    mode: str,
    stats: GuidedTrainingStats,
) -> tuple[np.ndarray, list[int]]:
    """Build a single guided drill sequence of length n_drills.

    Returns
    -------
    chosen_idx   : 1-D array of flat location indices in drill order
    guided_steps : list of step indices that were guided selections
    """
    n_x, n_y = yield_target.shape

    if not ore_features.has_ore:
        stats.n_guided_fallback += 1
        return rng.permutation(all_idx)[:n_drills].copy(), []

    undrilled = list(all_idx.copy())
    chosen_idx: list[int] = []
    guided_steps: list[int] = []

    drilled_flat: list[int] = []
    ore_vals_list: list[float] = []

    # Pre-compute available oracle pools as sets for fast intersection
    high_ore_set = set(ore_features.high_ore_idx.tolist())
    boundary_set = set(ore_features.boundary_idx.tolist())

    for step in range(n_drills):
        undrilled_arr = np.array(undrilled, dtype=np.int64)

        if step < cfg.phase1_drills or mode == "A" and guide_model is None and len(drilled_flat) == 0:
            # Phase 1: always random
            pick = int(rng.choice(undrilled_arr))
        else:
            # Phase 2: guided
            if mode == "A":
                # Uncertainty-guided every step
                if guide_model is not None:
                    pick = _model_uncertainty_select(
                        guide_model,
                        bh_arr,
                        np.array(drilled_flat, dtype=np.int64),
                        np.array(ore_vals_list, dtype=np.float32),
                        grid_pos,
                        undrilled_arr,
                        n_x, n_y,
                        normalizer,
                        device,
                        rng,
                    )
                else:
                    pick = _distance_heuristic_select(
                        np.array(drilled_flat, dtype=np.int64),
                        undrilled_arr,
                        n_x, n_y,
                        rng,
                    )
                guided_steps.append(step)

            elif mode == "AB":
                # Alternate: uncertainty / ore-assisted
                if (step - cfg.phase1_drills) % 2 == 0:
                    # Uncertainty step
                    if guide_model is not None:
                        pick = _model_uncertainty_select(
                            guide_model,
                            bh_arr,
                            np.array(drilled_flat, dtype=np.int64),
                            np.array(ore_vals_list, dtype=np.float32),
                            grid_pos,
                            undrilled_arr,
                            n_x, n_y,
                            normalizer,
                            device,
                            rng,
                        )
                    else:
                        pick = _distance_heuristic_select(
                            np.array(drilled_flat, dtype=np.int64),
                            undrilled_arr,
                            n_x, n_y,
                            rng,
                        )
                    guided_steps.append(step)
                else:
                    # Ore-assisted step
                    candidates = np.array(
                        [i for i in undrilled if i in high_ore_set], dtype=np.int64
                    )
                    if candidates.size > 0:
                        pick = int(rng.choice(candidates))
                        stats.n_ore_selected += 1
                        guided_steps.append(step)
                    else:
                        pick = int(rng.choice(undrilled_arr))

            elif mode == "AC":
                # Alternate: uncertainty / boundary-assisted
                if (step - cfg.phase1_drills) % 2 == 0:
                    if guide_model is not None:
                        pick = _model_uncertainty_select(
                            guide_model,
                            bh_arr,
                            np.array(drilled_flat, dtype=np.int64),
                            np.array(ore_vals_list, dtype=np.float32),
                            grid_pos,
                            undrilled_arr,
                            n_x, n_y,
                            normalizer,
                            device,
                            rng,
                        )
                    else:
                        pick = _distance_heuristic_select(
                            np.array(drilled_flat, dtype=np.int64),
                            undrilled_arr,
                            n_x, n_y,
                            rng,
                        )
                    guided_steps.append(step)
                else:
                    candidates = np.array(
                        [i for i in undrilled if i in boundary_set], dtype=np.int64
                    )
                    if candidates.size > 0:
                        pick = int(rng.choice(candidates))
                        stats.n_boundary_selected += 1
                        guided_steps.append(step)
                    else:
                        pick = int(rng.choice(undrilled_arr))
            else:
                pick = int(rng.choice(undrilled_arr))

        chosen_idx.append(pick)
        undrilled.remove(pick)
        drilled_flat.append(pick)

        # Record the ore value at this location for guide model input on next step
        all_locs_list = [(i, j) for i in range(n_x) for j in range(n_y)]
        loc = all_locs_list[pick]
        ore_vals_list.append(float(yield_target[loc[0], loc[1]]))

    return np.array(chosen_idx, dtype=np.int64), guided_steps


# ---------------------------------------------------------------------------
# Guided dataset
# ---------------------------------------------------------------------------


class GuidedE2EMapDataset(E2EMapDataset):
    """Dataset with a 75/25 random/guided mixture of drilling sequences.

    Each sample has the same model-facing fields as E2EMapDataset.
    Extra bookkeeping fields (``guided``, ``guided_mode``, ``guided_steps``)
    are stored for logging and visualisation but never passed to the model.
    """

    @classmethod
    def from_cache_guided(
        cls,
        cache: NpzMap,
        resources: DecisionSimulationResources,
        cfg: GuidedExplorationConfig,
        guide_model: nn.Module | None,
        normalizer: TargetNormalizer | None,
        device: str,
        verbose: bool = True,
        is_val: bool = False,
    ) -> "GuidedE2EMapDataset":
        """Build the guided curriculum dataset.

        For validation sets guided drilling is disabled (100 % random) so that
        validation metrics are directly comparable to the baseline training run.
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
        stats = GuidedTrainingStats()

        # Mode probabilities for guided samples
        mode_choices = ["A", "AB", "AC"]
        mode_probs = np.array([cfg.p_mode_a, cfg.p_mode_ab, cfg.p_mode_ac])

        for map_idx in range(cache.pool_size):
            bh_arr = cache.borehole_arrays[map_idx].copy()
            if resources.norm_stats:
                bh_arr = standardise(bh_arr, resources.norm_stats, variables)
            bh_arr = np.nan_to_num(bh_arr, nan=0.0).astype(np.float32)

            target_ore = cache.targets[map_idx]  # (n_x, n_y)

            ore_features = _derive_ore_features(
                target_ore, cfg.ore_top_pct, cfg.boundary_ore_pct
            )

            n_guided = 0 if is_val else round(spm * cfg.p_guided)
            n_random = spm - n_guided

            def _append(
                chosen_idx: np.ndarray,
                guided: bool = False,
                guided_mode: str = "",
                guided_steps: list[int] | None = None,
                sequence_id: int = 0,
            ) -> None:
                drill_locs = [all_locations[k] for k in chosen_idx]
                K = len(drill_locs)
                drill_bhs = np.stack(
                    [bh_arr[i * n_y + j] for i, j in drill_locs], axis=0
                )
                ore_vals_k = np.array(
                    [target_ore[i, j] for i, j in drill_locs], dtype=np.float32
                )
                positions_k = np.stack(
                    [grid_pos[i, j] for i, j in drill_locs], axis=0
                )
                samples.append(
                    {
                        "boreholes": drill_bhs,
                        "ore_vals": ore_vals_k,
                        "positions": positions_k,
                        "target_map": target_ore.copy(),
                        "drill_count": K,
                        "map_idx": map_idx,
                        "sequence_id": sequence_id,
                        # bookkeeping (not seen by model)
                        "guided": guided,
                        "guided_mode": guided_mode,
                        "guided_steps": guided_steps or [],
                    }
                )

            if cfg.use_sequential_dataset:
                # Sequential mode: generate prefix slices.
                # Guided variants replace random permutations for some sequences.
                n_seqs = cfg.n_sequences_per_map
                n_guided_seqs = 0 if is_val else round(n_seqs * cfg.p_guided)
                n_random_seqs = n_seqs - n_guided_seqs

                for seq_id in range(n_random_seqs):
                    sequence = rng.permutation(all_idx)
                    for step in cfg.prefix_steps:
                        if step >= len(all_locations):
                            continue
                        _append(sequence[:step], sequence_id=seq_id)
                    stats.n_random += 1

                for seq_id in range(n_random_seqs, n_seqs):
                    mode = str(rng.choice(mode_choices, p=mode_probs))
                    n_drills_max = max(cfg.prefix_steps)
                    # Build full sequence up to max step, then slice for each prefix
                    full_idx, guided_steps = _build_guided_sequence(
                        bh_arr, target_ore, ore_features, grid_pos, all_idx, rng,
                        guide_model, normalizer, device, cfg, n_drills_max, mode, stats,
                    )
                    for step in cfg.prefix_steps:
                        if step > len(full_idx):
                            continue
                        guided_at_step = [s for s in guided_steps if s < step]
                        _append(
                            full_idx[:step],
                            guided=True,
                            guided_mode=mode,
                            guided_steps=guided_at_step,
                            sequence_id=seq_id,
                        )
                    if mode == "A":
                        stats.n_guided_a += 1
                    elif mode == "AB":
                        stats.n_guided_ab += 1
                    else:
                        stats.n_guided_ac += 1

            else:
                for _ in range(n_random):
                    n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                    chosen = rng.choice(all_idx, size=n_drills, replace=False)
                    _append(chosen)
                    stats.n_random += 1

                for _ in range(n_guided):
                    mode = str(rng.choice(mode_choices, p=mode_probs))
                    n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                    chosen, guided_steps = _build_guided_sequence(
                        bh_arr, target_ore, ore_features, grid_pos, all_idx, rng,
                        guide_model, normalizer, device, cfg, n_drills, mode, stats,
                    )
                    _append(chosen, guided=True, guided_mode=mode, guided_steps=guided_steps)
                    if mode == "A":
                        stats.n_guided_a += 1
                    elif mode == "AB":
                        stats.n_guided_ab += 1
                    else:
                        stats.n_guided_ac += 1

            if verbose and (map_idx + 1) % 10 == 0:
                print(
                    f"  [guided dataset] {map_idx + 1}/{cache.pool_size} maps processed"
                )

        # Shuffle so random and guided samples are interleaved
        order = rng.permutation(len(samples))
        samples = [samples[i] for i in order]

        ds = cls(samples)
        ds._stats = stats  # type: ignore[attr-defined]
        return ds

    @property
    def stats(self) -> GuidedTrainingStats:
        return getattr(self, "_stats", GuidedTrainingStats())


# ---------------------------------------------------------------------------
# Trajectory visualisation
# ---------------------------------------------------------------------------


def _save_guided_trajectory_plots(
    train_ds: GuidedE2EMapDataset,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    n_viz_maps: int = 5,
) -> None:
    """Save trajectory PNGs showing random vs guided drill selections per map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(plot_dir) / f"guided_trajectories_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect one sample per map_idx (prefer guided samples with most guided steps)
    by_map: dict[int, dict] = {}
    for s in train_ds.samples:
        mid = int(s["map_idx"])
        if mid not in by_map:
            by_map[mid] = s
        elif s.get("guided", False) and len(s.get("guided_steps", [])) > len(
            by_map[mid].get("guided_steps", [])
        ):
            by_map[mid] = s

    selected = list(by_map.values())
    indices = np.linspace(0, len(selected) - 1, min(n_viz_maps, len(selected)), dtype=int)

    for k, idx in enumerate(indices):
        sample = selected[int(idx)]
        true_ore = normalizer.inverse(sample["target_map"])
        n_x, n_y = true_ore.shape

        positions = sample["positions"]  # (K, 2) normalised
        guided_steps_set = set(sample.get("guided_steps", []))
        mode = sample.get("guided_mode", "")

        rows = np.array([int(round(float(p[0]) * (n_x - 1))) for p in positions])
        cols = np.array([int(round(float(p[1]) * (n_y - 1))) for p in positions])

        rand_mask = np.array([i not in guided_steps_set for i in range(len(rows))])
        guided_mask = ~rand_mask

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        vmax = max(float(true_ore.max()), 1e-3)
        kw = dict(origin="lower", cmap="viridis", vmin=0, vmax=vmax)

        for ax, title in zip(axes, ["True ore map", "Drilling sequence"]):
            im = ax.imshow(true_ore.T, **kw)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[1]
        if rand_mask.any():
            ax.scatter(
                rows[rand_mask], cols[rand_mask],
                c="cyan", s=20, marker="o", label="random", zorder=3,
                linewidths=0.5, edgecolors="white",
            )
        if guided_mask.any():
            ax.scatter(
                rows[guided_mask], cols[guided_mask],
                c="red", s=30, marker="*", label=f"guided ({mode})", zorder=4,
            )
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=axes, shrink=0.7, label="ore value")
        fig.suptitle(
            f"Map {sample['map_idx']}  |  {len(rows)} drills"
            f"  |  {guided_mask.sum()} guided  |  Mode {mode or 'random'}",
            fontsize=9,
        )
        fig.savefig(out_dir / f"traj_{k:02d}_map{sample['map_idx']}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  trajectory plots -> {out_dir}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_guided_exploration_belief(
    resources: DecisionSimulationResources,
    cfg: GuidedExplorationConfig,
    device: str,
    checkpoint_dir: Path,
    train_cache: NpzMap,
    val_cache: NpzMap,
    plot_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[EndToEndMapBeliefTransformer, TargetNormalizer]:
    """Train EndToEndMapBeliefTransformer with a guided-exploration curriculum.

    Saves two checkpoints:
      guided_belief_best.pt  — lowest validation MSE (ore-value space)
      guided_belief_last.pt  — final epoch

    Parameters
    ----------
    resources       : shared resources (norm_stats for borehole standardisation)
    cfg             : guided-curriculum training configuration
    device          : torch device string
    checkpoint_dir  : directory for saved checkpoints
    train_cache     : NpzMap for training maps (pre-loaded)
    val_cache       : NpzMap for validation maps (pre-loaded)
    plot_dir        : if given, save validation + trajectory plots here
    verbose         : print per-epoch metrics
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if resources.variable_names:
        cfg.n_variables = len(resources.variable_names)

    # ---- guide model --------------------------------------------------------
    guide_model: nn.Module | None = None
    if cfg.guide_ckpt_path is not None:
        if verbose:
            print(f"  loading guide model from {cfg.guide_ckpt_path}")
        guide_model = _load_guide_model(cfg.guide_ckpt_path, device)
        if verbose:
            print("  guide model loaded (static)")

    # ---- initial dataset build ----------------------------------------------
    if verbose:
        print("  building training dataset ...")
    train_ds = GuidedE2EMapDataset.from_cache_guided(
        train_cache, resources, cfg,
        guide_model=guide_model, normalizer=None,
        device=device, verbose=verbose, is_val=False,
    )
    if verbose:
        train_ds.stats.print_summary("  ")

    if verbose:
        print("  building validation dataset ...")
    val_ds = GuidedE2EMapDataset.from_cache_guided(
        val_cache, resources, cfg,
        guide_model=guide_model, normalizer=None,
        device=device, verbose=verbose, is_val=True,
    )

    if len(train_ds) > 0:
        cfg.n_depth = train_ds.samples[0]["boreholes"].shape[2]
        sample_map = train_ds.samples[0]["target_map"]
        cfg.n_x, cfg.n_y = int(sample_map.shape[0]), int(sample_map.shape[1])

    # ---- target normalisation -----------------------------------------------
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

    # ---- data loaders -------------------------------------------------------
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_e2e_map,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_e2e_map,
    )

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model --------------------------------------------------------------
    model_cfg = cfg.to_model_config()
    model = EndToEndMapBeliefTransformer(model_cfg).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params  : {n_params:,}")
        print(f"  guided frac   : {cfg.p_guided:.0%}")
        print(f"  mode split    : A={cfg.p_mode_a:.0%}  AB={cfg.p_mode_ab:.0%}  AC={cfg.p_mode_ac:.0%}")
        print(f"  phase1 drills : {cfg.phase1_drills}")
        print(f"  guide model   : {'loaded' if guide_model else 'heuristic (distance)'}")

    # ---- training loop ------------------------------------------------------
    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch = 0

    for epoch in range(1, cfg.n_epochs + 1):

        # --- periodic guided dataset refresh (lagged self-update) ------------
        if (
            cfg.guide_update_interval > 0
            and cfg.guide_ckpt_path is None
            and epoch > 1
            and (epoch - 1) % cfg.guide_update_interval == 0
        ):
            if verbose:
                print(f"  [epoch {epoch}] refreshing guided sequences from current model ...")
            # Use current model as its own guide (lagged snapshot)
            snapshot = EndToEndMapBeliefTransformer(model_cfg).to(device)
            snapshot.load_state_dict(model.state_dict())
            snapshot.eval()
            for p in snapshot.parameters():
                p.requires_grad_(False)

            refreshed = GuidedE2EMapDataset.from_cache_guided(
                train_cache, resources, cfg,
                guide_model=snapshot, normalizer=normalizer,
                device=device, verbose=False, is_val=False,
            )
            refreshed.apply_target_normalizer(normalizer)
            train_ds = refreshed
            train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_e2e_map,
            )
            if verbose:
                train_ds.stats.print_summary("    refreshed ")

        # --- training step ---------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred = model(bh, ov, pos, pm)
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

        # Guided-fraction stats for this epoch
        ds_stats = train_ds.stats
        frac = ds_stats.fractions()
        row = {
            "epoch": epoch,
            "train_mse_norm": train_mse_norm,
            **val_metrics,
            "frac_random": frac["frac_random"],
            "frac_guided_a": frac["frac_guided_a"],
            "frac_guided_ab": frac["frac_guided_ab"],
            "frac_guided_ac": frac["frac_guided_ac"],
        }
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{cfg.n_epochs}"
                f"  train={train_mse_norm:.4f}"
                f"  val_mse={val_metrics['val_mse']:.4f}"
                f"  val_corr={val_metrics['val_corr']:.4f}"
                f"  guided={frac['frac_guided_a'] + frac['frac_guided_ab'] + frac['frac_guided_ac']:.2f}"
            )

        if val_metrics["val_mse"] < best_val_mse - cfg.min_delta:
            best_val_mse = val_metrics["val_mse"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint_model(
                checkpoint_dir / "guided_belief_best.pt",
                model, cfg, epoch, history, normalizer,
                model_cfg=model_cfg, n_x=cfg.n_x, n_y=cfg.n_y, latent_dim=cfg.latent_dim,
            )
        else:
            patience_counter += 1

        if cfg.early_stopping and patience_counter >= cfg.patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best val MSE: {best_val_mse:.4f} at epoch {best_epoch}."
                )
            break

    model_validation(
        model, checkpoint_dir, "guided_belief_best.pt", history,
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

    # ---- final dataset stats ------------------------------------------------
    stats_path = checkpoint_dir / "guided_dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(train_ds.stats.fractions(), f, indent=2)
    if verbose:
        train_ds.stats.print_summary("  final ")

    if plot_dir is not None:
        _save_guided_trajectory_plots(train_ds, normalizer, plot_dir, n_viz_maps=cfg.n_viz_maps)

    return model, normalizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_guided_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[EndToEndMapBeliefTransformer, GuidedExplorationConfig, TargetNormalizer, list[dict]]:
    """Load a checkpoint saved by ``train_guided_exploration_belief``."""

    def _model_fn(ckpt: dict) -> EndToEndMapBeliefTransformer:
        if "model_cfg" in ckpt:
            return EndToEndMapBeliefTransformer(ckpt["model_cfg"])
        return EndToEndMapBeliefTransformer(ckpt["cfg"].to_model_config())

    return load_model_encoder_checkpoint(
        path, _model_fn, GuidedExplorationConfig, device
    )
