"""Sequential drill-prefix dataset generation.

Converts a map cache into ordered borehole sequences with prefix samples,
enabling the belief model to learn from progressively revealed observations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from encoder.autoencoder import standardise
from decision_simulator.resources import DecisionSimulationResources
from ..utils import encode_full_latent_map, build_sample_input
from .dataset import GeologicalBeliefDataset

if TYPE_CHECKING:
    from ..map_cache import NpzMapCache

_DEFAULT_PREFIX_STEPS = [1, 2, 3, 5, 8, 10, 15]


def build_sequential_dataset_from_cache(
    cache: "NpzMapCache",
    resources: DecisionSimulationResources,
    device: str,
    n_sequences_per_map: int = 3,
    max_drills: int = 15,
    prefix_steps: list[int] | None = None,
    seed: int = 42,
    verbose: bool = False,
    shuffle_latents: bool = False,
    shuffle_seed: int = 0,
) -> GeologicalBeliefDataset:
    """Build a sequential prefix dataset from a pre-generated map cache.

    For each map, generates ``n_sequences_per_map`` random drill orderings.
    Each ordering produces one sample per checkpoint in ``prefix_steps``,
    where step *k* reveals the first *k* drills of the sequence.

    All prefixes from the same map are always in the same train/val split
    (guaranteed by the map-level split in ``NpzMapCacheStore``).

    Parameters
    ----------
    cache
        Pre-loaded map cache (train or val slice from ``NpzMapCacheStore``).
    resources
        Encoder model + normalisation stats.
    device
        Torch device for encoder forward passes.
    n_sequences_per_map
        Number of independent drill orderings per map.  More orderings reduce
        memorisation of a single exploration path.
    max_drills
        Length of each generated sequence.  ``prefix_steps`` values above this
        are clamped to ``max_drills``.
    prefix_steps
        Drill-count checkpoints at which to create samples.
        Defaults to ``[1, 2, 3, 5, 8, 10, 15]``.
    seed
        Master RNG seed.  Each (map, sequence) pair gets a deterministic
        sub-seed so the dataset is fully reproducible.
    verbose
        Print progress every 10 maps.
    shuffle_latents
        Spatially scramble latent vectors (control variant, same as
        ``build_dataset_from_cache``).
    shuffle_seed
        Base seed for per-map latent shuffle.

    Returns
    -------
    GeologicalBeliefDataset
        Dataset with ``metadata`` list containing ``map_id``, ``sequence_id``,
        ``step``, and ``n_drills`` per sample.
    """
    if prefix_steps is None:
        prefix_steps = _DEFAULT_PREFIX_STEPS

    n_x, n_y = cache.n_x, cache.n_y
    all_locs = [(i, j) for i in range(n_x) for j in range(n_y)]

    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    metadata_list: list[dict] = []
    drill_count_list: list[int] = []

    for map_idx, (bh_arr, target_ore) in enumerate(
        zip(cache.borehole_arrays, cache.targets)
    ):
        # Standardise raw boreholes (identical to build_dataset_from_cache)
        bh_std = bh_arr.astype(np.float32)
        if resources.norm_stats and resources.variable_names:
            bh_std = standardise(bh_std, resources.norm_stats, resources.variable_names)
            bh_std = np.nan_to_num(bh_std, nan=0.0)

        full_latent_map = encode_full_latent_map(bh_std, n_x, n_y, resources, device)

        if shuffle_latents and full_latent_map.shape[2] > 0:
            flat = full_latent_map.reshape(-1, full_latent_map.shape[2])
            np.random.default_rng(shuffle_seed + map_idx).shuffle(flat)
            full_latent_map = flat.reshape(n_x, n_y, -1)

        for seq_id in range(n_sequences_per_map):
            # Deterministic sub-seed per (map, sequence) pair
            seq_seed = seed + map_idx * n_sequences_per_map + seq_id
            perm = np.random.default_rng(seq_seed).permutation(len(all_locs))
            sequence_locs = [all_locs[k] for k in perm[:max_drills]]

            for step in prefix_steps:
                actual_step = min(step, len(sequence_locs))
                drill_locs = sequence_locs[:actual_step]
                ore_vals = [float(target_ore[i, j]) for i, j in drill_locs]

                inp = build_sample_input(drill_locs, ore_vals, full_latent_map)
                all_inputs.append(inp)
                all_targets.append(target_ore[np.newaxis])  # (1, n_x, n_y)
                metadata_list.append(
                    {
                        "map_id": map_idx,
                        "sequence_id": seq_id,
                        "step": actual_step,
                        "n_drills": actual_step,
                    }
                )
                drill_count_list.append(actual_step)

        if verbose and (map_idx + 1) % 10 == 0:
            print(
                f"  [sequential] {map_idx + 1}/{len(cache.borehole_arrays)} maps encoded"
            )

    inputs_t = torch.from_numpy(np.stack(all_inputs))
    targets_t = torch.from_numpy(np.stack(all_targets))
    drill_counts_t = torch.tensor(drill_count_list, dtype=torch.long)
    return GeologicalBeliefDataset(
        inputs_t,
        targets_t,
        drill_counts=drill_counts_t,
        metadata=metadata_list,
    )
