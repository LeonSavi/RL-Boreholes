"""In-memory map container for neural-belief training.

The :class:`NpzMap` dataclass holds a loaded batch of maps and provides
encoding and dataset-construction helpers consumed by the training pipeline.
Maps are loaded from HDF5 shards via :mod:`map_hdf5`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .datasets import BeliefDatasetConfig, GeologicalBeliefDataset
from .models import standardise
from decision_simulator.resources import DecisionSimulationResources

_DEFAULT_PREFIX_STEPS = [1, 2, 3, 5, 8, 10, 15]


# ---------------------------------------------------------------------------
# Layer 1: In-memory container
# ---------------------------------------------------------------------------


@dataclass
class NpzMap:
    """In-memory container for map pool data loaded from an npz cache.

    Created by :meth:`HDF5MapStore.load_subset` or :meth:`HDF5MapDirectory.load_map_data`.
    """

    borehole_arrays: list[np.ndarray]  # each (n_boreholes, V, D) float32, raw
    targets: list[np.ndarray]  # each (n_x, n_y) float32
    drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]]
    cfg: BeliefDatasetConfig
    n_x: int
    n_y: int
    rocks_arrays: list[np.ndarray] | None = None  # each (n_boreholes, D) int8, or None if unavailable

    @property
    def pool_size(self) -> int:
        return len(self.borehole_arrays)

    def encode_full_latent_map(
        self,
        borehole_array: np.ndarray,
        resources: DecisionSimulationResources,
        device: str,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode every borehole in a map in one batched pass.

        Parameters
        ----------
        borehole_array
            Pre-computed borehole tensor of shape ``(n_x * n_y, V, D)`` float32.
            Must already be standardised and nan-zeroed before calling.
        resources
            Shared experiment resources; encoder is resolved as:
            1. ``resources.borehole_encoder_fn`` if set
            2. ``resources.jepa_model.embed`` as fallback
            3. Neither set → returns empty ``(n_x, n_y, 0)`` array (no-encoder mode)
        device
            Torch device string.
        batch_size
            Boreholes encoded per forward pass.

        Returns
        -------
        np.ndarray of shape ``(n_x, n_y, latent_dim)`` float32
        """
        encoder_fn = getattr(resources, "borehole_encoder_fn", None)
        if encoder_fn is None:
            jepa = getattr(resources, "jepa_model", None)
            if jepa is not None:
                encoder_fn = jepa.embed

        if encoder_fn is None:
            return np.zeros((self.n_x, self.n_y, 0), dtype=np.float32)

        bh_tensor = torch.from_numpy(borehole_array.astype(np.float32)).to(device)

        chunks: list[np.ndarray] = []
        for start in range(0, bh_tensor.shape[0], batch_size):
            with torch.no_grad():
                lat = encoder_fn(bh_tensor[start : start + batch_size])
            chunks.append(lat.cpu().numpy())

        latents = np.concatenate(chunks, axis=0)  # (n_x*n_y, latent_dim)
        return latents.reshape(self.n_x, self.n_y, -1).astype(np.float32)

    @staticmethod
    def build_sample_input(
        drill_locations: list[tuple[int, int]],
        ore_values: list[float],
        full_latent_map: np.ndarray,  # (n_x, n_y, latent_dim)
    ) -> np.ndarray:
        """Construct the ``(2 + latent_dim, n_x, n_y)`` model input from a drill pattern.

        Channel layout
        --------------
        [0]       : sparse observed ore map  (0 at unobserved cells)
        [1]       : binary observation mask  (1 = observed, 0 = unobserved)
        [2 ...]   : JEPA latent vectors      (zero vector at unobserved cells)
        """
        n_x, n_y, latent_dim = full_latent_map.shape

        sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
        mask = np.zeros((n_x, n_y), dtype=np.float32)
        jepa_map = np.zeros((n_x, n_y, latent_dim), dtype=np.float32)

        for (i, j), ore_val in zip(drill_locations, ore_values):
            sparse_ore[i, j] = float(ore_val)
            mask[i, j] = 1.0
            jepa_map[i, j] = full_latent_map[i, j]

        return np.concatenate(
            [
                sparse_ore[np.newaxis],          # (1, n_x, n_y)
                mask[np.newaxis],                # (1, n_x, n_y)
                jepa_map.transpose(2, 0, 1),     # (latent_dim, n_x, n_y)
            ],
            axis=0,
        )  # (2 + latent_dim, n_x, n_y)

    def build_geo_train_maps(
        self,
        resources: DecisionSimulationResources,
        device: str,
        n_sequences_per_map: int = 1,
        max_drills: int = 15,
        prefix_steps: list[int] | None = None,
        seed: int = 42,
        verbose: bool = False,
        shuffle_latents: bool = False,
        shuffle_seed: int = 0,
    ) -> GeologicalBeliefDataset:
        """Build a sequential prefix dataset from this map cache.

        For each map, generates ``n_sequences_per_map`` random drill orderings.
        Each ordering produces one sample per checkpoint in ``prefix_steps``,
        where step *k* reveals the first *k* drills of the sequence.

        All prefixes from the same map are always in the same train/val split
        (guaranteed by the map-level split in :meth:`HDF5MapDirectory.load_map_data`).

        Parameters
        ----------
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
            Spatially scramble latent vectors (control variant).
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

        n_x, n_y = self.n_x, self.n_y
        all_locs = [(i, j) for i in range(n_x) for j in range(n_y)]

        all_inputs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        metadata_list: list[dict] = []
        drill_count_list: list[int] = []

        for map_idx, (bh_arr, target_ore) in enumerate(
            zip(self.borehole_arrays, self.targets)
        ):
            bh_std = bh_arr.astype(np.float32)
            if resources.norm_stats and resources.variable_names:
                bh_std = standardise(
                    bh_std, resources.norm_stats, resources.variable_names
                )
                bh_std = np.nan_to_num(bh_std, nan=0.0)

            full_latent_map = self.encode_full_latent_map(bh_std, resources, device)

            if shuffle_latents and full_latent_map.shape[2] > 0:
                flat = full_latent_map.reshape(-1, full_latent_map.shape[2])
                np.random.default_rng(shuffle_seed + map_idx).shuffle(flat)
                full_latent_map = flat.reshape(n_x, n_y, -1)

            for seq_id in range(n_sequences_per_map):
                seq_seed = seed + map_idx * n_sequences_per_map + seq_id
                perm = np.random.default_rng(seq_seed).permutation(len(all_locs))
                sequence_locs = [all_locs[k] for k in perm[:max_drills]]

                for step in prefix_steps:
                    actual_step = min(step, len(sequence_locs))
                    drill_locs = sequence_locs[:actual_step]
                    ore_vals = [float(target_ore[i, j]) for i, j in drill_locs]

                    inp = NpzMap.build_sample_input(drill_locs, ore_vals, full_latent_map)
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
                    f"  [encode] {map_idx + 1}/{len(self.borehole_arrays)} maps encoded"
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



