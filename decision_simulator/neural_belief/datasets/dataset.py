from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from simulator.map_generator import MapGenerator, SimConfig
from ..models.borehole_encoders.autoencoder import standardise
from decision_simulator.resources import DecisionSimulationResources
from ..utils import (
    LatentPCAReducer,
    TargetNormalizer,
    build_ore_target,
    encode_full_latent_map,
    build_sample_input,
    make_coordinate_grid,
)

if TYPE_CHECKING:
    from ..map_cache import NpzMapCache


@dataclass
class BeliefDatasetConfig:
    """Configuration for synthetic belief-state dataset generation."""

    n_maps: int = 50
    samples_per_map: int = 20
    min_drills: int = 1
    max_drills: int = 15
    seed: int = 42


class GeologicalBeliefDataset(Dataset):
    """Pre-generated (partial observation → full ore map) pairs.

    Each sample
    -----------
    input  : (2 + latent_dim, n_x, n_y) float32 tensor
               ch 0   : sparse observed ore map
               ch 1   : binary observation mask
               ch 2.. : JEPA latent vectors at drilled cells
    target : (1, n_x, n_y) float32 tensor
               ch 0   : true max-pooled ore map (yield_field.max(depth))
               NOTE   : may be in normalized space after apply_target_normalizer()
    drill_counts : (N,) int64 tensor, number of drills per sample (optional)
    metadata : list of per-sample dicts (optional); used by sequential evaluation
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        drill_counts: torch.Tensor | None = None,
        metadata: list[dict] | None = None,
    ) -> None:
        self.inputs = inputs
        self.targets = targets
        self.drill_counts = drill_counts
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    def apply_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Normalize stored targets in-place using a fitted TargetNormalizer."""
        targets_np = normalizer.transform(self.targets.numpy())
        self.targets = torch.from_numpy(targets_np)

    def apply_latent_pca(self, reducer: LatentPCAReducer) -> None:
        """Apply PCA dimensionality reduction to latent input channels in-place.

        Replaces channels ``[2:]`` with their PCA projections, changing the
        tensor from ``(N, 2+original_dim, n_x, n_y)`` to
        ``(N, 2+n_components, n_x, n_y)``.  Unobserved cells are re-zeroed
        after projection (PCA mean subtraction would otherwise give non-zero
        values at zero-filled positions).
        """
        if self.inputs.shape[1] <= 2:
            return  # no-encoder mode: nothing to reduce
        inputs_np = self.inputs.numpy()  # (N, 2+D, n_x, n_y)
        N, _, n_x, n_y = inputs_np.shape
        ore = inputs_np[:, 0:1, :, :]  # (N, 1, n_x, n_y)
        mask = inputs_np[:, 1:2, :, :]  # (N, 1, n_x, n_y)
        latent_ch = inputs_np[:, 2:, :, :]  # (N, D, n_x, n_y)

        D = latent_ch.shape[1]
        lat_flat = latent_ch.transpose(0, 2, 3, 1).reshape(-1, D)  # (N*n_x*n_y, D)
        reduced = reducer.transform(lat_flat)  # (N*n_x*n_y, K)
        K = reduced.shape[1]

        reduced = reduced.reshape(N, n_x, n_y, K).transpose(
            0, 3, 1, 2
        )  # (N, K, n_x, n_y)
        reduced *= mask > 0.0  # re-zero unobserved cells

        self.inputs = torch.from_numpy(np.concatenate([ore, mask, reduced], axis=1))

    def apply_coordinate_channels(self) -> None:
        """Append normalized x and y coordinate channels to every sample in-place.

        The two new channels are appended after all existing channels so the
        final layout becomes ``[ore, mask, latents..., x_coord, y_coord]``.
        Both grids are in ``[0, 1]``: x varies along the row axis, y along
        the column axis.
        """
        N, _, n_x, n_y = self.inputs.shape
        coord = torch.from_numpy(make_coordinate_grid(n_x, n_y))  # (2, n_x, n_y)
        coord = coord.unsqueeze(0).expand(N, -1, -1, -1)  # (N, 2, n_x, n_y)
        self.inputs = torch.cat([self.inputs, coord], dim=1)


def build_dataset_from_cache(
    cache: "NpzMapCache",
    resources: DecisionSimulationResources,
    device: str,
    verbose: bool = False,
    samples_per_map: int | None = None,
    shuffle_latents: bool = False,
    shuffle_seed: int = 0,
) -> GeologicalBeliefDataset:
    """Build an encoder-specific dataset from a shared :class:`NpzMapCache`.

    Standardises and encodes boreholes once per map using the encoder in
    ``resources``.  For the no-encoder case (both ``resources.jepa_model``
    and ``borehole_encoder_fn`` are ``None``), inputs will be ``(2, n_x, n_y)``.
    Targets are returned in raw (unnormalized) ore-value space.

    Parameters
    ----------
    shuffle_latents
        If True, randomly permute the spatial positions of the full latent map
        before building sample inputs.  Ore observations and drill locations are
        unchanged — only the latent-channel content is spatially scrambled.
        Used to create shuffled-latent control variants.
    shuffle_seed
        Base seed for the per-map shuffle RNG.  Map ``i`` uses seed
        ``shuffle_seed + i`` so each map gets a different but reproducible
        permutation.
    """
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    drill_count_list: list[int] = []

    for map_idx, (bh_arr, target_ore, samples) in enumerate(
        zip(cache.borehole_arrays, cache.targets, cache.drill_patterns)
    ):
        n_x, n_y = cache.n_x, cache.n_y

        # Standardise raw boreholes with this encoder's norm stats
        bh_std = bh_arr.astype(np.float32)
        if resources.norm_stats and resources.variable_names:
            bh_std = standardise(bh_std, resources.norm_stats, resources.variable_names)
            bh_std = np.nan_to_num(bh_std, nan=0.0)

        full_latent_map = encode_full_latent_map(bh_std, n_x, n_y, resources, device)

        # Shuffle spatial positions of latent vectors (control variant).
        # Latent_dim==0 means no encoder — nothing to shuffle.
        if shuffle_latents and full_latent_map.shape[2] > 0:
            flat = full_latent_map.reshape(-1, full_latent_map.shape[2])
            np.random.default_rng(shuffle_seed + map_idx).shuffle(flat)
            full_latent_map = flat.reshape(n_x, n_y, -1)

        used_samples = (
            samples[:samples_per_map] if samples_per_map is not None else samples
        )
        for drill_locs, ore_vals in used_samples:
            inp = build_sample_input(drill_locs, ore_vals, full_latent_map)
            all_inputs.append(inp)
            all_targets.append(target_ore[np.newaxis])  # (1, n_x, n_y)
            drill_count_list.append(len(drill_locs))

        if verbose and (map_idx + 1) % 10 == 0:
            print(f"  [encode] {map_idx + 1}/{len(cache.borehole_arrays)} maps encoded")

    inputs_t = torch.from_numpy(np.stack(all_inputs, axis=0))
    targets_t = torch.from_numpy(np.stack(all_targets, axis=0))
    drill_counts_t = torch.tensor(drill_count_list, dtype=torch.long)
    return GeologicalBeliefDataset(inputs_t, targets_t, drill_counts=drill_counts_t)
