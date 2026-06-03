from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ..training_utils import (
    TargetNormalizer,
    make_coordinate_grid,
)


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
