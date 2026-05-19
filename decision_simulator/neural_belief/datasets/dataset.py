from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from simulator.map_generator import MapGenerator, SimConfig
from encoder.autoencoder import standardise
from decision_simulator.resources import DecisionSimulationResources
from ..utils import LatentPCAReducer, TargetNormalizer, build_ore_target, encode_full_latent_map, build_sample_input, make_coordinate_grid

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


def _validate_sample(
    inp: np.ndarray,
    target: np.ndarray,
    drill_locs: list[tuple[int, int]],
    n_x: int,
    n_y: int,
) -> None:
    """Assert shape and content invariants for one generated sample."""
    assert inp.ndim == 3 and inp.shape[1:] == (n_x, n_y), (
        f"input spatial shape {inp.shape[1:]} != ({n_x}, {n_y})"
    )
    assert target.shape == (1, n_x, n_y), (
        f"target shape {target.shape} != (1, {n_x}, {n_y})"
    )

    mask = inp[1]
    assert np.all((mask == 0.0) | (mask == 1.0)), (
        "observation mask contains non-binary values"
    )

    drilled = {(i, j) for i, j in drill_locs}
    unobserved = np.array(
        [[(i, j) not in drilled for j in range(n_y)] for i in range(n_x)],
        dtype=bool,
    )
    latents = inp[2:]
    assert np.all(latents[:, unobserved] == 0.0), (
        "latent channels are non-zero at unobserved cells"
    )


def _extract_and_standardise_boreholes(
    true_map: dict,
    resources: DecisionSimulationResources,
) -> np.ndarray:
    """Extract borehole variables from a true_map dict and standardise them.

    Returns
    -------
    np.ndarray of shape (n_x * n_y, V, D) float32, standardised and nan-zeroed.
    """
    variables = resources.variable_names
    n_x, n_y = true_map["yield_field"].shape[:2]
    n_depth = len(true_map["depth_axis"])

    bh = np.empty((n_x * n_y, len(variables), n_depth), dtype=np.float32)
    for vi, v in enumerate(variables):
        bh[:, vi, :] = true_map["variables"][v].reshape(n_x * n_y, n_depth)

    if resources.norm_stats:
        bh = standardise(bh, resources.norm_stats, variables)

    return np.nan_to_num(bh, nan=0.0).astype(np.float32)


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
        inputs_np = self.inputs.numpy()          # (N, 2+D, n_x, n_y)
        N, _, n_x, n_y = inputs_np.shape
        ore  = inputs_np[:, 0:1, :, :]           # (N, 1, n_x, n_y)
        mask = inputs_np[:, 1:2, :, :]           # (N, 1, n_x, n_y)
        latent_ch = inputs_np[:, 2:, :, :]       # (N, D, n_x, n_y)

        D = latent_ch.shape[1]
        lat_flat = latent_ch.transpose(0, 2, 3, 1).reshape(-1, D)  # (N*n_x*n_y, D)
        reduced  = reducer.transform(lat_flat)                       # (N*n_x*n_y, K)
        K = reduced.shape[1]

        reduced = reduced.reshape(N, n_x, n_y, K).transpose(0, 3, 1, 2)  # (N, K, n_x, n_y)
        reduced *= (mask > 0.0)                  # re-zero unobserved cells

        self.inputs = torch.from_numpy(
            np.concatenate([ore, mask, reduced], axis=1)
        )

    def apply_coordinate_channels(self) -> None:
        """Append normalized x and y coordinate channels to every sample in-place.

        The two new channels are appended after all existing channels so the
        final layout becomes ``[ore, mask, latents..., x_coord, y_coord]``.
        Both grids are in ``[0, 1]``: x varies along the row axis, y along
        the column axis.
        """
        N, _, n_x, n_y = self.inputs.shape
        coord = torch.from_numpy(make_coordinate_grid(n_x, n_y))  # (2, n_x, n_y)
        coord = coord.unsqueeze(0).expand(N, -1, -1, -1)          # (N, 2, n_x, n_y)
        self.inputs = torch.cat([self.inputs, coord], dim=1)

    @classmethod
    def generate(
        cls,
        resources: DecisionSimulationResources,
        cfg: BeliefDatasetConfig,
        device: str,
        sim_cfg: SimConfig | None = None,
        validate_samples: bool = False,
        verbose: bool = True,
    ) -> "GeologicalBeliefDataset":
        """Generate the full dataset from scratch using map synthesis + encoding.

        For each map the full latent map is computed once (all boreholes encoded
        in a single batched pass), then ``cfg.samples_per_map`` random drill
        patterns are drawn from it.

        Parameters
        ----------
        resources        : shared experiment resources (encoder model, norm stats, ...)
        cfg              : dataset size and drilling parameters
        device           : torch device for encoder forward passes
        sim_cfg          : map dimensions / ore parameters (defaults to SimConfig())
        validate_samples : if True, assert shape/content invariants on every sample
        verbose          : print progress every 10 maps
        """
        if sim_cfg is None:
            sim_cfg = SimConfig()

        rng = np.random.default_rng(cfg.seed)
        gen = MapGenerator(
            resources.distribution_bank,
            resources.formation_geometry,
            sim_cfg,
            seed=int(rng.integers(1 << 31)),
            prior=resources.discovery_prior,
        )

        n_x, n_y = sim_cfg.n_x, sim_cfg.n_y
        all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]

        all_inputs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        drill_count_list: list[int] = []

        for map_idx in range(cfg.n_maps):
            true_map = next(gen)

            bh_arr = _extract_and_standardise_boreholes(true_map, resources)
            full_latent_map = encode_full_latent_map(bh_arr, n_x, n_y, resources, device)
            target_ore = build_ore_target(true_map)  # (n_x, n_y)

            for _ in range(cfg.samples_per_map):
                n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
                chosen = rng.choice(len(all_locations), size=n_drills, replace=False)
                drill_locs = [all_locations[k] for k in chosen]
                ore_vals = [float(target_ore[i, j]) for i, j in drill_locs]

                inp = build_sample_input(drill_locs, ore_vals, full_latent_map)
                tgt = target_ore[np.newaxis]  # (1, n_x, n_y)

                if validate_samples:
                    _validate_sample(inp, tgt, drill_locs, n_x, n_y)

                all_inputs.append(inp)
                all_targets.append(tgt)
                drill_count_list.append(n_drills)

            if verbose and (map_idx + 1) % 10 == 0:
                print(f"  [dataset] {map_idx + 1}/{cfg.n_maps} maps generated")

        inputs_t = torch.from_numpy(np.stack(all_inputs, axis=0))
        targets_t = torch.from_numpy(np.stack(all_targets, axis=0))
        drill_counts_t = torch.tensor(drill_count_list, dtype=torch.long)
        return cls(inputs_t, targets_t, drill_counts=drill_counts_t)


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

        used_samples = samples[:samples_per_map] if samples_per_map is not None else samples
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
