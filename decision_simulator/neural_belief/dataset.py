from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from simulator.map_generator import MapGenerator, SimConfig
from decision_simulator.resources import DecisionSimulationResources
from .utils import TargetNormalizer, build_ore_target, encode_full_latent_map, build_sample_input

if TYPE_CHECKING:
    from .map_cache import RawMapCache


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

    # Mask must be strictly binary
    mask = inp[1]
    assert np.all((mask == 0.0) | (mask == 1.0)), (
        "observation mask contains non-binary values"
    )

    # Latent channels must be zero at every unobserved cell
    drilled = {(i, j) for i, j in drill_locs}
    unobserved = np.array(
        [[(i, j) not in drilled for j in range(n_y)] for i in range(n_x)],
        dtype=bool,
    )  # (n_x, n_y)
    latents = inp[2:]  # (latent_dim, n_x, n_y)
    assert np.all(latents[:, unobserved] == 0.0), (
        "latent channels are non-zero at unobserved cells"
    )


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
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    def apply_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Normalize stored targets in-place using a fitted TargetNormalizer."""
        targets_np = normalizer.transform(self.targets.numpy())
        self.targets = torch.from_numpy(targets_np)

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
        """Generate the full dataset from scratch using map synthesis + JEPA encoding.

        For each map the full latent map is computed once (1024 boreholes encoded
        in a single batched pass), then ``cfg.samples_per_map`` random drill
        patterns are drawn from it.  This amortises the encoding cost.

        Parameters
        ----------
        resources        : shared experiment resources (JEPA model, norm stats, ...)
        cfg              : dataset size and drilling parameters
        device           : torch device for JEPA forward passes
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

        for map_idx in range(cfg.n_maps):
            true_map = next(gen)

            # Expensive per-map computation done once.
            full_latent_map = encode_full_latent_map(true_map, resources, device)
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

            if verbose and (map_idx + 1) % 10 == 0:
                print(f"  [dataset] {map_idx + 1}/{cfg.n_maps} maps generated")

        inputs_t = torch.from_numpy(np.stack(all_inputs, axis=0))    # (N, C, n_x, n_y)
        targets_t = torch.from_numpy(np.stack(all_targets, axis=0))  # (N, 1, n_x, n_y)
        return cls(inputs_t, targets_t)


def build_dataset_from_cache(
    cache: RawMapCache,
    resources: DecisionSimulationResources,
    device: str,
    verbose: bool = False,
    samples_per_map: int | None = None,
) -> GeologicalBeliefDataset:
    """Build an encoder-specific dataset from a shared :class:`RawMapCache`.

    Encodes boreholes once per map using the encoder in ``resources``.
    For the no-encoder case (``resources.jepa_model=None`` and
    ``borehole_encoder_fn=None``), ``encode_full_latent_map`` returns a
    zero-channel array and inputs will be ``(2, n_x, n_y)``.
    Targets are returned in raw (unnormalized) ore-value space.
    """
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for map_idx, (true_map, target_ore, samples) in enumerate(
        zip(cache.maps, cache.targets, cache.drill_patterns)
    ):
        full_latent_map = encode_full_latent_map(true_map, resources, device)

        used_samples = samples[:samples_per_map] if samples_per_map is not None else samples
        for drill_locs, ore_vals in used_samples:
            inp = build_sample_input(drill_locs, ore_vals, full_latent_map)
            all_inputs.append(inp)
            all_targets.append(target_ore[np.newaxis])  # (1, n_x, n_y)

        if verbose and (map_idx + 1) % 10 == 0:
            print(f"  [encode] {map_idx + 1}/{len(cache.maps)} maps encoded")

    inputs_t = torch.from_numpy(np.stack(all_inputs, axis=0))
    targets_t = torch.from_numpy(np.stack(all_targets, axis=0))
    return GeologicalBeliefDataset(inputs_t, targets_t)
