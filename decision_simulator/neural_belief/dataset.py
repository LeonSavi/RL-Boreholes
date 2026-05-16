from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from simulator.map_generator import MapGenerator, SimConfig
from decision_simulator.resources import DecisionSimulationResources
from .utils import TargetNormalizer, build_ore_target, encode_full_latent_map, build_sample_input


@dataclass
class BeliefDatasetConfig:
    """Configuration for synthetic belief-state dataset generation."""

    n_maps: int = 50
    samples_per_map: int = 20
    min_drills: int = 1
    max_drills: int = 15
    latent_dim: int = 128
    seed: int = 42


@dataclass
class RawMapCache:
    """Encoder-agnostic map and drill data, reusable across encoder variants.

    Stores raw ``true_map`` dicts, per-map target ore arrays, and drill
    patterns (locations + observed values) with no encoder-specific tensors.
    Latent embeddings are computed on demand per variant via
    ``build_dataset_from_cache``.
    """

    maps: list[dict]
    targets: list[np.ndarray]  # per-map: (n_x, n_y) float32
    # drill_patterns[map_idx][sample_idx] = (drill_locs, ore_vals)
    drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]]
    cfg: BeliefDatasetConfig

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> RawMapCache:
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def pool_size(self) -> int:
        return len(self.maps)

    def drill_params_match(self, cfg: BeliefDatasetConfig) -> bool:
        """Check drill parameters match, ignoring n_maps (pool can be larger)."""
        c = self.cfg
        return (
            c.samples_per_map == cfg.samples_per_map
            and c.min_drills == cfg.min_drills
            and c.max_drills == cfg.max_drills
            and c.seed == cfg.seed
        )

    def subset(self, indices: list[int]) -> "RawMapCache":
        """Return a new RawMapCache containing only the specified map indices."""
        return RawMapCache(
            maps=[self.maps[i] for i in indices],
            targets=[self.targets[i] for i in indices],
            drill_patterns=[self.drill_patterns[i] for i in indices],
            cfg=dataclasses.replace(self.cfg, n_maps=len(indices)),
        )

    def extend(self, other: "RawMapCache") -> None:
        """Append maps from another cache to this pool in-place."""
        self.maps.extend(other.maps)
        self.targets.extend(other.targets)
        self.drill_patterns.extend(other.drill_patterns)
        self.cfg = dataclasses.replace(self.cfg, n_maps=len(self.maps))

    def config_matches(self, other: BeliefDatasetConfig) -> bool:
        c = self.cfg
        return (
            c.n_maps == other.n_maps
            and c.samples_per_map == other.samples_per_map
            and c.min_drills == other.min_drills
            and c.max_drills == other.max_drills
            and c.seed == other.seed
        )


def _validate_sample(
    inp: np.ndarray,
    target: np.ndarray,
    drill_locs: list[tuple[int, int]],
    latent_dim: int,
    n_x: int,
    n_y: int,
) -> None:
    """Assert shape and content invariants for one generated sample."""
    assert inp.shape == (2 + latent_dim, n_x, n_y), (
        f"input shape {inp.shape} != ({2 + latent_dim}, {n_x}, {n_y})"
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
                    _validate_sample(inp, tgt, drill_locs, cfg.latent_dim, n_x, n_y)

                all_inputs.append(inp)
                all_targets.append(tgt)

            if verbose and (map_idx + 1) % 10 == 0:
                print(f"  [dataset] {map_idx + 1}/{cfg.n_maps} maps generated")

        inputs_t = torch.from_numpy(np.stack(all_inputs, axis=0))    # (N, C, n_x, n_y)
        targets_t = torch.from_numpy(np.stack(all_targets, axis=0))  # (N, 1, n_x, n_y)
        return cls(inputs_t, targets_t)


def generate_raw_map_cache(
    resources: DecisionSimulationResources,
    cfg: BeliefDatasetConfig,
    sim_cfg: SimConfig | None = None,
    verbose: bool = True,
) -> RawMapCache:
    """Generate maps and drill patterns without any encoder-specific encoding.

    The returned :class:`RawMapCache` can be passed to
    :func:`build_dataset_from_cache` repeatedly with different encoders to
    produce encoder-specific :class:`GeologicalBeliefDataset` instances while
    keeping maps and drill patterns identical across variants.
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

    maps: list[dict] = []
    targets: list[np.ndarray] = []
    drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]] = []

    for map_idx in range(cfg.n_maps):
        true_map = next(gen)
        target_ore = build_ore_target(true_map)

        samples: list[tuple[list[tuple[int, int]], list[float]]] = []
        for _ in range(cfg.samples_per_map):
            n_drills = int(rng.integers(cfg.min_drills, cfg.max_drills + 1))
            chosen = rng.choice(len(all_locations), size=n_drills, replace=False)
            drill_locs = [all_locations[k] for k in chosen]
            ore_vals = [float(target_ore[i, j]) for i, j in drill_locs]
            samples.append((drill_locs, ore_vals))

        maps.append(true_map)
        targets.append(target_ore)
        drill_patterns.append(samples)

        if verbose and (map_idx + 1) % 10 == 0:
            print(f"  [cache] {map_idx + 1}/{cfg.n_maps} maps generated")

    return RawMapCache(maps=maps, targets=targets, drill_patterns=drill_patterns, cfg=cfg)


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
