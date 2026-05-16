from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulator.map_generator import MapGenerator, SimConfig
from decision_simulator.resources import DecisionSimulationResources
from .dataset import BeliefDatasetConfig
from .utils import build_ore_target


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
    sim_cfg: SimConfig | None = None
    n_extra_maps: int = 0
    new_maps: list[dict] = field(default_factory=list)
    new_targets: list[np.ndarray] = field(default_factory=list)
    new_drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]] = field(
        default_factory=list
    )

    def __setstate__(self, state: dict) -> None:
        state.setdefault("sim_cfg", None)
        state.setdefault("n_extra_maps", 0)
        state.setdefault("new_maps", [])
        state.setdefault("new_targets", [])
        state.setdefault("new_drill_patterns", [])
        self.__dict__.update(state)

    def save(self, path: Path | str) -> None:
        self.maps.extend(self.new_maps)
        self.targets.extend(self.new_targets)
        self.drill_patterns.extend(self.new_drill_patterns)
        self.cfg = dataclasses.replace(self.cfg, n_maps=len(self.maps))
        self.new_maps.clear()
        self.new_targets.clear()
        self.new_drill_patterns.clear()
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
        """Check drill parameters match, ignoring n_maps (pool can be larger).

        samples_per_map uses >= so a pool with more stored patterns can serve
        a request for fewer (sliced at build time via build_dataset_from_cache).
        """
        c = self.cfg
        return (
            c.samples_per_map >= cfg.samples_per_map
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

    def ensure_pool_size(
        self,
        n_maps: int,
        resources: DecisionSimulationResources,
        pool_path: Path,
        sim_cfg: SimConfig | None = None,
    ) -> None:
        """Set n_extra_maps, then generate and save if the pool is too small."""
        if self.pool_size >= n_maps:
            self.n_extra_maps = 0
        else:
            self.n_extra_maps = n_maps - self.pool_size

        if self.n_extra_maps > 0:
            self.generate_new_maps(resources, sim_cfg)
            self.save(pool_path)
            print(f"  [pool] Saved -> {pool_path} ({self.pool_size} maps)")

    def generate_new_maps(
        self,
        resources: DecisionSimulationResources,
        sim_cfg: SimConfig | None = None,
        verbose: bool = True,
    ) -> None:
        """Generate self.n_extra_maps maps and store them in new_maps/new_targets/new_drill_patterns."""
        if sim_cfg is None:
            sim_cfg = SimConfig()
        self.sim_cfg = sim_cfg

        seed = self.cfg.seed + self.pool_size
        rng = np.random.default_rng(seed)
        map_generator = MapGenerator(
            resources.distribution_bank,
            resources.formation_geometry,
            sim_cfg,
            seed=int(rng.integers(1 << 31)),
            prior=resources.discovery_prior,
        )

        n_x, n_y = sim_cfg.n_x, sim_cfg.n_y
        all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]

        for map_idx in range(self.n_extra_maps):
            new_map = next(map_generator)
            target_ore = build_ore_target(new_map)

            samples: list[tuple[list[tuple[int, int]], list[float]]] = []
            for _ in range(self.cfg.samples_per_map):
                n_drills = int(
                    rng.integers(self.cfg.min_drills, self.cfg.max_drills + 1)
                )
                chosen = rng.choice(len(all_locations), size=n_drills, replace=False)
                drill_locs = [all_locations[k] for k in chosen]
                ore_vals = [float(target_ore[i, j]) for i, j in drill_locs]
                samples.append((drill_locs, ore_vals))

            self.new_maps.extend([new_map])
            self.new_targets.extend([target_ore])
            self.new_drill_patterns.extend([samples])

            if verbose and (map_idx + 1) % 10 == 0:
                print(f"  [cache] {map_idx + 1}/{self.n_extra_maps} maps generated")


def get_cache_handler(
    pool_path: Path,
    pool_cfg: BeliefDatasetConfig,
    n_total: int,
    sim_cfg: SimConfig | None = None,
) -> RawMapCache:
    """Load or create a RawMapCache, ready for ensure_pool_size."""
    pool: RawMapCache | None = None

    if pool_path.exists():
        pool = RawMapCache.load(pool_path)
        if not pool.drill_params_match(pool_cfg):
            print(
                f"  [pool] Drill parameters changed - discarding "
                f"{pool.pool_size}-map pool and regenerating ..."
            )
            pool = None
        elif (
            pool.sim_cfg is not None
            and sim_cfg is not None
            and pool.sim_cfg != sim_cfg
        ):
            print(
                f"  [pool] SimConfig changed - discarding "
                f"{pool.pool_size}-map pool and regenerating ..."
            )
            pool = None
        elif pool.pool_size >= n_total:
            print(
                f"  [pool] Loaded {pool_path.name} "
                f"({pool.pool_size} maps available, {n_total} needed)"
            )
        else:
            n_extra = n_total - pool.pool_size
            print(
                f"  [pool] Pool has {pool.pool_size} maps, need {n_total} - "
                f"generating {n_extra} more ..."
            )

    if pool is None:
        pool = RawMapCache(
            maps=[],
            targets=[],
            drill_patterns=[],
            cfg=dataclasses.replace(pool_cfg, n_maps=0),
        )
        print(f"  [pool] Generating {n_total} maps (seed={pool_cfg.seed}) ...")

    return pool
