"""NPZ-based map pool for neural-belief training.

Follows the same pattern as 4_pull_maps.py but also stores yield_field targets
and drill patterns needed by the neural-belief model.

Directory layout
----------------
    <pool_dir>/
        config.pkl              dataset + sim config, written map count
        stats.pkl               {var_name: (mean, std)} — saved for reference
        labels_vocab.pkl        {rocks: {name: idx}, formations: {name: idx}}
        map_00000.npz           per-map arrays (see below)
        map_00001.npz
        ...

Per-map npz keys
----------------
    boreholes     : (n_boreholes, V, D) float32 — raw (unstandardised) variables
    yield_target  : (n_x, n_y)         float32 — max-pooled yield field
    rocks         : (n_boreholes, D)   int8    — rock-type vocab indices
    formations    : (n_boreholes, D)   int8    — formation vocab indices
    drill_locs    : (S, max_drills, 2) int16   — drill locations per sample
    drill_ore_vals: (S, max_drills)    float32 — ore values at drill locations
    drill_counts  : (S,)               int16   — valid drills per sample
    where S = samples_per_map
"""
from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import MapGenerator, SimConfig
from decision_simulator.resources import DecisionSimulationResources
from .dataset import BeliefDatasetConfig
from .utils import build_ore_target


_CACHE_VERSION = "2"


# ---------------------------------------------------------------------------
# Label encoding helpers (ported from 4_pull_maps.py)
# ---------------------------------------------------------------------------

def _build_vocabs(
    bank: DistributionBank,
    geom: FormationGeometry,
) -> tuple[dict[str, int], dict[str, int]]:
    rocks_seen: set[str] = {"other"}
    rocks_seen.update(getattr(bank, "rock_types", []) or [])
    rocks_seen.update(k[0] for k in getattr(bank, "cells", {}).keys())
    rocks = ["other"] + sorted(r for r in rocks_seen if r != "other")
    rock_vocab = {r: i for i, r in enumerate(rocks)}

    fms_seen: set[str] = set(getattr(geom, "formations", {}) or {})
    fms_seen.add("other")
    formations = ["other"] + sorted(f for f in fms_seen if f != "other")
    fm_vocab = {f: i for i, f in enumerate(formations)}
    return rock_vocab, fm_vocab


def _encode_labels(arr: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    """Map a (nx, ny, nz) object array of strings to (nx*ny, nz) int8."""
    import pandas as pd
    nx, ny, nz = arr.shape
    flat = arr.reshape(nx * ny, nz).astype(object)
    cat = pd.Categorical(flat.ravel(), categories=list(vocab.keys()))
    codes = cat.codes.copy()
    codes[codes < 0] = vocab.get("other", 0)
    return codes.reshape(nx * ny, nz).astype(np.int8)


# ---------------------------------------------------------------------------
# Drill pattern helpers
# ---------------------------------------------------------------------------

def _pack_drill_patterns(
    samples: list[tuple[list[tuple[int, int]], list[float]]],
    max_drills: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(samples)
    locs = np.zeros((n, max_drills, 2), dtype=np.int16)
    vals = np.zeros((n, max_drills), dtype=np.float32)
    counts = np.zeros(n, dtype=np.int16)
    for i, (drill_locs, ore_vals) in enumerate(samples):
        k = len(drill_locs)
        counts[i] = k
        locs[i, :k] = drill_locs
        vals[i, :k] = ore_vals
    return locs, vals, counts


def _unpack_drill_patterns(
    locs: np.ndarray,
    vals: np.ndarray,
    counts: np.ndarray,
) -> list[tuple[list[tuple[int, int]], list[float]]]:
    samples = []
    for i, k in enumerate(counts):
        drill_locs = [(int(locs[i, j, 0]), int(locs[i, j, 1])) for j in range(k)]
        ore_vals = [float(vals[i, j]) for j in range(k)]
        samples.append((drill_locs, ore_vals))
    return samples


# ---------------------------------------------------------------------------
# Layer 1: In-memory container
# ---------------------------------------------------------------------------

@dataclass
class NpzMapCache:
    """In-memory container for map pool data loaded from an npz cache.

    Created by :meth:`NpzMapCacheStore.load_subset` and consumed by
    ``build_dataset_from_cache``.
    """

    borehole_arrays: list[np.ndarray]   # each (n_boreholes, V, D) float32, raw
    targets: list[np.ndarray]           # each (n_x, n_y) float32
    drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]]
    cfg: BeliefDatasetConfig
    n_x: int
    n_y: int

    @property
    def pool_size(self) -> int:
        return len(self.borehole_arrays)


# Backward-compatible alias
RawMapCache = NpzMapCache


# ---------------------------------------------------------------------------
# Layer 2: NPZ persistence
# ---------------------------------------------------------------------------

class _NpzWriteSession:
    """Context manager for batched map appends to an npz pool directory."""

    def __init__(
        self,
        store: "NpzMapCacheStore",
        cfg: BeliefDatasetConfig,
        sim_cfg: SimConfig,
        n_x: int,
        n_y: int,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._sim_cfg = sim_cfg
        self._n_x = n_x
        self._n_y = n_y

    def __enter__(self) -> "_NpzWriteSession":
        self._store.path.mkdir(parents=True, exist_ok=True)
        if not self._store._config_path().exists():
            self._flush_meta(0)
        return self

    def commit_map(
        self,
        global_idx: int,
        boreholes: np.ndarray,
        yield_target: np.ndarray,
        rocks: np.ndarray,
        formations: np.ndarray,
        samples: list[tuple[list[tuple[int, int]], list[float]]],
    ) -> None:
        locs, vals, counts = _pack_drill_patterns(samples, self._cfg.max_drills)
        np.savez_compressed(
            self._store._map_path(global_idx),
            boreholes=boreholes,
            yield_target=yield_target,
            rocks=rocks,
            formations=formations,
            drill_locs=locs,
            drill_ore_vals=vals,
            drill_counts=counts,
        )
        self._flush_meta(global_idx + 1)

    def _flush_meta(self, n_maps_written: int) -> None:
        cfg_path = self._store._config_path()
        meta: dict = {}
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                meta = pickle.load(f)
        meta.update({
            "cache_version": _CACHE_VERSION,
            "belief_cfg": self._cfg,
            "sim_cfg": dataclasses.asdict(self._sim_cfg),
            "n_maps_written": n_maps_written,
            "n_x": self._n_x,
            "n_y": self._n_y,
        })
        with open(cfg_path, "wb") as f:
            pickle.dump(meta, f)

    def __exit__(self, *_: object) -> None:
        pass


class NpzMapCacheStore:
    """Layer 2: NPZ directory persistence for raw belief map pools.

    Each map is stored as a compressed npz file.  Global metadata
    (config, stats, label vocab) lives in the same directory.

    Parameters
    ----------
    path
        Directory that holds the pool files.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _map_path(self, idx: int) -> Path:
        return self.path / f"map_{idx:05d}.npz"

    def _config_path(self) -> Path:
        return self.path / "config.pkl"

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def count_maps(self) -> int:
        """Return the number of fully committed maps."""
        cfg_path = self._config_path()
        if not cfg_path.exists():
            return 0
        with open(cfg_path, "rb") as f:
            meta = pickle.load(f)
        return int(meta.get("n_maps_written", 0))

    def load_subset(self, indices: list[int]) -> NpzMapCache:
        """Load the maps at *indices* and return an in-memory :class:`NpzMapCache`."""
        with open(self._config_path(), "rb") as f:
            meta = pickle.load(f)
        stored_cfg: BeliefDatasetConfig = meta["belief_cfg"]
        n_x: int = meta["n_x"]
        n_y: int = meta["n_y"]

        borehole_arrays, targets, drill_patterns = [], [], []
        for idx in indices:
            data = np.load(self._map_path(idx))
            borehole_arrays.append(data["boreholes"])
            targets.append(data["yield_target"])
            drill_patterns.append(
                _unpack_drill_patterns(
                    data["drill_locs"],
                    data["drill_ore_vals"],
                    data["drill_counts"],
                )
            )

        return NpzMapCache(
            borehole_arrays=borehole_arrays,
            targets=targets,
            drill_patterns=drill_patterns,
            cfg=dataclasses.replace(stored_cfg, n_maps=len(indices)),
            n_x=n_x,
            n_y=n_y,
        )

    # ------------------------------------------------------------------
    # Internal API (used by NpzMapCacheHandler)
    # ------------------------------------------------------------------

    def _read_metadata(
        self,
    ) -> tuple[str, BeliefDatasetConfig | None, SimConfig | None]:
        if not self._config_path().exists():
            return ("", None, None)
        with open(self._config_path(), "rb") as f:
            meta = pickle.load(f)
        version = str(meta.get("cache_version", ""))
        belief_cfg = meta.get("belief_cfg")
        sim_cfg_dict = meta.get("sim_cfg")
        sim_cfg = SimConfig(**sim_cfg_dict) if sim_cfg_dict is not None else None
        return version, belief_cfg, sim_cfg

    def _open_write_session(
        self,
        cfg: BeliefDatasetConfig,
        sim_cfg: SimConfig,
        n_x: int,
        n_y: int,
    ) -> _NpzWriteSession:
        return _NpzWriteSession(self, cfg, sim_cfg, n_x, n_y)


# ---------------------------------------------------------------------------
# Layer 3: Orchestration
# ---------------------------------------------------------------------------

class NpzMapCacheHandler:
    """Layer 3: High-level cache workflow.

    Owns compatibility validation, overwrite policy, map generation, and
    train/val split loading.  Delegates all I/O to :class:`NpzMapCacheStore`.

    Parameters
    ----------
    store
        The npz store to read from and write to.
    cfg
        Drill parameters; must match any existing pool.
    sim_cfg
        Map-generation config; must match any existing pool.
    overwrite_cache
        If ``True``, an incompatible existing directory is deleted and
        rebuilt.  If ``False`` (default), a :class:`ValueError` is raised.
    """

    def __init__(
        self,
        store: NpzMapCacheStore,
        cfg: BeliefDatasetConfig,
        sim_cfg: SimConfig | None = None,
        overwrite_cache: bool = False,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.overwrite_cache = overwrite_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_pool_size(
        self,
        n_maps: int,
        resources: DecisionSimulationResources,
    ) -> None:
        """Grow the pool to at least *n_maps* maps, generating as needed."""
        if self.store.path.exists() and self.store._config_path().exists():
            reasons = self._incompatibility_reasons()
            if reasons:
                reason_lines = "\n".join(f"  - {r}" for r in reasons)
                if self.overwrite_cache:
                    import shutil
                    print(
                        f"  [pool] Cache incompatible:\n{reason_lines}\n"
                        f"  [pool] overwrite_cache=True – deleting and regenerating ..."
                    )
                    shutil.rmtree(self.store.path)
                else:
                    raise ValueError(
                        f"Existing cache at '{self.store.path}' is incompatible:\n"
                        f"{reason_lines}\nPass overwrite_cache=True to delete and regenerate."
                    )

        current = self.store.count_maps()
        if current >= n_maps:
            print(
                f"  [pool] Loaded {self.store.path.name} "
                f"({current} maps available, {n_maps} needed)"
            )
            return

        n_extra = n_maps - current
        if current == 0:
            print(f"  [pool] Generating {n_maps} maps (seed={self.cfg.seed}) ...")
        else:
            print(
                f"  [pool] Pool has {current} maps, need {n_maps} – "
                f"generating {n_extra} more ..."
            )
        self._generate_and_append(n_extra, resources)

    def load_train_val_split(
        self,
        n_train_maps: int,
        n_val_maps: int,
    ) -> tuple[NpzMapCache, NpzMapCache]:
        """Load contiguous train and validation subsets from the pool."""
        train_cache = self.store.load_subset(list(range(n_train_maps)))
        val_cache = self.store.load_subset(
            list(range(n_train_maps, n_train_maps + n_val_maps))
        )
        return train_cache, val_cache

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _incompatibility_reasons(self) -> list[str]:
        try:
            stored_version, stored_cfg, stored_sim_cfg = self.store._read_metadata()
        except Exception as exc:
            return [f"could not read cache: {exc}"]

        if stored_version != _CACHE_VERSION:
            return [
                f"cache_version: stored={stored_version!r}, expected={_CACHE_VERSION!r}"
            ]
        if stored_cfg is None:
            return ["missing belief_cfg metadata"]

        reasons: list[str] = []
        if stored_cfg.samples_per_map < self.cfg.samples_per_map:
            reasons.append(
                f"samples_per_map: stored={stored_cfg.samples_per_map} "
                f"< requested={self.cfg.samples_per_map}"
            )
        if stored_cfg.min_drills != self.cfg.min_drills:
            reasons.append(
                f"min_drills: stored={stored_cfg.min_drills}, "
                f"requested={self.cfg.min_drills}"
            )
        if stored_cfg.max_drills != self.cfg.max_drills:
            reasons.append(
                f"max_drills: stored={stored_cfg.max_drills}, "
                f"requested={self.cfg.max_drills}"
            )
        if stored_cfg.seed != self.cfg.seed:
            reasons.append(
                f"seed: stored={stored_cfg.seed}, requested={self.cfg.seed}"
            )
        if (
            self.sim_cfg is not None
            and stored_sim_cfg is not None
            and stored_sim_cfg != self.sim_cfg
        ):
            reasons.append("SimConfig mismatch")
        return reasons

    def _generate_and_append(
        self,
        n_extra: int,
        resources: DecisionSimulationResources,
    ) -> None:
        sim_cfg = self.sim_cfg or SimConfig()
        current = self.store.count_maps()
        n_x, n_y = sim_cfg.n_x, sim_cfg.n_y

        # Label vocab — build once, save for reference
        vocab_path = self.store.path / "labels_vocab.pkl"
        self.store.path.mkdir(parents=True, exist_ok=True)
        if vocab_path.exists():
            with open(vocab_path, "rb") as f:
                vocabs = pickle.load(f)
            rock_vocab, fm_vocab = vocabs["rocks"], vocabs["formations"]
        else:
            rock_vocab, fm_vocab = _build_vocabs(
                resources.distribution_bank, resources.formation_geometry
            )
            with open(vocab_path, "wb") as f:
                pickle.dump({"rocks": rock_vocab, "formations": fm_vocab}, f)
            print(
                f"  [cache] label vocab: {len(rock_vocab)} rocks, "
                f"{len(fm_vocab)} formations"
            )

        # Standardisation stats — save for reference (not applied here)
        stats_path = self.store.path / "stats.pkl"
        if not stats_path.exists() and resources.norm_stats:
            with open(stats_path, "wb") as f:
                pickle.dump(resources.norm_stats, f)

        seed = self.cfg.seed + current
        rng = np.random.default_rng(seed)
        map_generator = MapGenerator(
            resources.distribution_bank,
            resources.formation_geometry,
            sim_cfg,
            seed=int(rng.integers(1 << 31)),
            prior=resources.discovery_prior,
        )

        all_locations = [(i, j) for i in range(n_x) for j in range(n_y)]
        variables = list(sim_cfg.variables)

        with self.store._open_write_session(self.cfg, sim_cfg, n_x, n_y) as session:
            for i in range(n_extra):
                global_idx = current + i
                m = next(map_generator)
                n_depth = len(m["depth_axis"])

                # Raw borehole variables — float32, unstandardised
                bh = np.empty((n_x * n_y, len(variables), n_depth), dtype=np.float32)
                for vi, v in enumerate(variables):
                    bh[:, vi, :] = m["variables"][v].reshape(n_x * n_y, n_depth)
                bh = np.nan_to_num(bh, nan=0.0)

                # Yield target: max-pool over depth → (n_x, n_y)
                yield_target = build_ore_target(m)

                # Geology labels
                rocks = _encode_labels(np.asarray(m["rock_types"]), rock_vocab)
                formations = _encode_labels(np.asarray(m["formations"]), fm_vocab)

                # Drill patterns
                samples: list[tuple[list[tuple[int, int]], list[float]]] = []
                for _ in range(self.cfg.samples_per_map):
                    n_drills = int(
                        rng.integers(self.cfg.min_drills, self.cfg.max_drills + 1)
                    )
                    chosen = rng.choice(len(all_locations), size=n_drills, replace=False)
                    drill_locs = [all_locations[k] for k in chosen]
                    ore_vals = [float(yield_target[r, c]) for r, c in drill_locs]
                    samples.append((drill_locs, ore_vals))

                session.commit_map(
                    global_idx, bh, yield_target, rocks, formations, samples
                )

                if (i + 1) % 10 == 0:
                    print(f"  [cache] {i + 1}/{n_extra} maps generated")

        print(f"  [pool] Saved -> {self.store.path} ({current + n_extra} maps)")


# Backward-compatible aliases
RawMapCacheStore = NpzMapCacheStore
RawMapCacheHandler = NpzMapCacheHandler
