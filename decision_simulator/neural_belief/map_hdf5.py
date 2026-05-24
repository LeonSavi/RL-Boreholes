"""HDF5-backed map store for neural-belief training.

Mirrors the interface of :class:`NpzMapCacheStore` but reads from a single
pre-built HDF5 shard instead of a directory of per-map ``.npz`` files.

The shard is produced by ``colab_npz_to_hdf5.py``.  Its structure is:

    File attributes : pool_n_x, pool_n_y, pool_samples_per_map,
                      pool_min_drills, pool_max_drills, pool_seed, n_maps
    Datasets (leading dim N = number of maps in shard):
        boreholes      (N, n_x*n_y, V, D) float32
        yield_target   (N, n_x, n_y)       float32
        rocks          (N, n_x*n_y, D)     int8
        formations     (N, n_x*n_y, D)     int8
        drill_locs     (N, S, max_drills, 2) int16
        drill_ore_vals (N, S, max_drills)  float32
        drill_counts   (N, S)              int16
        n_bodies       (N,)                int8
        class_label    (N,)                int8
        map_index      (N,)                int32
        filenames      (N,)                str
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np

from .datasets import BeliefDatasetConfig
from .map_cache import NpzMap, _unpack_drill_patterns


class HDF5MapStore:
    """HDF5 shard store for raw belief map pools.

    Parameters
    ----------
    path
        Path to the ``.h5`` shard file produced by ``colab_npz_to_hdf5.py``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _require_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"HDF5 shard not found: '{self.path}'. "
                "Generate it with colab_npz_to_hdf5.py first."
            )

    def load_subset(self, positions: list[int]) -> NpzMap:
        """Load maps at *positions* (row indices into the HDF5 shard).

        Returns an in-memory :class:`NpzMap` with the same structure as
        :meth:`NpzMapCacheStore.load_subset`.
        """
        self._require_file()

        with h5py.File(self.path, "r") as hf:
            n_x = int(hf.attrs["pool_n_x"])
            n_y = int(hf.attrs["pool_n_y"])
            stored_cfg = BeliefDatasetConfig(
                n_maps=int(hf.attrs["n_maps"]),
                samples_per_map=int(hf.attrs["pool_samples_per_map"]),
                min_drills=int(hf.attrs["pool_min_drills"]),
                max_drills=int(hf.attrs["pool_max_drills"]),
                seed=int(hf.attrs["pool_seed"]),
            )

            borehole_arrays: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            drill_patterns = []
            for pos in positions:
                borehole_arrays.append(hf["boreholes"][pos].astype(np.float32))
                targets.append(hf["yield_target"][pos].astype(np.float32))
                drill_patterns.append(
                    _unpack_drill_patterns(
                        hf["drill_locs"][pos],
                        hf["drill_ore_vals"][pos],
                        hf["drill_counts"][pos],
                    )
                )

        return NpzMap(
            borehole_arrays=borehole_arrays,
            targets=targets,
            drill_patterns=drill_patterns,
            cfg=dataclasses.replace(stored_cfg, n_maps=len(positions)),
            n_x=n_x,
            n_y=n_y,
        )

    def load_map_data(
        self,
        n_train_maps: int,
        n_val_maps: int,
        n_orebodies: int,
        seed: int = 42,
    ) -> tuple[NpzMap, NpzMap]:
        """Load train/val subsets stratified by ore body count.

        Mirrors :meth:`NpzMapCacheStore.load_map_data` exactly:

        * ``n_orebodies=1`` — 50 % zero-body, 50 % one-body
        * ``n_orebodies=2`` — 33 % each of 0, 1, 2 bodies
        * ``n_orebodies=3`` — 25 % each of 0, 1, 2, 3 bodies

        Parameters
        ----------
        n_orebodies
            Maximum number of ore bodies; must be 1, 2, or 3.

        Raises
        ------
        ValueError
            If ``n_orebodies`` is out of range or any required body-count class
            has too few maps in the shard.
        """
        if not isinstance(n_orebodies, int) or n_orebodies not in (1, 2, 3):
            raise ValueError(
                f"n_orebodies must be an integer 1, 2, or 3; got {n_orebodies!r}"
            )

        self._require_file()

        with h5py.File(self.path, "r") as hf:
            body_index: np.ndarray = hf["n_bodies"][:]  # (N,) int8

        classes = list(range(n_orebodies + 1))
        n_classes = len(classes)
        n_needed = n_train_maps + n_val_maps
        per_class = int(np.ceil(n_needed / n_classes))

        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for cls in classes:
            candidates = np.where(body_index == cls)[0].tolist()
            if len(candidates) < per_class:
                raise ValueError(
                    f"Not enough maps with {cls} ore bodies in shard "
                    f"(need {per_class}, have {len(candidates)}). "
                    "Use a larger HDF5 shard or reduce n_train_maps/n_val_maps."
                )
            chosen = rng.choice(candidates, size=per_class, replace=False).tolist()
            selected.extend(chosen)

        rng.shuffle(selected)
        selected = selected[:n_needed]

        train_positions = selected[:n_train_maps]
        val_positions = selected[n_train_maps:]
        return self.load_subset(train_positions), self.load_subset(val_positions)
