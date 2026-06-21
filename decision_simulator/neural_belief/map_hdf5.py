"""HDF5-backed map stores for neural-belief training.

:class:`HDF5MapStore` reads a single shard file.
:class:`HDF5MapDirectory` discovers and aggregates multiple shards in a directory.

Shards are produced by ``colab_npz_to_hdf5_full.py``.  Each shard's structure is:

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from .datasets import BeliefDatasetConfig, GeologicalBeliefDataset
from .models import standardise
from decision_simulator.resources import DecisionSimulationResources

_DEFAULT_PREFIX_STEPS = [1, 2, 3, 5, 8, 10, 15]


@dataclass
class MapPool:
    """In-memory container for map pool data loaded from HDF5 shards.

    Created by :meth:`HDF5MapStore.load_subset` or :meth:`HDF5MapDirectory.load_map_data`.
    """

    borehole_arrays: list[np.ndarray]  # each (n_boreholes, V, D) float32, raw
    targets: list[np.ndarray]          # each (n_x, n_y) float32
    drill_patterns: list[list[tuple[list[tuple[int, int]], list[float]]]]
    cfg: BeliefDatasetConfig
    n_x: int
    n_y: int
    rocks_arrays: list[np.ndarray] | None = None  # each (n_boreholes, D) int8, or None

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

        Returns np.ndarray of shape ``(n_x, n_y, latent_dim)`` float32.
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
        """Construct the ``(2 + latent_dim, n_x, n_y)`` model input from a drill pattern."""
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
                sparse_ore[np.newaxis],
                mask[np.newaxis],
                jepa_map.transpose(2, 0, 1),
            ],
            axis=0,
        )

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
        """Build a sequential prefix dataset from this map pool."""
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

                    inp = MapPool.build_sample_input(drill_locs, ore_vals, full_latent_map)
                    all_inputs.append(inp)
                    all_targets.append(target_ore[np.newaxis])
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


def unpack_drill_patterns(
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

    def load_subset(self, positions: list[int]) -> MapPool:
        """Load maps at *positions* (row indices into the HDF5 shard).

        Returns an in-memory :class:`MapPool`.
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
            rocks_arrays: list[np.ndarray] = []
            has_rocks = "rocks" in hf
            for pos in positions:
                borehole_arrays.append(hf["boreholes"][pos].astype(np.float32))
                targets.append(hf["yield_target"][pos].astype(np.float32))
                drill_patterns.append(
                    unpack_drill_patterns(
                        hf["drill_locs"][pos],
                        hf["drill_ore_vals"][pos],
                        hf["drill_counts"][pos],
                    )
                )
                if has_rocks:
                    rocks_arrays.append(hf["rocks"][pos])  # keep as int8

        return MapPool(
            borehole_arrays=borehole_arrays,
            targets=targets,
            drill_patterns=drill_patterns,
            cfg=dataclasses.replace(stored_cfg, n_maps=len(positions)),
            n_x=n_x,
            n_y=n_y,
            rocks_arrays=rocks_arrays if rocks_arrays else None,
        )

    def load_map_data(
        self,
        n_train_maps: int,
        n_val_maps: int,
        n_orebodies: int,
        seed: int = 42,
    ) -> tuple[MapPool, MapPool]:
        """Load train/val subsets stratified by ore body count.

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


class HDF5MapDirectory:
    """Multi-shard HDF5 store for a directory of shard files produced by
    ``colab_npz_to_hdf5_full.py``.

    Shard discovery priority
    ------------------------
    n_orebodies=1 (only 0- and 1-body maps needed):
        maps_0_and_1_orebodies_*.h5  →  maps_0_orebodies_*.h5  →  maps_1_orebodies_*.h5

    n_orebodies=2 or 3 (all body classes needed):
        maps_stratified_*.h5  →  maps_0_and_1_orebodies_*.h5
        →  maps_0_orebodies_*.h5  …  maps_N_orebodies_*.h5

    Parameters
    ----------
    directory
        Directory that holds the ``.h5`` shard files.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def _priority_files(self, n_orebodies: int) -> list[Path]:
        d = self.directory
        if n_orebodies == 1:
            priority = (
                sorted(d.glob("maps_0_and_1_orebodies_*.h5"))
                + sorted(d.glob("maps_0_orebodies_*.h5"))
                + sorted(d.glob("maps_1_orebodies_*.h5"))
            )
        else:
            priority = (
                sorted(d.glob("maps_stratified_*.h5"))
                + sorted(d.glob("maps_0_and_1_orebodies_*.h5"))
                + sorted(d.glob("maps_2_and_3_orebodies_*.h5"))
                + [p for cls in range(n_orebodies + 1)
                   for p in sorted(d.glob(f"maps_{cls}_orebodies_*.h5"))]
            )
        # Fall back to numeric-range shards produced by generate_training_maps.py
        # e.g. maps_00000_00499.h5, maps_00500_00999.h5, …
        return priority or sorted(d.glob("maps_[0-9][0-9][0-9][0-9][0-9]_*.h5"))

    def _load_by_positions(
        self,
        indices: list[int],
        file_row_map: list[tuple[Path, int]],
    ) -> MapPool:
        """Load maps referenced by combined-index positions, grouped by file."""
        by_file: dict[Path, list[int]] = defaultdict(list)
        for combined_idx in indices:
            fp, row = file_row_map[combined_idx]
            by_file[fp].append(row)

        all_bh: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_drills: list = []
        all_rocks: list[np.ndarray] = []
        cfg = None
        n_x = n_y = None

        for fp, rows in by_file.items():
            nm = HDF5MapStore(fp).load_subset(rows)
            all_bh.extend(nm.borehole_arrays)
            all_targets.extend(nm.targets)
            all_drills.extend(nm.drill_patterns)
            if nm.rocks_arrays is not None:
                all_rocks.extend(nm.rocks_arrays)
            if cfg is None:
                cfg = nm.cfg
                n_x, n_y = nm.n_x, nm.n_y

        return MapPool(
            borehole_arrays=all_bh,
            targets=all_targets,
            drill_patterns=all_drills,
            cfg=dataclasses.replace(cfg, n_maps=len(indices)),
            n_x=n_x,
            n_y=n_y,
            rocks_arrays=all_rocks if all_rocks else None,
        )

    def _select_files(self, n_total: int, n_orebodies: int) -> list[Path]:
        """Return the minimal set of shard files needed for this request.

        Avoids reading all shards when a smaller subset is sufficient:
          - n_orebodies=1, n_total < 2000 : only the 0+1-body shards
          - n_orebodies=1, n_total < 3000 : 0+1-body + 2+3-body shards
          - otherwise                     : full priority list
        """
        d = self.directory
        files_01 = sorted(d.glob("maps_0_and_1_orebodies_*.h5"))
        files_23 = (sorted(d.glob("maps_2_and_3_orebodies_*.h5"))
                    + sorted(d.glob("maps_2_orebodies_*.h5")))

        if n_orebodies == 1 and n_total < 2000 and files_01:
            return files_01
        if n_orebodies == 1 and n_total < 3000 and (files_01 or files_23):
            return files_01 + files_23
        return self._priority_files(n_orebodies)

    def load_map_data(
        self,
        n_train_maps: int,
        n_val_maps: int,
        n_orebodies: int,
        seed: int = 42,
    ) -> tuple[MapPool, MapPool]:
        """Load train/val subsets with the same stratified-sampling contract
        as :meth:`HDF5MapStore.load_map_data`, but spanning multiple shard files.

        Shards are discovered in priority order (see class docstring).  Only the
        ``n_bodies`` arrays are read from each file to build the combined index;
        actual borehole data is loaded only for the selected maps.

        Raises
        ------
        FileNotFoundError
            If no matching shard files are found in the directory.
        ValueError
            If ``n_orebodies`` is out of range or any class has too few maps.
        """
        if not isinstance(n_orebodies, int) or n_orebodies not in (1, 2, 3):
            raise ValueError(
                f"n_orebodies must be an integer 1, 2, or 3; got {n_orebodies!r}"
            )

        files = self._select_files(n_train_maps + n_val_maps, n_orebodies)
        if not files:
            raise FileNotFoundError(
                f"No HDF5 shards found in '{self.directory}' for n_orebodies={n_orebodies}. "
                "Run colab_npz_to_hdf5_full.py first."
            )

        # Build combined body index — cheap (small int8 arrays per file)
        combined_bodies: list[int] = []
        file_row_map: list[tuple[Path, int]] = []
        for fp in files:
            with h5py.File(fp, "r") as hf:
                bodies = hf["n_bodies"][:]
            for row, nb in enumerate(bodies):
                combined_bodies.append(int(nb))
                file_row_map.append((fp, row))

        combined_arr = np.array(combined_bodies, dtype=np.int8)
        classes = list(range(n_orebodies + 1))
        n_needed = n_train_maps + n_val_maps
        per_class = int(np.ceil(n_needed / len(classes)))

        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for cls in classes:
            candidates = np.where(combined_arr == cls)[0].tolist()
            if len(candidates) < per_class:
                raise ValueError(
                    f"Not enough maps with {cls} ore bodies across shards "
                    f"(need {per_class}, have {len(candidates)}). "
                    "Generate more shards or reduce n_train_maps/n_val_maps."
                )
            chosen = rng.choice(candidates, size=per_class, replace=False).tolist()
            selected.extend(chosen)

        rng.shuffle(selected)
        selected = selected[:n_needed]

        train_sel = selected[:n_train_maps]
        val_sel   = selected[n_train_maps:]
        return (
            self._load_by_positions(train_sel, file_row_map),
            self._load_by_positions(val_sel,   file_row_map),
        )
