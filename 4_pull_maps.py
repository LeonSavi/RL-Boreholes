"""Pre-generate N simulator maps and persist their borehole tensors.

Why pre-generate?
-----------------
Online generation in train_jepa.py is CPU-bound — KDE resampling and
per-cell rock lookups limit the loop to ~3 s/step on a GPU that could
otherwise do <10 ms/step.  Generating the dataset once, in parallel,
decouples that cost from training and lets the GPU train at its
natural rate.

Output layout
-------------
    OUT_DIR/
      stats.pkl                  pickle of standardisation stats
                                 (computed once from --n-stats-maps)
      config.pkl                 pickle of {variables, n_maps, seed}
      boreholes_00000.npy        float16, shape (1024, V, D)
      boreholes_00001.npy        each map already standardised and
      ...                        nan-zeroed — just load and stream.

Each map is ~5 MB on disk (float16 of (1024, 6, 440)).  10 000 maps ≈ 50 GB.

Usage
-----
    python pull_maps.py
    # or with explicit overrides
    python pull_maps.py --out-dir data/dataset --n-maps 10000 --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder.autoencoder import standardise
from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import MapGenerator, SimConfig
from train_encoder import boreholes_from_map, compute_standardisation_stats


# globals lazily populated in each worker process (see _worker_init)
_WORKER_GEN = None
_WORKER_VARS = None
_WORKER_STATS = None
_WORKER_OUT = None
_WORKER_ROCK_VOCAB: dict[str, int] | None = None   # rock_type_fine → idx
_WORKER_FM_VOCAB:   dict[str, int] | None = None   # formation     → idx


def _worker_init(bank_path, geom_path, prior_path, out_dir, stats,
                 variables, worker_seed_base, rock_vocab, fm_vocab):
    global _WORKER_GEN, _WORKER_VARS, _WORKER_STATS, _WORKER_OUT
    global _WORKER_ROCK_VOCAB, _WORKER_FM_VOCAB
    bank = DistributionBank.load(bank_path)
    geom = FormationGeometry.load(geom_path)
    prior = DiscoveryPrior.load(prior_path)
    # per-process seed: base + os.getpid() so workers never collide
    seed = worker_seed_base + os.getpid()
    _WORKER_GEN = MapGenerator(bank, geom, SimConfig(), seed=seed, prior=prior)
    _WORKER_VARS = list(variables)
    _WORKER_STATS = stats
    _WORKER_OUT = Path(out_dir)
    _WORKER_ROCK_VOCAB = rock_vocab
    _WORKER_FM_VOCAB = fm_vocab


def _encode_labels(arr: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    """Map a (nx, ny, nz) object array of strings to (nx*ny, nz) int8.

    Unknown labels (i.e. not in vocab) become 0; index 0 is reserved for
    'other'/unknown in the saved vocab (see _build_vocabs).
    """
    nx, ny, nz = arr.shape
    flat = arr.reshape(nx * ny, nz).astype(object)
    out = np.zeros_like(flat, dtype=np.int8)
    # vocab keys could be many; pandas Categorical lookup is fast enough
    import pandas as pd
    cat = pd.Categorical(flat.ravel(), categories=list(vocab.keys()))
    codes = cat.codes.copy()
    codes[codes < 0] = vocab.get("other", 0)        # treat OOV as 'other'
    out = codes.reshape(nx * ny, nz).astype(np.int8)
    return out


def _generate_one(map_idx: int) -> tuple[int, float]:
    """Generate one map.  Saves two files per map:
      boreholes_NNNNN.npy   (1024, V, D)  float16 — standardised values
      labels_NNNNN.npz      contains 'rocks' and 'formations'
                            (1024, D) int8 — vocab indices for downstream
                            sim-vs-real analysis.

    Both files are skipped if they already exist (resume support).
    """
    bh_path     = _WORKER_OUT / f"boreholes_{map_idx:05d}.npy"
    labels_path = _WORKER_OUT / f"labels_{map_idx:05d}.npz"
    if bh_path.exists() and labels_path.exists():
        return map_idx, 0.0
    t0 = time.perf_counter()
    m = next(_WORKER_GEN)

    bh = boreholes_from_map(m, _WORKER_VARS)
    bh = standardise(bh, _WORKER_STATS, _WORKER_VARS)
    bh = np.nan_to_num(bh, nan=0.0).astype(np.float16)
    np.save(bh_path, bh)

    if _WORKER_ROCK_VOCAB is not None and _WORKER_FM_VOCAB is not None:
        rocks = _encode_labels(np.asarray(m["rock_types"]),
                                _WORKER_ROCK_VOCAB)        # (1024, D) int8
        forms = _encode_labels(np.asarray(m["formations"]),
                                _WORKER_FM_VOCAB)
        np.savez_compressed(labels_path, rocks=rocks, formations=forms)

    return map_idx, time.perf_counter() - t0


def _build_vocabs(bank: DistributionBank, geom: FormationGeometry
                  ) -> tuple[dict[str, int], dict[str, int]]:
    """Fixed string→idx maps for rocks and formations.

    Index 0 is reserved for 'other' so unknown / OOV labels map to 'other'
    on encode (and decode back cleanly).  All known labels get a unique
    positive index.  Order is deterministic (sorted) so re-running yields
    identical files.
    """
    rocks_seen = {"other"}
    rocks_seen.update(getattr(bank, "rock_types", []) or [])
    rocks_seen.update(k[0] for k in getattr(bank, "cells", {}).keys())
    rocks = ["other"] + sorted(r for r in rocks_seen if r != "other")
    rock_vocab = {r: i for i, r in enumerate(rocks)}

    fms_seen = set(getattr(geom, "formations", {}) or {})
    fms_seen.add("other")
    formations = ["other"] + sorted(f for f in fms_seen if f != "other")
    fm_vocab = {f: i for i, f in enumerate(formations)}
    return rock_vocab, fm_vocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/dataset"))
    p.add_argument("--n-maps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--n-stats-maps", type=int, default=10,
                   help="number of maps used to fit standardisation stats")
    p.add_argument("--bank", type=Path,
                   default=Path("data/clean/distributions.pkl"))
    p.add_argument("--geom", type=Path,
                   default=Path("data/clean/formation_geometry.pkl"))
    p.add_argument("--prior", type=Path,
                   default=Path("data/clean/discovery_prior.pkl"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- stats + metadata (computed once, serially) ---------------------
    stats_path = args.out_dir / "stats.pkl"
    config_path = args.out_dir / "config.pkl"
    variables = list(SimConfig().variables)

    vocab_path = args.out_dir / "labels_vocab.pkl"

    if stats_path.exists() and config_path.exists():
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        with open(config_path, "rb") as f:
            saved_cfg = pickle.load(f)
        if saved_cfg.get("n_maps") != args.n_maps:
            print(f"  note: existing config has n_maps={saved_cfg.get('n_maps')},"
                  f" requested {args.n_maps}.  Will continue toward the new total.")
            saved_cfg["n_maps"] = args.n_maps
            with open(config_path, "wb") as f:
                pickle.dump(saved_cfg, f)
        variables = saved_cfg.get("variables", variables)
        print(f"  resuming with stats from {stats_path}")
        # vocabs may not have existed in older runs; (re)build & save if missing
        if not vocab_path.exists():
            print(f"  building label vocab ...")
            bank = DistributionBank.load(args.bank)
            geom = FormationGeometry.load(args.geom)
            rock_vocab, fm_vocab = _build_vocabs(bank, geom)
            with open(vocab_path, "wb") as f:
                pickle.dump({"rocks": rock_vocab, "formations": fm_vocab}, f)
            del bank, geom
        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        rock_vocab = vocabs["rocks"]
        fm_vocab = vocabs["formations"]
    else:
        print(f"  computing standardisation stats from "
              f"{args.n_stats_maps} maps...")
        bank = DistributionBank.load(args.bank)
        geom = FormationGeometry.load(args.geom)
        prior = DiscoveryPrior.load(args.prior)
        gen = MapGenerator(bank, geom, SimConfig(),
                           seed=args.seed, prior=prior)
        stats = compute_standardisation_stats(
            gen, variables, n_maps=args.n_stats_maps,
        )
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)
        with open(config_path, "wb") as f:
            pickle.dump({
                "variables": variables,
                "n_maps": args.n_maps,
                "seed": args.seed,
            }, f)
        for v, (mu, sd) in stats.items():
            print(f"    {v:14s}  mean={mu:8.3f}  std={sd:8.3f}")

        rock_vocab, fm_vocab = _build_vocabs(bank, geom)
        with open(vocab_path, "wb") as f:
            pickle.dump({"rocks": rock_vocab, "formations": fm_vocab}, f)
        print(f"    label vocab: {len(rock_vocab)} rocks, "
              f"{len(fm_vocab)} formations  -> {vocab_path}")

        del bank, geom, prior, gen   # free before workers start

    # ---- which indices need to be generated -----------------------------
    pending = [i for i in range(args.n_maps)
               if not ((args.out_dir / f"boreholes_{i:05d}.npy").exists()
                       and (args.out_dir / f"labels_{i:05d}.npz").exists())]
    if not pending:
        print(f"  all {args.n_maps} maps already exist in {args.out_dir}")
        return
    print(f"  generating {len(pending)} maps with {args.workers} workers "
          f"(skipping {args.n_maps - len(pending)} already saved)")

    # ---- parallel generation -------------------------------------------
    t0 = time.perf_counter()
    completed = 0
    last_report = t0
    init_args = (str(args.bank), str(args.geom), str(args.prior),
                 str(args.out_dir), stats, variables, args.seed * 7919,
                 rock_vocab, fm_vocab)
    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=init_args) as pool:
        for idx, dt in pool.imap_unordered(
            _generate_one, pending, chunksize=4,
        ):
            completed += 1
            now = time.perf_counter()
            if now - last_report >= 10 or completed == len(pending):
                elapsed = now - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(pending) - completed
                eta = remaining / rate if rate > 0 else 0
                print(f"    {completed:>6d}/{len(pending)}  "
                      f"({rate:.1f} maps/s, ETA {eta/60:.1f} min)")
                last_report = now

    print(f"\n  done in {(time.perf_counter()-t0)/60:.1f} min")
    print(f"  dataset -> {args.out_dir}")


if __name__ == "__main__":
    main()
