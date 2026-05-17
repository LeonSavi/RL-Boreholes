"""Pre-generate N formation-resolution simulator maps.

Parallel companion to `4_pull_maps.py`. Uses `FormationMapGenerator`
(variable values drawn from a formation-indexed bank) instead of
`MapGenerator` (rock-indexed). Output layout is identical so the
existing training scripts can consume `data/dataset_formation/`
without modification:

    OUT_DIR/
      stats.pkl
      config.pkl
      labels_vocab.pkl                {"rocks": {...}, "formations": {...}}
      boreholes_NNNNN.npy             (1024, V, D) float16, standardised
      labels_NNNNN.npz                {"rocks": int8, "formations": int8}

Both rocks and formations are still saved in the labels file — the
column-level stratigraphy (rock Markov chains inside each formation)
is unchanged; only the source of variable values changes.

Usage
-----
    python 4b_pull_maps_formation.py
    python 4b_pull_maps_formation.py --n-maps 5000 --workers 16
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
from simulator.formation_distributions import FormationDistributionBank
from simulator.distributions import DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import FormationMapGenerator, SimConfig
from train_encoder import boreholes_from_map, compute_standardisation_stats


_WORKER_GEN = None
_WORKER_VARS = None
_WORKER_STATS = None
_WORKER_OUT = None
_WORKER_ROCK_VOCAB: dict[str, int] | None = None
_WORKER_FM_VOCAB:   dict[str, int] | None = None


def _worker_init(bank_path, geom_path, prior_path, out_dir, stats,
                 variables, worker_seed_base, rock_vocab, fm_vocab):
    global _WORKER_GEN, _WORKER_VARS, _WORKER_STATS, _WORKER_OUT
    global _WORKER_ROCK_VOCAB, _WORKER_FM_VOCAB
    bank = FormationDistributionBank.load(bank_path)
    geom = FormationGeometry.load(geom_path)
    prior = DiscoveryPrior.load(prior_path)
    seed = worker_seed_base + os.getpid()
    _WORKER_GEN = FormationMapGenerator(
        bank, geom, SimConfig(), seed=seed, prior=prior,
    )
    _WORKER_VARS = list(variables)
    _WORKER_STATS = stats
    _WORKER_OUT = Path(out_dir)
    _WORKER_ROCK_VOCAB = rock_vocab
    _WORKER_FM_VOCAB = fm_vocab


def _encode_labels(arr: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    nx, ny, nz = arr.shape
    flat = arr.reshape(nx * ny, nz).astype(object)
    import pandas as pd
    cat = pd.Categorical(flat.ravel(), categories=list(vocab.keys()))
    codes = cat.codes.copy()
    codes[codes < 0] = vocab.get("other", 0)
    return codes.reshape(nx * ny, nz).astype(np.int8)


def _generate_one(map_idx: int) -> tuple[int, float]:
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
                                _WORKER_ROCK_VOCAB)
        forms = _encode_labels(np.asarray(m["formations"]),
                                _WORKER_FM_VOCAB)
        np.savez_compressed(labels_path, rocks=rocks, formations=forms)

    return map_idx, time.perf_counter() - t0


def _build_vocabs(bank: FormationDistributionBank,
                  geom: FormationGeometry,
                  ) -> tuple[dict[str, int], dict[str, int]]:
    """Build {rocks, formations} string→idx maps.

    Rocks come from FormationGeometry (the rock Markov chain pool is
    what actually appears in generated columns). Formations come from
    both the bank and the geometry to cover the full set.
    """
    rocks_seen: set[str] = {"other"}
    for stats in geom.formations.values():
        rocks_seen.update(getattr(stats, "facies", {}).keys() or [])
    rocks = ["other"] + sorted(r for r in rocks_seen if r != "other")
    rock_vocab = {r: i for i, r in enumerate(rocks)}

    fms_seen: set[str] = {"other"}
    fms_seen.update(getattr(bank, "formations", []) or [])
    fms_seen.update(getattr(geom, "formations", {}) or {})
    formations = ["other"] + sorted(f for f in fms_seen if f != "other")
    fm_vocab = {f: i for i, f in enumerate(formations)}
    return rock_vocab, fm_vocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/dataset_formation"))
    p.add_argument("--n-maps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--n-stats-maps", type=int, default=10)
    p.add_argument("--bank", type=Path,
                   default=Path("data/clean/formation_distributions.pkl"))
    p.add_argument("--geom", type=Path,
                   default=Path("data/clean/formation_geometry.pkl"))
    p.add_argument("--prior", type=Path,
                   default=Path("data/clean/discovery_prior.pkl"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = args.out_dir / "stats.pkl"
    config_path = args.out_dir / "config.pkl"
    vocab_path = args.out_dir / "labels_vocab.pkl"
    variables = list(SimConfig().variables)

    if stats_path.exists() and config_path.exists():
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        with open(config_path, "rb") as f:
            saved_cfg = pickle.load(f)
        if saved_cfg.get("n_maps") != args.n_maps:
            print(f"  note: existing config has n_maps={saved_cfg.get('n_maps')},"
                  f" requested {args.n_maps}.  Continuing toward the new total.")
            saved_cfg["n_maps"] = args.n_maps
            with open(config_path, "wb") as f:
                pickle.dump(saved_cfg, f)
        variables = saved_cfg.get("variables", variables)
        print(f"  resuming with stats from {stats_path}")
        if not vocab_path.exists():
            print(f"  building label vocab ...")
            bank = FormationDistributionBank.load(args.bank)
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
        bank = FormationDistributionBank.load(args.bank)
        geom = FormationGeometry.load(args.geom)
        prior = DiscoveryPrior.load(args.prior)
        gen = FormationMapGenerator(bank, geom, SimConfig(),
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

        del bank, geom, prior, gen

    pending = [i for i in range(args.n_maps)
               if not ((args.out_dir / f"boreholes_{i:05d}.npy").exists()
                       and (args.out_dir / f"labels_{i:05d}.npz").exists())]
    if not pending:
        print(f"  all {args.n_maps} maps already exist in {args.out_dir}")
        return
    print(f"  generating {len(pending)} maps with {args.workers} workers "
          f"(skipping {args.n_maps - len(pending)} already saved)")

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
