"""
Training script — trains the BoreholeAutoencoder on simulator-generated
boreholes.  Streams data on the fly from MapGenerator.

Usage
-----
    python -m simulator.train \
        --distributions data/clean/distributions.pkl \
        --out checkpoints/ae.pt \
        --steps 5000

Notes on scale: each step generates one map (32x32=1024 boreholes) and
takes a training step on a batch drawn from those boreholes.  With a
5070 Ti, expect roughly 50-200 steps/min depending on map size.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from simulator.distributions import DistributionBank
from simulator.map_generator import MapGenerator, SimConfig
from simulator.autoencoder import (
    BoreholeAutoencoder, AEConfig, save_checkpoint,
    standardise,
)

if not Path("data/clean/distributions.pkl").exists():
    bank = DistributionBank.fit(
        "data/clean/samples.parquet",
        variables=["rhob", "gr_api", "dt_us_ft", "nphi", "pef",
                "cali_in", "res_deep_log", "sp_mv", "drho", "msus_si"],
        depth_bins=[0, 300, 800, 1500, 3000, 6000],
    )
    bank.save("data/clean/distributions.pkl")
    print(bank.summary())


def boreholes_from_map(map_data: dict, variables: list[str]) -> np.ndarray:
    """Extract all boreholes from a generated map as (n_boreholes, V, D)."""
    nx, ny = map_data["rock_types"].shape[:2]
    nz = len(map_data["depth_axis"])
    out = np.empty((nx * ny, len(variables), nz), dtype=np.float32)
    for i, v in enumerate(variables):
        arr = map_data["variables"][v]  # (nx, ny, nz)
        out[:, i, :] = arr.reshape(nx * ny, nz)
    return out


def compute_standardisation_stats(
    gen: MapGenerator,
    variables: list[str],
    n_maps: int = 10,
) -> dict[str, tuple[float, float]]:
    """Run a few maps to estimate per-variable mean and std."""
    accum = {v: [] for v in variables}
    for _ in range(n_maps):
        m = next(gen)
        for v in variables:
            arr = m["variables"][v]
            finite = arr[np.isfinite(arr)]
            accum[v].append(finite)
    stats = {}
    for v in variables:
        if accum[v]:
            all_vals = np.concatenate(accum[v])
            stats[v] = (float(all_vals.mean()), float(all_vals.std()))
        else:
            stats[v] = (0.0, 1.0)
    return stats


def stream_batches(
    gen: MapGenerator,
    variables: list[str],
    stats: dict,
    batch_size: int = 128,
    device: str = "cpu",
) -> Iterator[torch.Tensor]:
    """Yield batches of standardised boreholes from a stream of generated maps."""
    buf = []
    while True:
        map_data = next(gen)
        boreholes = boreholes_from_map(map_data, variables)  # (1024, V, D)
        # NaN guard — replace with 0 after standardisation (z=0 means mean)
        boreholes = standardise(boreholes, stats, variables)
        boreholes = np.nan_to_num(boreholes, nan=0.0)
        buf.append(boreholes)
        total = sum(b.shape[0] for b in buf)
        while total >= batch_size:
            stacked = np.concatenate(buf, axis=0)
            batch = stacked[:batch_size]
            rest = stacked[batch_size:]
            buf = [rest] if rest.shape[0] > 0 else []
            total = rest.shape[0]
            yield torch.from_numpy(batch).to(device)


def train(
    distributions_path: Path,
    out_path: Path,
    steps: int = 5000,
    batch_size: int = 128,
    lr: float = 3e-4,
    latent_dim: int = 128,
    mask_prob: float = 0.3,
    log_every: int = 50,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    bank = DistributionBank.load(distributions_path)
    sim_cfg = SimConfig()
    variables = list(sim_cfg.variables)

    # streaming data
    gen = MapGenerator(bank, sim_cfg, seed=42)
    print(f"computing standardisation stats (10 maps)...")
    stats = compute_standardisation_stats(gen, variables, n_maps=10)
    for v, (m, s) in stats.items():
        print(f"  {v:14s}  mean={m:8.3f}  std={s:8.3f}")

    # model
    ae_cfg = AEConfig(
        n_variables=len(variables),
        n_depth=sim_cfg.n_depth,
        latent_dim=latent_dim,
        mask_prob=mask_prob,
    )
    model = BoreholeAutoencoder(ae_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    batches = stream_batches(gen, variables, stats, batch_size, device)

    loss_running = 0.0
    for step in range(1, steps + 1):
        x = next(batches)
        recon, z = model(x)
        # reconstruction target is the UNMASKED input, even if the encoder
        # saw a masked version.  This forces the latent to encode enough
        # info to reconstruct the full profile from partial observations.
        loss = loss_fn(recon, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_running = 0.95 * loss_running + 0.05 * loss.item() if step > 1 else loss.item()

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  (ema {loss_running:.4f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, stats, variables, out_path)
    print(f"checkpoint -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--mask-prob", type=float, default=0.3)
    args = p.parse_args()
    train(
        distributions_path=args.distributions,
        out_path=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        mask_prob=args.mask_prob,
    )


if __name__ == "__main__":
    main()
