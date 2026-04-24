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
    maps_per_refill: int = 4,
) -> Iterator[torch.Tensor]:
    """Yield batches of standardised boreholes, shuffled across multiple maps.

    Key design decision: each map produces ~1024 correlated boreholes
    (they share one stratigraphic sequence). If we emit sequential
    batches from the same map, consecutive gradient steps overfit to that
    map's lithology and then get whiplashed when the next map arrives.
    Loss oscillates instead of descending.

    Fix: generate `maps_per_refill` maps (~4-8k boreholes), shuffle the
    pooled boreholes, then emit batches. This decorrelates the batches
    and lets the model see diverse stratigraphies per gradient step.
    """
    rng = np.random.default_rng(0)
    while True:
        # pool boreholes from several maps
        pooled = []
        for _ in range(maps_per_refill):
            map_data = next(gen)
            bh = boreholes_from_map(map_data, variables)  # (nxny, V, D)
            bh = standardise(bh, stats, variables)
            bh = np.nan_to_num(bh, nan=0.0)
            pooled.append(bh)
        pooled = np.concatenate(pooled, axis=0)  # (maps_per_refill * nxny, V, D)
        # shuffle so consecutive batches come from different maps
        perm = rng.permutation(len(pooled))
        pooled = pooled[perm]
        # emit all full-size batches
        n_full = len(pooled) // batch_size
        for i in range(n_full):
            batch = pooled[i * batch_size : (i + 1) * batch_size]
            yield torch.from_numpy(batch).to(device)
        # remainder gets dropped — acceptable, next refill brings fresh data


def train(
    distributions_path: Path,
    out_path: Path,
    steps: int = 5000,
    batch_size: int = 128,
    lr: float = 3e-5,
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
    # SmoothL1 (Huber with δ=1) is more robust to heavy-tailed residuals
    # than MSE.  MSE would punish a ±4σ error 16× more than a ±1σ error;
    # Huber switches to linear beyond δ=1, so it's only 4× more.  Combined
    # with the ±4σ winsorisation in standardise(), this stabilises the
    # loss surface enough for the optimiser to descend instead of bouncing.
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    batches = stream_batches(gen, variables, stats, batch_size, device,
                             maps_per_refill=8)

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
        # gradient clipping: prevents catastrophic updates on batches with
        # outlier values (heavy-tailed variables like sp_mv and
        # res_deep_log would occasionally send the loss to 2+ and
        # destabilise training without this).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()

        loss_running = 0.95 * loss_running + 0.05 * loss.item() if step > 1 else loss.item()

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  (ema {loss_running:.4f})")

        # every 500 steps: show per-variable breakdown so we can see which
        # variables the model is struggling with.  Uses SmoothL1 (Huber)
        # to match the training loss.
        if step % 500 == 0:
            with torch.no_grad():
                per_var = []
                for i, v in enumerate(variables):
                    l = nn.functional.smooth_l1_loss(
                        recon[:, i, :], x[:, i, :], beta=1.0).item()
                    per_var.append((v, l))
            print(f"    per-variable: " + "  ".join(
                f"{v}={l:.2f}" for v, l in per_var))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, stats, variables, out_path)
    print(f"checkpoint -> {out_path}")


DISTR_DEFAULT = Path("data/clean/distributions.pkl")
CHECKPOINT_DEFAULT = Path("checkpoints/ae.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=Path, default=DISTR_DEFAULT,
                   help=f"path to fitted DistributionBank (default: {DISTR_DEFAULT})")
    p.add_argument("--out", type=Path, default=CHECKPOINT_DEFAULT,
                   help=f"path to save checkpoint (default: {CHECKPOINT_DEFAULT})")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
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