"""
Training script for the JEPA encoder.

Parallel to simulator/train.py but uses JEPA (latent-prediction) loss
instead of reconstruction loss.

Usage
-----
    python -m simulator.train_jepa \
        --distributions data/clean/distributions.pkl \
        --out checkpoints/jepa.pt \
        --steps 5000

Notes
-----
* Same data streaming as reconstruction training (stream_batches from
  simulator.train, same maps_per_refill=8 shuffling)
* Same grad clipping (0.5) and LR (3e-5) for comparability
* Target encoder updated via EMA after each step (momentum=0.996)
* The JEPA loss operates on token-level latents (not per-borehole),
  so loss magnitudes are NOT comparable to the reconstruction loss —
  we only compare via downstream metrics (silhouette in latent space)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from simulator import FormationGeometry, DiscoveryPrior

from simulator.distributions import DistributionBank
from simulator.map_generator import MapGenerator, SimConfig
from train_encoder import (
    compute_standardisation_stats, stream_batches,
)
from encoder.jepa_encoder import (
    JEPAModel, JEPAConfig, save_jepa_checkpoint,
    sample_context_target_masks,
)


DISTR_DEFAULT = Path("data/clean/distributions.pkl")
CHECKPOINT_DEFAULT = Path("checkpoints/jepa.pt")


def train_jepa(
    distributions_path: Path,
    out_path: Path,
    steps: int = 5000,
    batch_size: int = 128,
    lr: float = 3e-5,
    latent_dim: int = 128,
    context_keep_frac: float = 0.6,
    target_window_frac: float = 0.25,
    ema_momentum: float = 0.996,
    log_every: int = 50,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    bank = DistributionBank.load(distributions_path)
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    sim_cfg = SimConfig()
    variables = list(sim_cfg.variables)

    gen = MapGenerator(bank, geom, sim_cfg, seed=42, prior=prior)
    print(f"computing standardisation stats (10 maps)...")
    stats = compute_standardisation_stats(gen, variables, n_maps=10)
    for v, (m, s) in stats.items():
        print(f"  {v:14s}  mean={m:8.3f}  std={s:8.3f}")

    jepa_cfg = JEPAConfig(
        n_variables=len(variables),
        n_depth=sim_cfg.n_depth,
        latent_dim=latent_dim,
        context_keep_frac=context_keep_frac,
        target_window_frac=target_window_frac,
        ema_momentum=ema_momentum,
    )
    model = JEPAModel(jepa_cfg).to(device)
    opt = torch.optim.AdamW(
        # only the context encoder + predictor have gradients
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr, weight_decay=1e-5,
    )

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params_total:,} total "
          f"({n_params_trainable:,} trainable, "
          f"{n_params_total - n_params_trainable:,} frozen target encoder)")

    batches = stream_batches(gen, variables, stats, batch_size, device,
                             maps_per_refill=8)

    mask_rng = np.random.default_rng(0)
    T_tokens = model.context_encoder.n_tokens
    print(f"token sequence length (after CNN pooling): {T_tokens}")
    print(f"context tokens per sample: {int(context_keep_frac * T_tokens)}")
    print(f"target  tokens per sample: {int(target_window_frac * T_tokens)}")

    loss_running = 0.0
    for step in range(1, steps + 1):
        x = next(batches)                                    # (B, V, D)
        ctx_mask, tgt_mask = sample_context_target_masks(
            n_tokens=T_tokens, batch_size=x.size(0),
            cfg=jepa_cfg, rng=mask_rng,
        )
        ctx_mask = ctx_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        loss = model(x, ctx_mask, tgt_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
            max_norm=0.5,
        )
        opt.step()

        # EMA update of the target encoder (no grad)
        model.ema_update()

        loss_running = (0.95 * loss_running + 0.05 * loss.item()
                        if step > 1 else loss.item())

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  "
                  f"(ema {loss_running:.4f})")

        # collapse diagnostic every 500 steps — if JEPA is collapsing
        # (encoder outputting constants), both context and target
        # embeddings approach zero variance. Print their stds to catch this.
        if step % 500 == 0:
            with torch.no_grad():
                ctx_tokens = model.context_encoder(x[:32])   # small subset
                tgt_tokens = model.target_encoder(x[:32])
                ctx_std = ctx_tokens.std().item()
                tgt_std = tgt_tokens.std().item()
                # embedding std per dim averaged — if <0.1 something's off
                print(f"    collapse check: ctx_std={ctx_std:.3f}  "
                      f"tgt_std={tgt_std:.3f}  "
                      f"(healthy: both > 0.3, similar)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jepa_checkpoint(model, stats, variables, out_path)
    print(f"checkpoint -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=Path, default=DISTR_DEFAULT)
    p.add_argument("--out", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--context-keep-frac", type=float, default=0.6)
    p.add_argument("--target-window-frac", type=float, default=0.25)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    args = p.parse_args()
    train_jepa(
        distributions_path=args.distributions,
        out_path=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        context_keep_frac=args.context_keep_frac,
        target_window_frac=args.target_window_frac,
        ema_momentum=args.ema_momentum,
    )


if __name__ == "__main__":
    main()