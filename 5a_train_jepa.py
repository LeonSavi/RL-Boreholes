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
import cProfile
from pathlib import Path

import numpy as np
import torch

from simulator import FormationGeometry, DiscoveryPrior

from simulator.distributions import DistributionBank
from simulator.map_generator import MapGenerator, SimConfig
from train_encoder import (
    compute_standardisation_stats, stream_batches,
    stream_batches_from_dir, batches_per_map,
    _Prefetcher, EarlyStopper, _underlying,
    PROFILE_STEPS, _dump_profile, _TrainLog,
)
from encoder.jepa_encoder import (
    JEPAModel, JEPAConfig, save_jepa_checkpoint,
    sample_context_target_positions,
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
    amp: bool = False,
    compile_model: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4,
    profile: bool = False,
    dataset_dir: Path | None = None,
    min_maps_warmup: int | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Always-on free GPU speedups: TF32 matmuls (Ampere+), cuDNN autotuner.
    # AMP / autocast: opt-in only.  For small models (latent~128, T~30) the
    # autocast cast overhead can outweigh the bf16 matmul savings; benchmark
    # before turning on.  Larger latent_dim or batch_size benefit more.
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    use_amp = amp and on_cuda
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    print(f"AMP: {'bf16' if (use_amp and amp_dtype is torch.bfloat16) else ('fp16' if use_amp else 'off')}  "
          f"(TF32: {'on' if on_cuda else 'off'})")

    sim_cfg = SimConfig()
    use_dataset = dataset_dir is not None
    if use_dataset:
        # Pre-generated dataset path: stats and variables come from the
        # dataset directory (saved by pull_maps.py).  We skip
        # loading the bank / geom / prior since no online generation
        # happens.
        print(f"using pre-generated dataset at: {dataset_dir}")
        raw_batches, stats, variables = stream_batches_from_dir(
            dataset_dir=dataset_dir,
            batch_size=batch_size, device=device,
            maps_per_refill=8, seed=42,
            recompute_stats_n_maps=100,
        )
        for v, (mu, sd) in stats.items():
            print(f"  {v:14s}  mean={mu:8.3f}  std={sd:8.3f}")
    else:
        bank = DistributionBank.load(distributions_path)
        geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
        prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
        variables = list(sim_cfg.variables)
        gen = MapGenerator(bank, geom, sim_cfg, seed=42, prior=prior)
        print(f"computing standardisation stats (10 maps)...")
        stats = compute_standardisation_stats(gen, variables, n_maps=10)
        for v, (m, s) in stats.items():
            print(f"  {v:14s}  mean={m:8.3f}  std={s:8.3f}")
        raw_batches = stream_batches(
            gen, variables, stats, batch_size, device, maps_per_refill=8,
        )

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
    if compile_model and on_cuda:
        # torch.compile gives a one-time graph compile cost (first ~step)
        # then a steady-state speedup on the conv backbone + transformer.
        # Off by default to keep startup fast; enable with --compile.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params_total:,} total "
          f"({n_params_trainable:,} trainable, "
          f"{n_params_total - n_params_trainable:,} frozen target encoder)")

    # Prefetcher overlaps CPU map generation (or disk reads) with GPU
    # compute — typically the dominant speedup on small models where
    # the data path is the bottleneck.
    batches = _Prefetcher(raw_batches, max_ahead=4)

    mask_rng = np.random.default_rng(0)
    T_tokens = model.context_encoder.n_tokens
    print(f"token sequence length (after CNN pooling): {T_tokens}")
    print(f"context tokens per sample: {int(context_keep_frac * T_tokens)}")
    print(f"target  tokens per sample: {int(target_window_frac * T_tokens)}")

    # fp16 needs a GradScaler; bf16 / fp32 do not.
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None
    trainable_params = (list(model.context_encoder.parameters())
                        + list(model.predictor.parameters()))

    if profile:
        steps = PROFILE_STEPS
        print(f"profiling mode: forcing steps={steps}, early stop disabled")

    # Early-stop warmup: by default 1/10 of training (clipped 50-500 steps).
    # When --min-maps-warmup N is provided, instead translate "N maps seen"
    # into the equivalent number of optimiser steps via batches_per_map.
    if min_maps_warmup is not None:
        bpm = batches_per_map(batch_size, sim_cfg.n_x, sim_cfg.n_y)
        es_min_steps = max(50, min_maps_warmup * bpm)
        print(f"early-stop warmup pinned to {min_maps_warmup} maps x "
              f"{bpm} batches/map = {es_min_steps} steps")
    else:
        es_min_steps = min(500, max(50, steps // 10))

    stopper = EarlyStopper(
        patience=patience, min_delta=min_delta,
        check_every=log_every, min_steps=es_min_steps,
    )
    if profile:
        stopper.enabled = False
    best_state: dict | None = None
    print(f"early stopping: {'on' if stopper.enabled else 'off'} "
          f"(patience={patience} checks of {log_every} steps, "
          f"min_delta={min_delta}, warmup={stopper.min_steps})")

    profile_dump = out_path.with_suffix(".prof") if profile else None
    pr = cProfile.Profile() if profile else None
    if pr is not None:
        pr.enable()

    train_log = _TrainLog(out_path)

    loss_running = 0.0
    last_step = 0
    for step in range(1, steps + 1):
        last_step = step
        x = next(batches)                                    # (B, V, D)
        ctx_pos, tgt_pos = sample_context_target_positions(
            n_tokens=T_tokens, batch_size=x.size(0),
            cfg=jepa_cfg, rng=mask_rng,
        )
        ctx_pos = ctx_pos.to(device, non_blocking=True)
        tgt_pos = tgt_pos.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            loss = model(x, ctx_pos, tgt_pos)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            opt.step()

        # EMA update of the target encoder (no grad, fused foreach)
        model.ema_update()

        loss_running = (0.95 * loss_running + 0.05 * loss.item()
                        if step > 1 else loss.item())

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  "
                  f"(ema {loss_running:.4f})")
            train_log.log(step, loss=loss.item(), loss_ema=loss_running)

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
            train_log.log(step, ctx_std=ctx_std, tgt_std=tgt_std)

        # early stopping + best-state snapshotting
        should_stop = stopper.step(step, loss_running)
        if stopper.improved:
            best_state = {k: v.detach().clone()
                          for k, v in _underlying(model).state_dict().items()}
        if should_stop:
            print(f"  early stop at step {step}: no improvement for "
                  f"{stopper.patience} checks "
                  f"(best ema {stopper.best:.4f} at step {stopper.best_step})")
            break

    if pr is not None:
        pr.disable()
        _dump_profile(pr, profile_dump, top_n=20)

    if best_state is not None and (last_step != stopper.best_step or stopper.stale_checks):
        _underlying(model).load_state_dict(best_state)
        print(f"  restored best model state (ema {stopper.best:.4f} @ step {stopper.best_step})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jepa_checkpoint(_underlying(model), stats, variables, out_path)
    print(f"checkpoint -> {out_path}")

    train_log.save_csv()
    train_log.plot(
        title=f"JEPA training - {last_step} steps "
              f"(best ema {stopper.best:.4f} @ {stopper.best_step})",
        best_step=stopper.best_step if stopper.best_step else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=Path, default=DISTR_DEFAULT)
    p.add_argument("--out", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--context-keep-frac", type=float, default=0.6)
    p.add_argument("--target-window-frac", type=float, default=0.25)
    p.add_argument("--ema-momentum", type=float, default=0.996)
    p.add_argument("--amp", action="store_true",
                   help="enable bf16/fp16 autocast (helps mostly at large "
                        "batch_size or latent_dim - can be neutral or slow "
                        "on small configs; benchmark first)")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="wrap model in torch.compile (one-time compile cost, "
                        "then steady-state speedup on the transformer/conv)")
    p.add_argument("--patience", type=int, default=40,
                   help="early-stop after this many checks (of log_every "
                        "steps each) without ema-loss improvement. <=0 disables.")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="minimum ema-loss improvement to count as progress")
    p.add_argument("--profile", action="store_true",
                   help=f"run cProfile over {PROFILE_STEPS} training steps, "
                        "dump .prof file next to checkpoint, print top 20 hot "
                        "functions by cumulative and self time")
    p.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"),
                   help="read pre-generated maps from this directory (output "
                        "of pull_maps.py) instead of generating "
                        "online during training. Pass an empty string ('') "
                        "to force online generation.")
    p.add_argument("--min-maps-warmup", type=int, default=3500,
                   help="block early-stopping until this many maps' worth "
                        "of batches have been consumed.  At batch_size=128 "
                        "and 32x32 maps, one map = 8 batches.")
    args = p.parse_args()
    # Resolve --dataset-dir: empty string = online generation, missing
    # directory = warn and fall back to online so a fresh checkout still works.
    dataset_dir = args.dataset_dir
    if dataset_dir is not None and str(dataset_dir) == "":
        dataset_dir = None
    elif dataset_dir is not None and not dataset_dir.exists():
        print(f"  note: --dataset-dir {dataset_dir} does not exist, "
              f"falling back to online generation.")
        dataset_dir = None
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
        amp=args.amp,
        compile_model=args.compile_model,
        patience=args.patience,
        min_delta=args.min_delta,
        profile=args.profile,
        dataset_dir=dataset_dir,
        min_maps_warmup=args.min_maps_warmup,
    )


if __name__ == "__main__":
    main()