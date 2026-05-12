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
import cProfile
import pickle
import pstats
import queue
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


from simulator import FormationGeometry, DiscoveryPrior

from simulator.distributions import DistributionBank
from simulator.map_generator import MapGenerator, SimConfig
from encoder.autoencoder import (
    BoreholeAutoencoder, AEConfig, save_checkpoint,
    standardise,
)


class _Prefetcher:
    """Background-thread prefetcher.  Pulls from `iterator` and buffers
    up to `max_ahead` items so the GPU never waits on CPU map generation.
    Daemon thread; auto-stops at process exit.
    """

    def __init__(self, iterator, max_ahead: int = 4):
        self._q: queue.Queue = queue.Queue(maxsize=max_ahead)
        self._iter = iterator
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        try:
            for item in self._iter:
                self._q.put(item)
        except Exception as e:           # surface in main thread
            self._q.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        return item


class EarlyStopper:
    """Stop training when the smoothed loss stops improving.

    Re-evaluated every `check_every` steps.  If the smoothed loss hasn't
    improved by at least `min_delta` for `patience` consecutive checks
    after the `min_steps` warmup, .step() returns True.  Setting
    patience <= 0 disables early stopping entirely.

    After each .step() call, `improved` is True iff the just-recorded
    check beat the previous best — callers use this to snapshot the
    best model state.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4,
                 check_every: int = 50, min_steps: int = 500):
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.min_steps = min_steps
        self.best: float = float("inf")
        self.best_step: int = 0
        self.stale_checks: int = 0
        self.last_check_step: int = 0
        self.improved: bool = False
        self.enabled: bool = patience > 0

    def step(self, current_step: int, smoothed_loss: float) -> bool:
        self.improved = False
        if not self.enabled:
            return False
        if current_step - self.last_check_step < self.check_every:
            return False
        self.last_check_step = current_step
        if current_step < self.min_steps:
            # still in warmup; update best silently so we have a baseline
            if smoothed_loss < self.best:
                self.best = smoothed_loss
                self.best_step = current_step
            return False
        if smoothed_loss < self.best - self.min_delta:
            self.best = smoothed_loss
            self.best_step = current_step
            self.stale_checks = 0
            self.improved = True
            return False
        self.stale_checks += 1
        return self.stale_checks >= self.patience


def _underlying(m: nn.Module) -> nn.Module:
    """Peel a torch.compile wrapper so state_dict ops touch the real module."""
    return getattr(m, "_orig_mod", m)


# Batches-per-map: each map yields n_x * n_y = 1024 boreholes, and we draw
# batch_size at a time.  Used to convert "number of maps" to "number of
# training steps" in the dataset-driven path.
def batches_per_map(batch_size: int, n_x: int = 32, n_y: int = 32) -> int:
    return max(1, (n_x * n_y) // batch_size)


def _refine_stats_from_dir(
    map_files: list[Path],
    old_stats: dict[str, tuple[float, float]],
    variables: list[str],
    n_maps: int = 100,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Recompute per-variable mean/std from a sample of pre-saved maps.

    Saved .npy files are already standardised with `old_stats` (computed
    at dataset-build time from a small sample). To recover raw values we
    un-standardise (raw = std_val * old_std + old_mean), then take fresh
    mean/std over a larger sample. NaNs were already replaced with 0 at
    save time, so we filter only finite values to match the original
    stats discipline.
    """
    rng = np.random.default_rng(seed)
    n = min(n_maps, len(map_files))
    pick = rng.choice(len(map_files), size=n, replace=False)
    accum = {v: [] for v in variables}
    for idx in pick:
        bh = np.load(map_files[idx]).astype(np.float32)   # (N, V, D) standardised
        for i, v in enumerate(variables):
            old_mean, old_std = old_stats[v]
            raw = bh[:, i, :] * old_std + old_mean
            finite = raw[np.isfinite(raw)]
            accum[v].append(finite)
    new_stats: dict[str, tuple[float, float]] = {}
    for v in variables:
        if accum[v]:
            arr = np.concatenate(accum[v])
            new_stats[v] = (float(arr.mean()), float(arr.std()))
        else:
            new_stats[v] = old_stats[v]
    return new_stats


def stream_batches_from_dir(
    dataset_dir: Path,
    batch_size: int,
    device: str,
    maps_per_refill: int = 8,
    seed: int = 0,
    recompute_stats_n_maps: int | None = None,
) -> tuple[Iterator[torch.Tensor], dict, list[str]]:
    """Yield training batches from a directory produced by
    pull_maps.py.  Same shuffling discipline as
    stream_batches(): pool `maps_per_refill` maps, shuffle pooled
    boreholes, emit full-size batches, then refill.  Infinite stream —
    after a full pass through the dataset, the map order reshuffles.

    If `recompute_stats_n_maps` is given, re-derive per-variable mean/std
    from that many sampled maps (un-standardising first), then apply an
    on-the-fly re-standardisation to every batch so the model sees data
    standardised by the refined stats.
    """
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "stats.pkl", "rb") as f:
        stats = pickle.load(f)
    with open(dataset_dir / "config.pkl", "rb") as f:
        cfg = pickle.load(f)
    variables = list(cfg["variables"])

    map_files = sorted(dataset_dir.glob("boreholes_*.npy"))
    if not map_files:
        raise RuntimeError(
            f"no boreholes_*.npy files in {dataset_dir} "
            f"(did you run pull_maps.py?)"
        )
    print(f"  found {len(map_files):,} pre-generated maps in {dataset_dir}")

    # Optional stats refinement -----------------------------------------
    # File values are stored as v_old = (raw - old_mean) / old_std.
    # We want the streamed batch as v_new = (raw - new_mean) / new_std,
    # i.e. v_new = v_old * a + b with
    #   a = old_std / new_std
    #   b = (old_mean - new_mean) / new_std
    transform_a = None
    transform_b = None
    if recompute_stats_n_maps is not None:
        print(f"  refining standardisation stats from "
              f"{min(recompute_stats_n_maps, len(map_files))} sampled maps...")
        new_stats = _refine_stats_from_dir(
            map_files, stats, variables,
            n_maps=recompute_stats_n_maps, seed=seed + 1,
        )
        transform_a = np.array(
            [stats[v][1] / max(new_stats[v][1], 1e-8) for v in variables],
            dtype=np.float32,
        ).reshape(1, -1, 1)
        transform_b = np.array(
            [(stats[v][0] - new_stats[v][0]) / max(new_stats[v][1], 1e-8)
             for v in variables],
            dtype=np.float32,
        ).reshape(1, -1, 1)
        stats = new_stats

    rng = np.random.default_rng(seed)

    def _iter():
        while True:
            perm = rng.permutation(len(map_files))
            for i in range(0, len(perm), maps_per_refill):
                chunk = perm[i:i + maps_per_refill]
                pooled = []
                for j in chunk:
                    bh = np.load(map_files[j]).astype(np.float32)
                    pooled.append(bh)
                pooled = np.concatenate(pooled, axis=0)
                if transform_a is not None:
                    pooled = pooled * transform_a + transform_b
                inner = rng.permutation(len(pooled))
                pooled = pooled[inner]
                n_full = len(pooled) // batch_size
                for k in range(n_full):
                    batch = pooled[k * batch_size : (k + 1) * batch_size]
                    yield torch.from_numpy(batch).to(device)

    return _iter(), stats, variables


PROFILE_STEPS = 200


class _TrainLog:
    """Step-indexed training log.  Per call merges fields into the row
    for that step so a CSV ends up with one row per logged step and one
    column per metric, with blanks where a metric wasn't recorded.

    On finalise(): writes <stem>_log.csv next to the checkpoint and
    renders <stem>_log.png — a two-panel chart with the loss curve up
    top and auxiliary metrics underneath.
    """

    def __init__(self, out_path: Path):
        stem = out_path.stem
        self.csv_path = out_path.with_name(f"{stem}_log.csv")
        self.png_path = out_path.with_name(f"{stem}_log.png")
        self._rows: dict[int, dict] = {}
        self._fields: list[str] = ["step"]

    def log(self, step: int, **fields) -> None:
        row = self._rows.setdefault(step, {"step": step})
        row.update(fields)
        for k in fields:
            if k not in self._fields:
                self._fields.append(k)

    def save_csv(self) -> None:
        import csv
        if not self._rows:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._rows.values(), key=lambda r: r["step"])
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            for r in ordered:
                w.writerow(r)
        print(f"  training log -> {self.csv_path}")

    def plot(self, title: str = "", best_step: int | None = None) -> None:
        import matplotlib.pyplot as plt
        if not self._rows:
            return
        ordered = sorted(self._rows.values(), key=lambda r: r["step"])
        steps_all = [r["step"] for r in ordered]
        loss_fields = [f for f in self._fields
                       if f in ("loss", "loss_ema")]
        extra_fields = [f for f in self._fields
                        if f not in ("step",) + tuple(loss_fields)]

        n_panels = 1 + (1 if extra_fields else 0)
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(10, 3.2 * n_panels),
                                 sharex=True)
        if n_panels == 1:
            axes = [axes]

        ax0 = axes[0]
        for fld in loss_fields:
            ys = [r.get(fld, float("nan")) for r in ordered]
            ax0.plot(steps_all, ys, label=fld,
                     lw=1.0 if fld != "loss_ema" else 1.6)
        ax0.set_ylabel("loss")
        ax0.grid(alpha=0.3)
        if loss_fields:
            ax0.legend(loc="upper right")
        if title:
            ax0.set_title(title)
        if best_step is not None:
            ax0.axvline(best_step, color="grey", lw=0.8,
                        ls="--", label=f"best @ {best_step}")

        if extra_fields:
            ax1 = axes[1]
            for fld in extra_fields:
                pts = [(r["step"], r[fld])
                       for r in ordered
                       if fld in r and r[fld] == r[fld]]   # filter NaN
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax1.plot(xs, ys, marker="o", ms=2.5, lw=0.9,
                         label=fld)
            ax1.set_ylabel("aux metrics")
            ax1.grid(alpha=0.3)
            ax1.legend(fontsize=7, ncol=min(len(extra_fields), 4),
                       loc="upper right")

        axes[-1].set_xlabel("step")
        self.png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  training plot -> {self.png_path}")


def _dump_profile(pr: cProfile.Profile, dump_path: Path | None,
                  top_n: int = 20) -> None:
    """Dump pstats file and print top hot functions.

    Two views: cumulative time finds the slow call sites, self time
    (tottime) finds the slow primitives.  Open the .prof file with
    snakeviz / pstats for an interactive view.
    """
    stats = pstats.Stats(pr)
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        stats.dump_stats(str(dump_path))
        print(f"\nprofile dumped to {dump_path}")
    print(f"\n=== top {top_n} by cumulative time ===")
    stats.sort_stats("cumulative").print_stats(top_n)
    print(f"=== top {top_n} by self (tottime) ===")
    stats.sort_stats("tottime").print_stats(top_n)


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

    # Always-on free GPU speedups: TF32 matmuls (Ampere+) + cuDNN autotuner.
    # AMP/autocast is opt-in: for small models it can add more cast overhead
    # than it saves; benchmark before enabling at scale.
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    use_amp = amp and on_cuda
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else (torch.float16 if use_amp else torch.float32))
    print(f"AMP: {'bf16' if (use_amp and amp_dtype is torch.bfloat16) else ('fp16' if use_amp else 'off')}  "
          f"(TF32: {'on' if on_cuda else 'off'})")

    sim_cfg = SimConfig()
    use_dataset = dataset_dir is not None
    if use_dataset:
        # Pre-generated dataset path: stats and variables come from the
        # dataset directory (saved by pull_maps.py), then
        # refined from 100 sampled maps to tighten the standardisation
        # before training.  Bank / geom / prior aren't needed here.
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

    # model
    ae_cfg = AEConfig(
        n_variables=len(variables),
        n_depth=sim_cfg.n_depth,
        latent_dim=latent_dim,
        mask_prob=mask_prob,
    )
    model = BoreholeAutoencoder(ae_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if compile_model and on_cuda:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    # SmoothL1 (Huber with δ=1) is more robust to heavy-tailed residuals
    # than MSE.  MSE would punish a ±4σ error 16× more than a ±1σ error;
    # Huber switches to linear beyond δ=1, so it's only 4× more.  Combined
    # with the ±4σ winsorisation in standardise(), this stabilises the
    # loss surface enough for the optimiser to descend instead of bouncing.
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # background prefetcher overlaps CPU map generation (or disk reads,
    # in the --dataset-dir path) with GPU compute.
    batches = _Prefetcher(raw_batches, max_ahead=4)

    # fp16 needs a GradScaler; bf16 / fp32 do not
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None

    if profile:
        steps = PROFILE_STEPS
        print(f"profiling mode: forcing steps={steps}, early stop disabled")

    # Early-stop warmup: by default 1/10 of training (clipped 50-500 steps).
    # When --min-maps-warmup N is given, instead translate "N maps seen" into
    # the equivalent number of optimiser steps via batches_per_map.
    if min_maps_warmup is not None:
        bpm = batches_per_map(batch_size, sim_cfg.n_x, sim_cfg.n_y)
        es_min_steps = max(50, min_maps_warmup * bpm)
        print(f"early-stop warmup pinned to {min_maps_warmup} maps × "
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
        x = next(batches)
        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            recon, z = model(x)
            # reconstruction target is the UNMASKED input, even if the encoder
            # saw a masked version.  This forces the latent to encode enough
            # info to reconstruct the full profile from partial observations.
            loss = loss_fn(recon, x)
        opt.zero_grad(set_to_none=True)
        # gradient clipping: prevents catastrophic updates on batches with
        # outlier values (heavy-tailed variables like sp_mv and
        # res_deep_log would occasionally send the loss to 2+ and
        # destabilise training without this).
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()

        loss_running = 0.95 * loss_running + 0.05 * loss.item() if step > 1 else loss.item()

        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  (ema {loss_running:.4f})")
            train_log.log(step, loss=loss.item(), loss_ema=loss_running)

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
            train_log.log(step, **{f"loss_{v}": l for v, l in per_var})

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
    save_checkpoint(_underlying(model), stats, variables, out_path)
    print(f"checkpoint -> {out_path}")

    train_log.save_csv()
    train_log.plot(
        title=f"autoencoder training — {last_step} steps "
              f"(best ema {stopper.best:.4f} @ {stopper.best_step})",
        best_step=stopper.best_step if stopper.best_step else None,
    )


DISTR_DEFAULT = Path("data/clean/distributions.pkl")
CHECKPOINT_DEFAULT = Path("checkpoints/ae.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", type=Path, default=DISTR_DEFAULT,
                   help=f"path to fitted DistributionBank (default: {DISTR_DEFAULT})")
    p.add_argument("--out", type=Path, default=CHECKPOINT_DEFAULT,
                   help=f"path to save checkpoint (default: {CHECKPOINT_DEFAULT})")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--mask-prob", type=float, default=0.3)
    p.add_argument("--amp", action="store_true",
                   help="enable bf16/fp16 autocast (helps mostly at larger "
                        "batch_size / latent_dim; benchmark first)")
    p.add_argument("--compile", dest="compile_model", action="store_true",
                   help="wrap model in torch.compile")
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
                   help="path to pre-generated dataset directory "
                        "(produced by pull_maps.py). "
                        "If set, skips online map generation and streams "
                        "saved boreholes from disk. Pass an empty string "
                        "('') to force online generation.")
    p.add_argument("--min-maps-warmup", type=int, default=3500,
                   help="pin the early-stop warmup to the number of "
                        "optimiser steps needed to consume this many maps "
                        "(overrides the default steps//10 heuristic).")
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
    train(
        distributions_path=args.distributions,
        out_path=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        mask_prob=args.mask_prob,
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