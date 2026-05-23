"""Training pipeline for UNetBelief.

Trains the convolutional U-Net reconstruction model end-to-end using MSE
loss in normalised target space, with optional latent normalisation, PCA
dimensionality reduction, and spatial coordinate channels.

Checkpoint files: ``belief_best.pt``, ``belief_last.pt``
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulator.map_generator import SimConfig
from decision_simulator.resources import DecisionSimulationResources
from ..datasets import BeliefDatasetConfig, GeologicalBeliefDataset
from ..models.map_encoders.unet_belief import UNetBelief
from ..utils import TargetNormalizer
from ..baselines import evaluate_baselines
from ..training_utils import (
    DRILL_BINS,
    validate_by_drill_bins,
    validate,
    validate_no_ore,
    false_positive_loss,
    save_no_ore_metrics,
    save_val_plots,
    export_history,
    load_model_encoder_checkpoint,
)
from ..dataset_transformations import fit_and_apply_latent_pca
from .training_configs import NeuralBeliefTrainingConfig


def train_neural_belief(
    resources: DecisionSimulationResources,
    cfg: NeuralBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    sim_cfg: SimConfig | None = None,
    plot_dir: Path | None = None,
    verbose: bool = True,
    train_ds: GeologicalBeliefDataset | None = None,
    val_ds: GeologicalBeliefDataset | None = None,
    train_cache=None,
    val_cache=None,
) -> tuple[UNetBelief, TargetNormalizer]:
    """Train the neural geological belief updater end-to-end.

    Saves two checkpoints to ``checkpoint_dir``:
      * ``belief_best.pt``  — lowest validation MSE (ore-value space)
      * ``belief_last.pt``  — final epoch

    Parameters
    ----------
    resources      : JEPA model + map-generation components
    cfg            : training hyper-parameters
    device         : torch device string
    checkpoint_dir : directory for saved checkpoints
    sim_cfg        : map dimensions / ore parameters (defaults to SimConfig())
    plot_dir       : if given, save 4 validation plots here after training
    verbose        : print per-epoch metrics

    Returns
    -------
    (trained UNetBelief with best weights, fitted TargetNormalizer)
    """
    if sim_cfg is None:
        sim_cfg = SimConfig()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- datasets -------------------------------------------------------------
    if train_ds is None or val_ds is None:
        if cfg.use_sequential_dataset:
            from ..datasets import build_sequential_dataset_from_cache
            if train_cache is None or val_cache is None:
                raise ValueError(
                    "train_cache and val_cache must be provided when "
                    "use_sequential_dataset=True"
                )
            if verbose:
                print("Building sequential training dataset ...")
            train_ds = build_sequential_dataset_from_cache(
                train_cache,
                resources,
                device,
                n_sequences_per_map=cfg.n_sequences_per_map,
                max_drills=cfg.max_drills,
                prefix_steps=cfg.prefix_steps,
                seed=cfg.sequential_seed,
                verbose=verbose,
            )
            if verbose:
                print("Building sequential validation dataset ...")
            val_ds = build_sequential_dataset_from_cache(
                val_cache,
                resources,
                device,
                n_sequences_per_map=cfg.n_sequences_per_map,
                max_drills=cfg.max_drills,
                prefix_steps=cfg.prefix_steps,
                seed=cfg.sequential_seed + 1,
                verbose=verbose,
            )
        else:
            if verbose:
                print("Generating training dataset ...")
            train_ds = GeologicalBeliefDataset.generate(
                resources,
                BeliefDatasetConfig(
                    n_maps=cfg.n_train_maps,
                    samples_per_map=cfg.samples_per_map,
                    min_drills=cfg.min_drills,
                    max_drills=cfg.max_drills,
                    seed=cfg.seed,
                ),
                device=device,
                sim_cfg=sim_cfg,
                verbose=verbose,
            )

            if verbose:
                print("Generating validation dataset ...")
            val_ds = GeologicalBeliefDataset.generate(
                resources,
                BeliefDatasetConfig(
                    n_maps=cfg.n_val_maps,
                    samples_per_map=cfg.val_samples_per_map,
                    min_drills=cfg.min_drills,
                    max_drills=cfg.max_drills,
                    seed=cfg.seed + 1,
                ),
                device=device,
                sim_cfg=sim_cfg,
                verbose=verbose,
            )
    else:
        if verbose:
            print("Using pre-built datasets.")
        # Wrap in new objects so attribute reassignments (normalisation, PCA,
        # coordinate channels) do not mutate the caller's datasets.
        train_ds = GeologicalBeliefDataset(
            train_ds.inputs, train_ds.targets, train_ds.drill_counts, train_ds.metadata
        )
        val_ds = GeologicalBeliefDataset(
            val_ds.inputs, val_ds.targets, val_ds.drill_counts, val_ds.metadata
        )

    # ---- target normalization -------------------------------------------------
    normalizer = TargetNormalizer(mode=cfg.norm_mode)
    normalizer.fit(train_ds.targets.numpy())
    train_ds.apply_target_normalizer(normalizer)
    val_ds.apply_target_normalizer(normalizer)

    if verbose:
        if cfg.norm_mode == "zscore":
            print(
                f"  target norm   : zscore  mean={normalizer.mean:.4f}  std={normalizer.std:.4f}"
            )
        elif cfg.norm_mode != "none":
            print(f"  target norm   : {cfg.norm_mode}")

    # ---- latent PCA reduction ------------------------------------------------
    pca_reducer = None

    if cfg.use_latent_pca and cfg.latent_dim > 0:
        pca_reducer, actual_k = fit_and_apply_latent_pca(
            train_ds, val_ds,
            n_components=cfg.latent_pca_components,
            checkpoint_dir=checkpoint_dir,
            borehole_encoder=cfg.borehole_encoder,
            verbose=verbose,
        )
        cfg.latent_dim  = actual_k
        cfg.in_channels = 2 + actual_k
        if verbose:
            print(f"  in_channels   : {cfg.in_channels}")
    else:
        if verbose:
            if cfg.latent_dim == 0:
                print("  PCA reduction : skipped (no encoder)")
            else:
                print("  PCA reduction : disabled")

    # ---- coordinate channels -------------------------------------------------
    if cfg.use_coordinate_channels:
        train_ds.apply_coordinate_channels()
        val_ds.apply_coordinate_channels()
        cfg.in_channels += 2
        if verbose:
            print(f"  coord channels: enabled  →  in_channels={cfg.in_channels}")
    else:
        if verbose:
            print("  coord channels: disabled")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if verbose:
        print(f"  train samples : {len(train_ds)}")
        print(f"  val   samples : {len(val_ds)}")

    # ---- model ----------------------------------------------------------------
    model = UNetBelief(in_channels=cfg.in_channels, base_channels=cfg.base_channels).to(
        device
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if verbose:
        print(f"  model params  : {sum(p.numel() for p in model.parameters()):,}")

    # ---- helpers --------------------------------------------------------------
    def _save_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "history": history,
                "normalizer": normalizer,
                "pca_reducer": pca_reducer,
                "sim_cfg": sim_cfg,
                "n_x": sim_cfg.n_x,
                "n_y": sim_cfg.n_y,
                "latent_dim": cfg.latent_dim,
            },
            path,
        )

    # ---- training loop --------------------------------------------------------
    best_val_mse = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)  # loss in normalized space
            if cfg.use_false_positive_penalty:
                loss = loss + cfg.false_positive_weight * false_positive_loss(pred, y, normalizer)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_mse_norm = train_loss_sum / n_batches
        val_metrics = validate(model, val_loader, device, normalizer)

        row = {"epoch": epoch, "train_mse_norm": train_mse_norm, **val_metrics}
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch:3d}/{cfg.n_epochs}"
                f"  train_mse(norm)={train_mse_norm:.4f}"
                f"  val_mse={val_metrics['val_mse']:.4f}"
                f"  val_mae={val_metrics['val_mae']:.4f}"
                f"  val_corr={val_metrics['val_corr']:.4f}"
            )

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            _save_ckpt(checkpoint_dir / "belief_best.pt", epoch)

    _save_ckpt(checkpoint_dir / "belief_last.pt", cfg.n_epochs)

    # ---- baseline comparison --------------------------------------------------
    if verbose:
        print("\nBaseline comparison (val set, ore-value space):")
        for name, m in evaluate_baselines(val_loader, normalizer).items():
            print(
                f"  {name:22s}  mse={m['mse']:.4f}  mae={m['mae']:.4f}  corr={m['corr']:.4f}"
            )
        best = min(history, key=lambda r: r["val_mse"])
        print(
            f"  {'neural_belief':22s}  mse={best['val_mse']:.4f}"
            f"  mae={best['val_mae']:.4f}  corr={best['val_corr']:.4f}"
            f"  (epoch {best['epoch']})"
        )

    # ---- optional plots -------------------------------------------------------
    if plot_dir is not None:
        if cfg.use_sequential_dataset:
            from ..sequential_eval import save_sequential_val_plots
            save_sequential_val_plots(
                model, val_ds, normalizer, plot_dir, device,
                n_sequences=cfg.n_val_plots,
            )
        else:
            save_val_plots(model, val_ds, normalizer, plot_dir, device, n_plots=cfg.n_val_plots)

    # Reload best weights before returning
    best_ckpt = torch.load(
        checkpoint_dir / "belief_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\nTraining complete. Best val MSE: {best_val_mse:.4f}")
        print(f"  checkpoints -> {checkpoint_dir}")

    # ---- per-bin validation ---------------------------------------------------
    bin_metrics = validate_by_drill_bins(model, val_ds, normalizer, device)
    if bin_metrics:
        if verbose:
            print("\nValidation metrics by drill count (best model, ore-value space):")
            for lo, hi in DRILL_BINS:
                key = f"{lo}_{hi}"
                n = bin_metrics.get(f"n_{key}", 0)
                mse = bin_metrics.get(f"mse_{key}", float("nan"))
                mae = bin_metrics.get(f"mae_{key}", float("nan"))
                corr = bin_metrics.get(f"corr_{key}", float("nan"))
                print(
                    f"  drills {lo:2d}-{hi:2d}"
                    f"  n={n:5d}"
                    f"  mse={mse:.4f}"
                    f"  mae={mae:.4f}"
                    f"  corr={corr:.4f}"
                )
        bin_path = checkpoint_dir / "val_metrics_by_drills.json"
        with open(bin_path, "w") as f:
            json.dump(bin_metrics, f, indent=2)
        if verbose:
            print(f"  drill-bin metrics -> {bin_path}")

    # ---- sequential step metrics -----------------------------------------------
    if cfg.use_sequential_dataset:
        from ..sequential_eval import validate_by_step
        step_metrics = validate_by_step(model, val_ds, normalizer, device)
        if step_metrics:
            if verbose:
                print("\nValidation metrics by step (best model, ore-value space):")
                for step, m in sorted(step_metrics.items()):
                    print(
                        f"  step {step:3d}  n={m['n']:5d}"
                        f"  mse={m['mse']:.4f}  mae={m['mae']:.4f}"
                        f"  corr={m['corr']:.4f}"
                    )
            step_path = checkpoint_dir / "val_metrics_by_step.json"
            with open(step_path, "w") as f:
                json.dump({str(k): v for k, v in step_metrics.items()}, f, indent=2)
            if verbose:
                print(f"  step metrics -> {step_path}")

    # ---- no-ore false-positive metrics -----------------------------------------
    no_ore_metrics = validate_no_ore(
        model, val_loader, device, normalizer,
        threshold=cfg.false_positive_threshold,
    )
    save_no_ore_metrics(no_ore_metrics, checkpoint_dir, verbose=verbose)

    if cfg.use_sequential_dataset:
        from ..sequential_eval import validate_no_ore_by_step
        no_ore_step = validate_no_ore_by_step(
            model, val_ds, normalizer, device,
            threshold=cfg.false_positive_threshold,
        )
        if no_ore_step:
            if verbose:
                print("\nNo-ore metrics by step (best model):")
                for step, m in sorted(no_ore_step.items()):
                    print(
                        f"  step {step:3d}  n={m['no_ore_n']:5d}"
                        f"  pred_total={m['no_ore_pred_total']:.4f}"
                        f"  pred_max={m['no_ore_pred_max']:.4f}"
                        f"  fp_area={m['no_ore_fp_area']:.4f}"
                    )
            no_ore_step_path = checkpoint_dir / "val_metrics_no_ore_by_step.json"
            with open(no_ore_step_path, "w") as f:
                json.dump({str(k): v for k, v in no_ore_step.items()}, f, indent=2)
            if verbose:
                print(f"  no-ore by-step metrics -> {no_ore_step_path}")

    return model, normalizer


def load_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[UNetBelief, NeuralBeliefTrainingConfig, TargetNormalizer, list[dict]]:
    def _model_fn(ckpt: dict) -> UNetBelief:
        cfg = ckpt["cfg"]
        return UNetBelief(in_channels=cfg.in_channels, base_channels=cfg.base_channels)
    return load_model_encoder_checkpoint(path, _model_fn, NeuralBeliefTrainingConfig, device)




def save_experiment_config(
    checkpoint_dir: Path,
    cfg: NeuralBeliefTrainingConfig,
    cache_path: Path | None,
    sim_cfg: SimConfig | None,
) -> None:
    """Save a JSON capturing full experiment provenance next to the checkpoints."""
    record = {
        **dataclasses.asdict(cfg),
        "cache_path": str(cache_path) if cache_path is not None else None,
        "sim_cfg": dataclasses.asdict(sim_cfg) if sim_cfg is not None else None,
    }
    out = Path(checkpoint_dir) / "experiment_config.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  experiment config -> {out}")
