from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulator.map_generator import SimConfig
from decision_simulator.resources import DecisionSimulationResources
from .dataset import BeliefDatasetConfig, GeologicalBeliefDataset
from .model import UNetBelief
from .utils import TargetNormalizer
from .baselines import evaluate_baselines


@dataclass
class NeuralBeliefTrainingConfig:
    """Hyper-parameters for training the neural belief updater."""

    # --- dataset ---
    n_train_maps: int = 50
    samples_per_map: int = 20
    n_val_maps: int = 10
    val_samples_per_map: int = 10
    min_drills: int = 1
    max_drills: int = 15

    # --- model ---
    in_channels: int = 130
    base_channels: int = 64

    # --- target normalization ---
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # --- optimisation ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- misc ---
    seed: int = 42
    latent_dim: int = 128


def _pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-sample Pearson correlation over a batch."""
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    num = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=1e-8)
    return (num / denom).mean().item()


def _validate(
    model: UNetBelief,
    loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Compute val metrics in ore-value space (predictions and targets are denormalized)."""
    model.eval()
    mse_total = mae_total = corr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred_norm = model(x)

            pred = normalizer.inverse_tensor(pred_norm)
            tgt = normalizer.inverse_tensor(y)

            mse_total += nn.functional.mse_loss(pred, tgt).item()
            mae_total += (pred - tgt).abs().mean().item()
            corr_total += _pearson_correlation(pred, tgt)
            n_batches += 1

    return {
        "val_mse": mse_total / n_batches,
        "val_mae": mae_total / n_batches,
        "val_corr": corr_total / n_batches,
    }


def _save_val_plots(
    model: UNetBelief,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 4,
) -> None:
    """Save n_plots side-by-side validation figures to plot_dir."""
    from .visualize import plot_belief_sample

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    indices = np.linspace(0, len(val_ds) - 1, n_plots, dtype=int)

    for k, idx in enumerate(indices):
        inp, tgt = val_ds[int(idx)]
        with torch.no_grad():
            pred_norm = model(inp.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        plot_belief_sample(
            sparse_ore_map=inp[0].numpy(),
            observation_mask=inp[1].numpy(),
            true_ore_map=normalizer.inverse(tgt.squeeze(0).numpy()),
            predicted_ore_map=normalizer.inverse(pred_norm),
            save_path=plot_dir / f"val_sample_{k:02d}.png",
            title=f"Val sample {k}",
        )

    print(f"  plots saved -> {plot_dir}")


def train_neural_belief(
    resources: DecisionSimulationResources,
    cfg: NeuralBeliefTrainingConfig,
    device: str,
    checkpoint_dir: Path,
    sim_cfg: SimConfig | None = None,
    plot_dir: Path | None = None,
    verbose: bool = True,
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
    if verbose:
        print("Generating training dataset ...")
    train_ds = GeologicalBeliefDataset.generate(
        resources,
        BeliefDatasetConfig(
            n_maps=cfg.n_train_maps,
            samples_per_map=cfg.samples_per_map,
            min_drills=cfg.min_drills,
            max_drills=cfg.max_drills,
            latent_dim=cfg.latent_dim,
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
            latent_dim=cfg.latent_dim,
            seed=cfg.seed + 1,
        ),
        device=device,
        sim_cfg=sim_cfg,
        verbose=verbose,
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
            loss = nn.functional.mse_loss(model(x), y)  # loss in normalized space
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_mse_norm = train_loss_sum / n_batches
        val_metrics = _validate(model, val_loader, device, normalizer)

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
        _save_val_plots(model, val_ds, normalizer, plot_dir, device)

    # Reload best weights before returning
    best_ckpt = torch.load(
        checkpoint_dir / "belief_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    if verbose:
        print(f"\nTraining complete. Best val MSE: {best_val_mse:.4f}")
        print(f"  checkpoints -> {checkpoint_dir}")

    return model, normalizer


def load_belief_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[UNetBelief, NeuralBeliefTrainingConfig, TargetNormalizer, list[dict]]:
    """Load a saved belief model checkpoint.

    Returns
    -------
    (model, training_config, normalizer, training_history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg: NeuralBeliefTrainingConfig = ckpt["cfg"]
    model = UNetBelief(in_channels=cfg.in_channels, base_channels=cfg.base_channels).to(
        device
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer: TargetNormalizer = ckpt.get("normalizer", TargetNormalizer(mode="none"))
    return model, cfg, normalizer, ckpt.get("history", [])


def make_debug_config() -> NeuralBeliefTrainingConfig:
    """Return a minimal training config for fast pipeline verification."""
    return NeuralBeliefTrainingConfig(
        n_train_maps=2,
        samples_per_map=3,
        n_val_maps=1,
        val_samples_per_map=3,
        min_drills=2,
        max_drills=6,
        batch_size=4,
        n_epochs=3,
        seed=0,
    )


def debug_run(
    resources: DecisionSimulationResources,
    device: str,
    checkpoint_dir: Path,
    sim_cfg: SimConfig | None = None,
) -> None:
    """Minimal end-to-end pipeline check.

    Verifies dataset generation (with sample validation), training loop,
    checkpoint saving/loading, and inference.  Raises on any failure.
    """
    from .dataset import BeliefDatasetConfig
    from .inference import predict_from_observations

    print("=== Neural Belief Debug Run ===")
    cfg = make_debug_config()

    print("[1/4] Training (3 epochs) ...")
    train_neural_belief(resources, cfg, device, checkpoint_dir, sim_cfg, verbose=True)
    print("  OK")

    print("[2/4] Loading checkpoint ...")
    model2, loaded_cfg, loaded_norm, history = load_belief_checkpoint(
        Path(checkpoint_dir) / "belief_best.pt", device
    )
    assert (
        len(history) == cfg.n_epochs
    ), f"expected {cfg.n_epochs} epochs, got {len(history)}"
    print(f"  OK  ({len(history)} epochs in history)")

    print("[3/4] Inference ...")
    obs = [
        {
            "location": (5, 5),
            "latent": np.ones(loaded_cfg.latent_dim, dtype=np.float32),
            "ore_value": 1.0,
        },
        {
            "location": (10, 12),
            "latent": np.zeros(loaded_cfg.latent_dim, dtype=np.float32),
            "ore_value": 0.0,
        },
    ]
    pred = predict_from_observations(obs, model2, device, normalizer=loaded_norm)
    assert pred.shape == (32, 32), f"expected (32, 32), got {pred.shape}"
    print(f"  OK  output shape: {pred.shape}")

    print("[4/4] Dataset sample validation ...")
    GeologicalBeliefDataset.generate(
        resources,
        BeliefDatasetConfig(
            n_maps=1, samples_per_map=2, min_drills=2, max_drills=4, seed=99
        ),
        device=device,
        sim_cfg=sim_cfg,
        validate_samples=True,
        verbose=False,
    )
    print("  OK  shape/content invariants passed")

    print("\n=== Debug Run PASSED ===")
