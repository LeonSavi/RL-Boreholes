from __future__ import annotations

import csv
import dataclasses
import json
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
from .utils import LatentNormalizer, LatentPCAReducer, TargetNormalizer
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
    # in_channels must equal 2 + latent_dim:
    #   ch 0   : sparse ore map
    #   ch 1   : observation mask
    #   ch 2.. : JEPA latent vector (latent_dim channels)
    in_channels: int = 130
    base_channels: int = 64

    # --- target normalization ---
    norm_mode: str = "log1p"  # "log1p" | "zscore" | "none"

    # --- optimisation ---
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50

    # --- latent normalization ---
    latent_norm_mode: str = "none"  # "zscore" | "none"

    # --- latent PCA ---
    use_latent_pca: bool = False
    latent_pca_components: int = 32

    # --- coordinate channels ---
    use_coordinate_channels: bool = False

    # --- misc ---
    seed: int = 42
    latent_dim: int = 128
    borehole_encoder: str = "unknown"

    def __post_init__(self) -> None:
        expected = 2 + self.latent_dim
        if self.in_channels != expected:
            raise ValueError(
                f"NeuralBeliefTrainingConfig: in_channels={self.in_channels} does not "
                f"match 2 + latent_dim = {expected}. "
                f"UNetBelief input layout is [ore_map, mask, JEPA_latent], so "
                f"in_channels must always equal 2 + latent_dim. "
                f"Either set in_channels={expected} or latent_dim={self.in_channels - 2}."
            )


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


_DRILL_BINS: list[tuple[int, int]] = [(1, 3), (4, 8), (9, 15)]


def validate_by_drill_bins(
    model: UNetBelief,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    device: str,
    bins: list[tuple[int, int]] | None = None,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Compute val metrics grouped by number of drilled boreholes.

    Returns a flat dict with keys ``mse_<lo>_<hi>``, ``mae_<lo>_<hi>``,
    ``corr_<lo>_<hi>``, and ``n_<lo>_<hi>`` for each bin ``(lo, hi)``.
    Returns an empty dict when ``val_ds.drill_counts`` is not set.
    Metrics are in ore-value space (predictions and targets are denormalized).
    """
    if val_ds.drill_counts is None:
        return {}

    if bins is None:
        bins = _DRILL_BINS

    model.eval()
    counts = val_ds.drill_counts  # (N,) int64, on CPU

    # Full inference pass — collect all predictions and targets
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = normalizer.inverse_tensor(model(x)).cpu()
            tgt = normalizer.inverse_tensor(y).cpu()
            all_pred.append(pred)
            all_tgt.append(tgt)

    preds = torch.cat(all_pred, dim=0)   # (N, 1, n_x, n_y)
    tgts = torch.cat(all_tgt, dim=0)    # (N, 1, n_x, n_y)

    result: dict[str, float | int] = {}
    for lo, hi in bins:
        key = f"{lo}_{hi}"
        sel = (counts >= lo) & (counts <= hi)
        n = int(sel.sum().item())
        result[f"n_{key}"] = n
        if n == 0:
            result[f"mse_{key}"] = float("nan")
            result[f"mae_{key}"] = float("nan")
            result[f"corr_{key}"] = float("nan")
            continue
        p = preds[sel]
        t = tgts[sel]
        result[f"mse_{key}"] = nn.functional.mse_loss(p, t).item()
        result[f"mae_{key}"] = (p - t).abs().mean().item()
        result[f"corr_{key}"] = _pearson_correlation(p, t)

    return result


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
    train_ds: GeologicalBeliefDataset | None = None,
    val_ds: GeologicalBeliefDataset | None = None,
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

    # ---- latent normalization -------------------------------------------------
    latent_normalizer: LatentNormalizer | None = None

    if cfg.latent_dim > 0 and cfg.latent_norm_mode != "none":
        # Collect observed latent vectors from training inputs.
        # Only cells where mask==1 contributed real encoder output;
        # zero-filled unobserved cells must not bias the statistics.
        inputs_np = train_ds.inputs.numpy()      # (N, 2+latent_dim, n_x, n_y)
        obs_mask = inputs_np[:, 1, :, :] > 0.0  # (N, n_x, n_y) bool
        latent_t = inputs_np[:, 2:, :, :].transpose(0, 2, 3, 1)  # (N, n_x, n_y, latent_dim)
        observed = latent_t[obs_mask]            # (N_obs, latent_dim)

        latent_normalizer = LatentNormalizer(mode=cfg.latent_norm_mode)
        latent_normalizer.fit(observed)

        train_ds.apply_latent_normalizer(latent_normalizer)
        val_ds.apply_latent_normalizer(latent_normalizer)

        lnorm_path = checkpoint_dir / "latent_norm_stats.json"
        d = latent_normalizer.to_dict()
        d["encoder_variant"] = cfg.borehole_encoder
        d["n_observed_samples"] = int(observed.shape[0])
        with open(lnorm_path, "w") as f:
            json.dump(d, f, indent=2)

        if verbose:
            assert latent_normalizer.mean is not None and latent_normalizer.std is not None
            print(
                f"  latent norm   : {cfg.latent_norm_mode}"
                f"  (fitted on {observed.shape[0]:,} observed cells)"
            )
            print(
                f"  latent mean   : min={latent_normalizer.mean.min():.4f}"
                f"  max={latent_normalizer.mean.max():.4f}"
            )
            print(
                f"  latent std    : min={latent_normalizer.std.min():.4f}"
                f"  max={latent_normalizer.std.max():.4f}"
            )
            print(f"  latent stats  -> {lnorm_path}")
    else:
        if verbose:
            if cfg.latent_dim == 0:
                print("  latent norm   : skipped (no encoder)")
            else:
                print(f"  latent norm   : {cfg.latent_norm_mode}")

    # ---- latent PCA reduction ------------------------------------------------
    pca_reducer: LatentPCAReducer | None = None

    if cfg.use_latent_pca and cfg.latent_dim > 0:
        original_latent_dim = cfg.latent_dim

        # Fit PCA only on observed (drilled) training latents.
        inputs_np = train_ds.inputs.numpy()       # (N, 2+D, n_x, n_y)
        obs_mask  = inputs_np[:, 1, :, :] > 0.0  # (N, n_x, n_y) bool
        latent_t  = inputs_np[:, 2:, :, :].transpose(0, 2, 3, 1)  # (N, n_x, n_y, D)
        observed  = latent_t[obs_mask]            # (N_obs, D)

        pca_reducer = LatentPCAReducer(n_components=cfg.latent_pca_components)
        pca_reducer.fit(observed)

        train_ds.apply_latent_pca(pca_reducer)
        val_ds.apply_latent_pca(pca_reducer)

        # Update cfg so the model is built with the reduced channel count.
        actual_k = pca_reducer.n_output_components
        cfg.latent_dim   = actual_k
        cfg.in_channels  = 2 + actual_k

        evr = pca_reducer.explained_variance_ratio
        pca_meta = {
            "original_latent_dim": original_latent_dim,
            "n_components": actual_k,
            "explained_variance_ratio": evr.tolist() if evr is not None else [],
            "explained_variance_total": float(evr.sum()) if evr is not None else 0.0,
            "encoder_variant": cfg.borehole_encoder,
            "n_observed_samples": int(observed.shape[0]),
        }
        pca_path = checkpoint_dir / "latent_pca_stats.json"
        with open(pca_path, "w") as f:
            json.dump(pca_meta, f, indent=2)

        # Also persist the fitted reducer so it can be reloaded for inference.
        pca_reducer.save(checkpoint_dir / "latent_pca_reducer.pkl")

        if verbose:
            print(
                f"  PCA reduction : {original_latent_dim} → {actual_k}"
                f"  (fitted on {observed.shape[0]:,} observed cells)"
            )
            if evr is not None:
                print(f"  expl. variance: {evr.sum():.3f}")
            print(f"  in_channels   : {cfg.in_channels}")
            print(f"  PCA stats     -> {pca_path}")
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
                "latent_normalizer": latent_normalizer,
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

    # ---- per-bin validation ---------------------------------------------------
    bin_metrics = validate_by_drill_bins(model, val_ds, normalizer, device)
    if bin_metrics:
        if verbose:
            print("\nValidation metrics by drill count (best model, ore-value space):")
            for lo, hi in _DRILL_BINS:
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


def build_training_config(debug: bool, overrides: dict) -> NeuralBeliefTrainingConfig:
    valid_fields = {f.name for f in dataclasses.fields(NeuralBeliefTrainingConfig)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise ValueError(
            f"Unknown NeuralBeliefTrainingConfig field(s): {sorted(invalid)}.\n"
            f"Valid fields: {sorted(valid_fields)}"
        )

    if debug:
        cfg = NeuralBeliefTrainingConfig(
            n_train_maps=2,
            samples_per_map=2,
            n_val_maps=1,
            val_samples_per_map=2,
            n_epochs=2,
            batch_size=2,
            base_channels=16,
        )
    else:
        cfg = NeuralBeliefTrainingConfig()

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


def export_history(history: list[dict], checkpoint_dir: Path) -> None:
    if not history:
        return

    json_path = checkpoint_dir / "training_history.json"
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {json_path}")

    csv_path = checkpoint_dir / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"History -> {csv_path}")


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
