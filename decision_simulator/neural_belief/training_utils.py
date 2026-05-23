"""Shared training utilities for belief-model training loops.

Functions and constants used by both ``training.py`` (UNetBelief) and
``models/training.py`` (MapBeliefModel) live here to avoid duplication.
All helpers are model-agnostic: they accept ``nn.Module`` rather than a
specific model class, so they work with any model whose ``forward()``
returns ``(B, 1, n_x, n_y)``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import GeologicalBeliefDataset
from .utils import TargetNormalizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRILL_BINS: list[tuple[int, int]] = [(1, 3), (4, 8), (9, 15)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-sample Pearson correlation over a batch."""
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    num = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=1e-8)
    return (num / denom).mean().item()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Compute validation metrics in ore-value space (predictions and targets denormalised)."""
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
            corr_total += pearson_correlation(pred, tgt)
            n_batches += 1

    return {
        "val_mse": mse_total / n_batches,
        "val_mae": mae_total / n_batches,
        "val_corr": corr_total / n_batches,
    }


def validate_by_drill_bins(
    model: nn.Module,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    device: str,
    bins: list[tuple[int, int]] | None = None,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Compute validation metrics grouped by number of drilled boreholes.

    Returns a flat dict with keys ``mse_<lo>_<hi>``, ``mae_<lo>_<hi>``,
    ``corr_<lo>_<hi>``, and ``n_<lo>_<hi>`` for each bin ``(lo, hi)``.
    Returns an empty dict when ``val_ds.drill_counts`` is not set.
    Metrics are in ore-value space (predictions and targets are denormalised).
    """
    if val_ds.drill_counts is None:
        return {}

    if bins is None:
        bins = DRILL_BINS

    model.eval()
    counts = val_ds.drill_counts  # (N,) int64, on CPU

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_pred.append(normalizer.inverse_tensor(model(x)).cpu())
            all_tgt.append(normalizer.inverse_tensor(y).cpu())

    preds = torch.cat(all_pred, dim=0)  # (N, 1, n_x, n_y)
    tgts = torch.cat(all_tgt, dim=0)  # (N, 1, n_x, n_y)

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
        result[f"corr_{key}"] = pearson_correlation(p, t)

    return result


# ---------------------------------------------------------------------------
# No-ore false-positive metrics
# ---------------------------------------------------------------------------


def validate_no_ore(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """Compute false-positive metrics on val samples where the true ore map is empty.

    Returns
    -------
    Dict with keys:
      ``no_ore_n``          – number of no-ore samples evaluated
      ``no_ore_pred_total`` – mean of pred.sum() across no-ore samples (ore-value space)
      ``no_ore_pred_max``   – mean of pred.max() across no-ore samples (ore-value space)
      ``no_ore_fp_area``    – mean fraction of cells where pred > threshold
    All float metrics are NaN when no no-ore samples are present.
    """
    model.eval()
    pred_totals: list[torch.Tensor] = []
    pred_maxes: list[torch.Tensor] = []
    fp_areas: list[torch.Tensor] = []
    n_no_ore = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = normalizer.inverse_tensor(model(x))  # (B, 1, H, W)
            tgt = normalizer.inverse_tensor(y)  # (B, 1, H, W)

            B = pred.shape[0]
            no_ore = tgt.view(B, -1).sum(dim=1) == 0  # (B,) bool
            if not no_ore.any():
                continue

            p = pred[no_ore].cpu()  # (k, 1, H, W)
            n = p.shape[0]
            n_no_ore += n

            pv = p.view(n, -1)
            pred_totals.append(pv.sum(dim=1))
            pred_maxes.append(pv.max(dim=1).values)
            fp_areas.append((pv > threshold).float().mean(dim=1))

    if n_no_ore == 0:
        return {
            "no_ore_n": 0,
            "no_ore_pred_total": float("nan"),
            "no_ore_pred_max": float("nan"),
            "no_ore_fp_area": float("nan"),
        }

    return {
        "no_ore_n": n_no_ore,
        "no_ore_pred_total": torch.cat(pred_totals).mean().item(),
        "no_ore_pred_max": torch.cat(pred_maxes).mean().item(),
        "no_ore_fp_area": torch.cat(fp_areas).mean().item(),
    }


def false_positive_loss(
    pred_norm: torch.Tensor,
    tgt_norm: torch.Tensor,
    normalizer: TargetNormalizer,
    threshold: float = 0.0,
) -> torch.Tensor:
    """FP penalty: mean of clamp(pred, min=0) restricted to no-ore samples in the batch.

    *threshold* is the maximum summed ore value still considered "no ore".
    For map outputs (B, 1, H, W) the default 0.0 matches the exact-zero check.
    For scalar targets (B, 1) pass the per-sample threshold (e.g. 1e-3).

    Returns a zero scalar when no no-ore samples are present.
    """
    tgt_ore = normalizer.inverse_tensor(tgt_norm)
    B = tgt_ore.shape[0]
    no_ore = tgt_ore.reshape(B, -1).sum(dim=1) <= threshold  # (B,)
    if not no_ore.any():
        return pred_norm.new_zeros(())
    return pred_norm[no_ore].clamp(min=0).mean()


def save_no_ore_metrics(
    metrics: dict[str, float | int],
    checkpoint_dir: Path,
    verbose: bool = True,
) -> None:
    """Save no-ore validation metrics to JSON + CSV and optionally print a summary."""
    n = int(metrics.get("no_ore_n", 0))
    if verbose:
        print(f"\nNo-ore validation (n={n} samples):")
        if n > 0:
            print(f"  pred_total_ore : {metrics['no_ore_pred_total']:.4f}")
            print(f"  pred_max_ore   : {metrics['no_ore_pred_max']:.4f}")
            print(f"  fp_area        : {metrics['no_ore_fp_area']:.4f}")
        else:
            print("  (no no-ore samples found in validation set)")

    json_path = Path(checkpoint_dir) / "val_metrics_no_ore.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = Path(checkpoint_dir) / "val_metrics_no_ore.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    if verbose:
        print(f"  no-ore metrics -> {json_path}")


def save_checkpoint_model(
    path: Path,
    model: nn.Module,
    cfg: Any,
    epoch: int,
    history: list[dict],
    normalizer: TargetNormalizer,
    **extra: Any,
) -> None:
    """Save a model checkpoint to *path*.

    Common fields (state_dict, cfg, epoch, history, normalizer) are always
    written. Pass any model-specific extras as keyword arguments (e.g.
    ``model_cfg=…``, ``pca_reducer=…``, ``n_x=…``).
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "history": history,
            "normalizer": normalizer,
            **extra,
        },
        path,
    )


def load_model_encoder_checkpoint(
    path: Path,
    model_fn: Callable[[dict], nn.Module],
    cfg_class: type,
    device: str = "cpu",
) -> tuple[nn.Module, Any, TargetNormalizer, list[dict]]:
    """Load a model checkpoint generically.

    *model_fn* receives the raw checkpoint dict and must return an un-moved,
    un-eval'd ``nn.Module``.  The function moves the model to *device* and
    sets eval mode before returning.  *cfg_class* is used as the default when
    the checkpoint has no ``"cfg"`` key.

    Returns
    -------
    (model, cfg, normalizer, history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_fn(ckpt).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer: TargetNormalizer = ckpt.get("normalizer", TargetNormalizer(mode="none"))
    cfg = ckpt.get("cfg", cfg_class())
    return model, cfg, normalizer, ckpt.get("history", [])


def build_training_config(cfg_class: type, overrides: dict | None = None):
    """Instantiate *cfg_class* with optional field overrides, validating keys."""
    overrides = overrides or {}
    valid_fields = {f.name for f in dataclasses.fields(cfg_class)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise ValueError(
            f"Unknown {cfg_class.__name__} field(s): {sorted(invalid)}.\n"
            f"Valid fields: {sorted(valid_fields)}"
        )
    cfg = cfg_class()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def save_experiment_config(
    checkpoint_dir: Path,
    cfg: Any,
    cache_path: Path | None,
) -> None:
    """Save a JSON capturing full experiment provenance next to the checkpoints."""
    record = {
        **dataclasses.asdict(cfg),
        "cache_path": str(cache_path) if cache_path is not None else None,
    }
    out = Path(checkpoint_dir) / "experiment_config.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  experiment config -> {out}")


def export_history(history: list[dict], directory: Path) -> None:
    """Write training history to JSON and CSV."""
    if not history:
        return
    json_path = directory / "training_history.json"
    csv_path = directory / "training_history.csv"
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def save_val_plots(
    model: nn.Module,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 4,
) -> None:
    """Save ``n_plots`` side-by-side validation figures to a timestamped subdirectory of ``plot_dir``."""
    import datetime
    from .visualize import plot_belief_sample

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = Path(plot_dir) / timestamp
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
            timestamp=timestamp,
        )

    print(f"  plots saved -> {plot_dir}")


