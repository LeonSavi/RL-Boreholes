"""Shared training utilities for belief-model training loops.

Functions and constants used by both ``training.py`` (UNetBelief) and
``models/training.py`` (MapBeliefModel) live here to avoid duplication.
All helpers are model-agnostic: they accept ``nn.Module`` rather than a
specific model class, so they work with any model whose ``forward()``
returns ``(B, 1, n_x, n_y)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import GeologicalBeliefDataset
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
            tgt  = normalizer.inverse_tensor(y)

            mse_total  += nn.functional.mse_loss(pred, tgt).item()
            mae_total  += (pred - tgt).abs().mean().item()
            corr_total += pearson_correlation(pred, tgt)
            n_batches  += 1

    return {
        "val_mse":  mse_total  / n_batches,
        "val_mae":  mae_total  / n_batches,
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
    all_tgt:  list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_pred.append(normalizer.inverse_tensor(model(x)).cpu())
            all_tgt.append(normalizer.inverse_tensor(y).cpu())

    preds = torch.cat(all_pred, dim=0)  # (N, 1, n_x, n_y)
    tgts  = torch.cat(all_tgt,  dim=0)  # (N, 1, n_x, n_y)

    result: dict[str, float | int] = {}
    for lo, hi in bins:
        key = f"{lo}_{hi}"
        sel = (counts >= lo) & (counts <= hi)
        n = int(sel.sum().item())
        result[f"n_{key}"] = n
        if n == 0:
            result[f"mse_{key}"]  = float("nan")
            result[f"mae_{key}"]  = float("nan")
            result[f"corr_{key}"] = float("nan")
            continue
        p = preds[sel]
        t = tgts[sel]
        result[f"mse_{key}"]  = nn.functional.mse_loss(p, t).item()
        result[f"mae_{key}"]  = (p - t).abs().mean().item()
        result[f"corr_{key}"] = pearson_correlation(p, t)

    return result


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
    """Save ``n_plots`` side-by-side validation figures to ``plot_dir``."""
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
