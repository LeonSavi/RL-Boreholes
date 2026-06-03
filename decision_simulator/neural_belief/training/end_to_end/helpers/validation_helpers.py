"""Validation helpers for end-to-end map belief experiments.

These helpers work with any model that exposes the same forward signature as
EndToEndMapBeliefTransformer / PatchBoreholeEndToEndMapBeliefTransformer:

    model(boreholes, ore_vals, positions, padding_mask) -> (B, 1, n_x, n_y)

Also contains sequential-step validation helpers for models that consume
GeologicalBeliefDataset (UNet/MapBelief pipelines).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ....training_utils import TargetNormalizer
from ....training_utils import (
    DRILL_BINS,
    pearson_correlation,
    group_metrics,
    no_ore_metrics_from_flat,
)


def validate_e2e_map(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[str, float]:
    """Compute MSE, MAE, and Pearson in ore-value (denormalised) space."""
    model.eval()
    mse_total = mae_total = corr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)

            mse_total += F.mse_loss(pred, tgt_raw).item()
            mae_total += (pred - tgt_raw).abs().mean().item()
            corr_total += pearson_correlation(pred, tgt_raw)
            n_batches += 1

    return {
        "val_mse": mse_total / n_batches,
        "val_mae": mae_total / n_batches,
        "val_corr": corr_total / n_batches,
    }


def validate_e2e_map_by_drill_bins(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    bins: list[tuple[int, int]] | None = None,
) -> dict[str, float | int]:
    """Compute MSE/MAE/Pearson grouped by number of drilled boreholes."""
    if bins is None:
        bins = DRILL_BINS

    model.eval()
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            all_pred.append(normalizer.inverse_tensor(pred_norm).cpu())
            all_tgt.append(normalizer.inverse_tensor(tgt).cpu())
            all_counts.append(batch["drill_counts"])

    preds = torch.cat(all_pred, dim=0)
    tgts = torch.cat(all_tgt, dim=0)
    counts = torch.cat(all_counts, dim=0)

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
        m = group_metrics(preds[sel], tgts[sel])
        result[f"mse_{key}"] = m["mse"]
        result[f"mae_{key}"] = m["mae"]
        result[f"corr_{key}"] = m["corr"]

    return result


def validate_no_ore_e2e_map(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """False-positive metrics on samples whose true ore map is entirely zero."""
    model.eval()
    all_pred_flat: list[torch.Tensor] = []
    n_no_ore = 0

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm)
            tgt_raw = normalizer.inverse_tensor(tgt)

            B = pred.shape[0]
            no_ore = tgt_raw.view(B, -1).sum(dim=1) == 0
            if not no_ore.any():
                continue

            p = pred[no_ore].cpu()
            n = p.shape[0]
            n_no_ore += n
            all_pred_flat.append(p.view(n, -1))

    if n_no_ore == 0:
        return {
            "no_ore_n": 0,
            "no_ore_pred_total": float("nan"),
            "no_ore_pred_max": float("nan"),
            "no_ore_fp_area": float("nan"),
        }

    return {
        "no_ore_n": n_no_ore,
        **no_ore_metrics_from_flat(torch.cat(all_pred_flat), threshold),
    }


def validate_e2e_map_by_step(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> dict[int, dict[str, float]]:
    """MSE/MAE/Pearson grouped by exact drill count K.

    In sequential mode K equals the prefix step, giving per-step metrics.
    In random mode K is drawn from [min_drills, max_drills].
    """
    model.eval()
    groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))

    with torch.no_grad():
        for batch in val_loader:
            bh = batch["boreholes"].to(device)
            ov = batch["ore_vals"].to(device)
            pos = batch["positions"].to(device)
            pm = batch["padding_mask"].to(device)
            tgt = batch["target_map"].to(device)

            pred_norm = model(bh, ov, pos, pm)
            pred = normalizer.inverse_tensor(pred_norm).cpu()
            tgt_raw = normalizer.inverse_tensor(tgt).cpu()

            for k, p, t in zip(batch["drill_counts"].tolist(), pred, tgt_raw):
                groups[int(k)][0].append(p.unsqueeze(0))
                groups[int(k)][1].append(t.unsqueeze(0))

    return {
        k: group_metrics(torch.cat(ps), torch.cat(ts))
        for k, (ps, ts) in sorted(groups.items())
    }


def validate_by_step(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    device: str,
    batch_size: int = 64,
) -> dict[int, dict[str, float | int]]:
    """Validation metrics grouped by sequential drill step (GeologicalBeliefDataset).

    Returns an empty dict when ``val_ds.metadata`` is not set.
    Metrics are in ore-value space (denormalised).
    """
    if getattr(val_ds, "metadata", None) is None:
        return {}

    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_pred.append(normalizer.inverse_tensor(model(x)).cpu())
            all_tgt.append(normalizer.inverse_tensor(y).cpu())

    preds = torch.cat(all_pred, dim=0)
    tgts = torch.cat(all_tgt, dim=0)
    steps = torch.tensor([m["step"] for m in val_ds.metadata], dtype=torch.long)

    return {
        step: group_metrics(preds[steps == step], tgts[steps == step])
        for step in sorted(set(steps.tolist()))
    }


def validate_no_ore_by_step(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    device: str,
    threshold: float = 0.05,
    batch_size: int = 64,
) -> dict[int, dict[str, float | int]]:
    """No-ore false-positive metrics grouped by sequential drill step (GeologicalBeliefDataset).

    Returns an empty dict when ``val_ds.metadata`` is not set.
    """
    if getattr(val_ds, "metadata", None) is None:
        return {}

    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    all_pred: list[torch.Tensor] = []
    all_tgt: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_pred.append(normalizer.inverse_tensor(model(x)).cpu())
            all_tgt.append(normalizer.inverse_tensor(y).cpu())

    preds = torch.cat(all_pred, dim=0)
    tgts = torch.cat(all_tgt, dim=0)
    steps = torch.tensor([m["step"] for m in val_ds.metadata], dtype=torch.long)
    no_ore_mask = tgts.view(tgts.shape[0], -1).sum(dim=1) == 0

    result: dict[int, dict[str, float | int]] = {}
    for step in sorted(set(steps.tolist())):
        sel = (steps == step) & no_ore_mask
        n = int(sel.sum().item())
        if n == 0:
            result[step] = {
                "no_ore_n": 0,
                "no_ore_pred_total": float("nan"),
                "no_ore_pred_max": float("nan"),
                "no_ore_fp_area": float("nan"),
            }
        else:
            result[step] = {
                "no_ore_n": n,
                **no_ore_metrics_from_flat(preds[sel].view(n, -1), threshold),
            }

    return result
