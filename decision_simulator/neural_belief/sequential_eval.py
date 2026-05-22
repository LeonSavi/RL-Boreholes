"""Sequential evaluation utilities for belief models trained with ordered drill prefixes.

Functions in this module require ``GeologicalBeliefDataset.metadata`` to be set,
which is the case for datasets built by ``build_sequential_dataset_from_cache``.
They fall back gracefully (empty dict / delegate to ``save_val_plots``) when
metadata is absent, so they are safe to call on non-sequential datasets.
"""
from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import GeologicalBeliefDataset
from .utils import TargetNormalizer
from .training_utils import pearson_correlation, save_val_plots, validate_no_ore


def validate_by_step(
    model: nn.Module,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    device: str,
    batch_size: int = 64,
) -> dict[int, dict[str, float | int]]:
    """Compute validation metrics grouped by sequential drill step.

    Parameters
    ----------
    model       : trained belief model (any architecture)
    val_ds      : validation dataset with ``metadata`` list set
    normalizer  : fitted TargetNormalizer (used for denormalisation)
    device      : torch device string
    batch_size  : inference batch size

    Returns
    -------
    Dict mapping each unique step value to ``{"mse", "mae", "corr", "n"}``.
    Returns an empty dict when ``val_ds.metadata`` is not set.
    Metrics are in ore-value space (denormalised).
    """
    if val_ds.metadata is None:
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

    preds = torch.cat(all_pred, dim=0)  # (N, 1, n_x, n_y)
    tgts = torch.cat(all_tgt, dim=0)   # (N, 1, n_x, n_y)

    steps_tensor = torch.tensor([m["step"] for m in val_ds.metadata], dtype=torch.long)
    unique_steps = sorted(set(steps_tensor.tolist()))

    result: dict[int, dict[str, float | int]] = {}
    for step in unique_steps:
        sel = steps_tensor == step
        n = int(sel.sum().item())
        p = preds[sel]
        t = tgts[sel]
        result[step] = {
            "mse": nn.functional.mse_loss(p, t).item(),
            "mae": (p - t).abs().mean().item(),
            "corr": pearson_correlation(p, t),
            "n": n,
        }

    return result


def validate_no_ore_by_step(
    model: nn.Module,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    device: str,
    threshold: float = 0.05,
    batch_size: int = 64,
) -> dict[int, dict[str, float | int]]:
    """No-ore FP metrics grouped by sequential drill step.

    Returns a dict mapping each step to
    ``{"no_ore_n", "no_ore_pred_total", "no_ore_pred_max", "no_ore_fp_area"}``.
    Returns an empty dict when ``val_ds.metadata`` is not set.
    """
    if val_ds.metadata is None:
        return {}

    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_pred: list[torch.Tensor] = []
    all_tgt:  list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_pred.append(normalizer.inverse_tensor(model(x)).cpu())
            all_tgt.append(normalizer.inverse_tensor(y).cpu())

    preds = torch.cat(all_pred, dim=0)  # (N, 1, H, W)
    tgts  = torch.cat(all_tgt,  dim=0)

    steps_tensor = torch.tensor([m["step"] for m in val_ds.metadata], dtype=torch.long)
    no_ore_global = tgts.view(tgts.shape[0], -1).sum(dim=1) == 0  # (N,)
    unique_steps = sorted(set(steps_tensor.tolist()))

    result: dict[int, dict[str, float | int]] = {}
    for step in unique_steps:
        step_sel = steps_tensor == step
        sel = step_sel & no_ore_global
        n = int(sel.sum().item())
        if n == 0:
            result[step] = {
                "no_ore_n": 0,
                "no_ore_pred_total": float("nan"),
                "no_ore_pred_max":   float("nan"),
                "no_ore_fp_area":    float("nan"),
            }
            continue

        p = preds[sel].view(n, -1)
        result[step] = {
            "no_ore_n":          n,
            "no_ore_pred_total": p.sum(dim=1).mean().item(),
            "no_ore_pred_max":   p.max(dim=1).values.mean().item(),
            "no_ore_fp_area":    (p > threshold).float().mean(dim=1).mean().item(),
        }

    return result


def save_sequential_val_plots(
    model: nn.Module,
    val_ds: GeologicalBeliefDataset,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_sequences: int = 3,
) -> None:
    """Save per-step prediction plots for a selection of validation sequences.

    For each selected sequence, saves:
    * individual PNGs per step: ``step_001.png``, ``step_002.png``, ...
    * a combined ``evolution.png`` grid (rows=steps, cols=4 panels)

    Falls back to ``save_val_plots`` when ``val_ds.metadata`` is not set.

    Parameters
    ----------
    model       : trained belief model
    val_ds      : validation dataset with ``metadata`` list set
    normalizer  : fitted TargetNormalizer
    plot_dir    : parent directory; timestamped sub-dirs are created inside
    device      : torch device string
    n_sequences : number of (map, sequence) pairs to visualise
    """
    if val_ds.metadata is None:
        save_val_plots(model, val_ds, normalizer, plot_dir, device, n_plots=n_sequences)
        return

    from .visualize import plot_belief_sample
    import matplotlib.pyplot as plt

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(plot_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Group sample indices by (map_id, sequence_id)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for sample_idx, meta in enumerate(val_ds.metadata):
        key = (meta["map_id"], meta["sequence_id"])
        groups[key].append((meta["step"], sample_idx))

    # Sort each group by step and select n_sequences groups uniformly
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    all_keys = list(groups.keys())
    n_select = min(n_sequences, len(all_keys))
    indices = np.linspace(0, len(all_keys) - 1, n_select, dtype=int)
    selected_keys = [all_keys[i] for i in indices]

    model.eval()

    for map_id, seq_id in selected_keys:
        seq_dir = base_dir / f"seq_{map_id:04d}_{seq_id:02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        step_pairs = groups[(map_id, seq_id)]  # [(step, sample_idx), ...]
        n_steps = len(step_pairs)

        # Collect panels for the combined evolution figure
        panel_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for step, sample_idx in step_pairs:
            inp, tgt = val_ds[sample_idx]
            with torch.no_grad():
                pred_norm = model(inp.unsqueeze(0).to(device)).squeeze().cpu().numpy()

            sparse_ore = inp[0].numpy()
            obs_mask = inp[1].numpy()
            true_ore = normalizer.inverse(tgt.squeeze(0).numpy())
            pred_ore = normalizer.inverse(pred_norm)

            plot_belief_sample(
                sparse_ore_map=sparse_ore,
                observation_mask=obs_mask,
                true_ore_map=true_ore,
                predicted_ore_map=pred_ore,
                save_path=seq_dir / f"step_{step:03d}.png",
                title=f"Map {map_id} / Seq {seq_id} / Step {step} ({step} drills)",
                timestamp=timestamp,
            )

            panel_rows.append((sparse_ore, obs_mask, true_ore, pred_ore))

        # Single scale across all steps so rows are directly comparable
        global_vmax = max(
            float(max(t.max(), p.max())) for _, _, t, p in panel_rows
        )
        global_vmax = max(global_vmax, 1e-3)

        # Build combined evolution figure: rows=steps, cols=[obs|true|pred|error]
        fig, axes = plt.subplots(n_steps, 4, figsize=(18, 4 * n_steps))
        if n_steps == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Belief evolution — Map {map_id} / Seq {seq_id}"
            f"  (colour scale max = {global_vmax:.3f})",
            fontsize=12,
        )

        col_titles = ["Observations", "True ore map", "Predicted ore map", "Abs error"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        for row_idx, ((step, _), (sparse_ore, obs_mask, true_ore, pred_ore)) in enumerate(
            zip(step_pairs, panel_rows)
        ):
            kw = dict(origin="lower", cmap="viridis", vmin=0, vmax=global_vmax)
            drill_rows, drill_cols = np.where(obs_mask > 0)

            ax = axes[row_idx, 0]
            ax.imshow(sparse_ore.T, **kw)
            ax.scatter(drill_rows, drill_cols, c="red", s=8, marker="x", linewidths=0.6)
            ax.set_ylabel(f"step {step}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 1]
            ax.imshow(true_ore.T, **kw)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 2]
            im_pred = ax.imshow(pred_ore.T, **kw)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 3]
            abs_err = np.abs(pred_ore - true_ore)
            # vmax matches prediction scale so "light pink" and "dark purple"
            # represent the same numerical value across panels
            im_err = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=global_vmax, cmap="Reds")
            # Annotate max error so the reader can see the absolute magnitude
            ax.set_xlabel(f"max err = {abs_err.max():.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        # Shared colourbars — one for prediction (viridis) and one for error (Reds)
        # placed to the right of the last row's panels
        fig.colorbar(im_pred, ax=axes[:, 2], shrink=0.6, label="ore value")
        fig.colorbar(im_err,  ax=axes[:, 3], shrink=0.6, label="abs error")

        fig.tight_layout()
        evo_path = seq_dir / "evolution.png"
        fig.savefig(evo_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  sequential plots -> {base_dir}")
