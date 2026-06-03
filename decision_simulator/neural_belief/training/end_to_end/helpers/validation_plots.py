"""Validation plot helpers for end-to-end map belief experiments."""

from __future__ import annotations

import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from decision_simulator.utils.plotting import plot_belief_sample
from ....training_utils import TargetNormalizer


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unpack_model_output(
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    normalizer: TargetNormalizer,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (pred_ore_np, pred_unc_np_or_None) from model output."""
    if isinstance(out, tuple):
        pred_norm, unc_norm = out
        return (
            normalizer.inverse(pred_norm.squeeze().cpu().numpy()),
            unc_norm.squeeze().cpu().numpy(),
        )
    return normalizer.inverse(out.squeeze().cpu().numpy()), None


def _panels_from_e2e_sample(
    model: nn.Module,
    sample: dict,
    normalizer: TargetNormalizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract (sparse_ore, obs_mask, true_ore, pred_ore, pred_unc_or_None) from an E2E sample dict."""
    bh = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
    ov = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
    pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ore, pred_unc = _unpack_model_output(model(bh, ov, pos), normalizer)

    n_x, n_y = sample["target_map"].shape
    sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
    obs_mask = np.zeros((n_x, n_y), dtype=np.float32)
    for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
        i = int(round(float(px) * (n_x - 1)))
        j = int(round(float(py) * (n_y - 1)))
        sparse_ore[i, j] = float(ov_val)
        obs_mask[i, j] = 1.0
    true_ore = normalizer.inverse(sample["target_map"])
    return sparse_ore, obs_mask, true_ore, pred_ore, pred_unc


def _panels_from_geo_sample(
    model: nn.Module,
    val_ds: object,
    idx: int,
    normalizer: TargetNormalizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract (sparse_ore, obs_mask, true_ore, pred_ore, pred_unc_or_None) from a GeologicalBeliefDataset sample."""
    inp, tgt = val_ds[idx]
    with torch.no_grad():
        pred_ore, pred_unc = _unpack_model_output(model(inp.unsqueeze(0).to(device)), normalizer)
    sparse_ore = inp[0].numpy()
    obs_mask = inp[1].numpy()
    true_ore = normalizer.inverse(tgt.squeeze(0).numpy())
    return sparse_ore, obs_mask, true_ore, pred_ore, pred_unc


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def save_e2e_map_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 20,
) -> None:
    """Save n_plots belief-map figures to a timestamped subdirectory.

    Shows an uncertainty panel when the model returns a (pred, uncertainty) tuple.
    """
    matplotlib.use("Agg")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(plot_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    indices = np.linspace(0, len(val_ds) - 1, n_plots, dtype=int)

    for plot_k, idx in enumerate(indices):
        sample = val_ds.samples[int(idx)]
        sparse_ore, obs_mask, true_ore, pred_ore, pred_unc = _panels_from_e2e_sample(
            model, sample, normalizer, device
        )
        plot_belief_sample(
            sparse_ore_map=sparse_ore,
            observation_mask=obs_mask,
            true_ore_map=true_ore,
            predicted_ore_map=pred_ore,
            predicted_uncertainty_map=pred_unc,
            save_path=out_dir / f"val_sample_{plot_k:02d}.png",
            title=f"Val sample {plot_k}  ({sample['drill_count']} drills)",
            timestamp=timestamp,
        )

    print(f"  plots saved -> {out_dir}")


def save_sequential_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_sequences: int = 3,
) -> None:
    """Save per-step belief-evolution plots for sequential validation sequences.

    Works with both E2EMapDataset (samples dict format) and GeologicalBeliefDataset
    (metadata format). Automatically includes an uncertainty panel when the model
    returns a (pred, uncertainty) tuple; uses 4 panels otherwise.

    Falls back to ``save_e2e_map_val_plots`` when no sequential structure is detected.
    """
    # Detect dataset format and build groups: {(map_id, seq_id): [(step, sample_idx), ...]}
    if hasattr(val_ds, "samples") and val_ds.samples and "sequence_id" in val_ds.samples[0]:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for idx, s in enumerate(val_ds.samples):
            key = (int(s["map_idx"]), int(s["sequence_id"]))
            groups[key].append((int(s["drill_count"]), idx))

        def get_panels(idx: int):
            return _panels_from_e2e_sample(model, val_ds.samples[idx], normalizer, device)

    elif getattr(val_ds, "metadata", None) is not None:
        groups = defaultdict(list)
        for idx, meta in enumerate(val_ds.metadata):
            key = (meta["map_id"], meta["sequence_id"])
            groups[key].append((meta["step"], idx))

        def get_panels(idx: int):
            return _panels_from_geo_sample(model, val_ds, idx, normalizer, device)

    else:
        save_e2e_map_val_plots(model, val_ds, normalizer, plot_dir, device, n_plots=n_sequences)
        return

    matplotlib.use("Agg")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(plot_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    all_keys = list(groups.keys())
    n_select = min(n_sequences, len(all_keys))
    sel_indices = np.linspace(0, len(all_keys) - 1, n_select, dtype=int)
    selected_keys = [all_keys[i] for i in sel_indices]

    model.eval()

    for map_id, seq_id in selected_keys:
        seq_dir = base_dir / f"seq_{map_id:04d}_{seq_id:02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        step_pairs = groups[(map_id, seq_id)]
        n_steps = len(step_pairs)
        panel_rows: list[tuple] = []
        has_unc = False

        for step, sample_idx in step_pairs:
            sparse_ore, obs_mask, true_ore, pred_ore, pred_unc = get_panels(sample_idx)
            if pred_unc is not None:
                has_unc = True

            plot_belief_sample(
                sparse_ore_map=sparse_ore,
                observation_mask=obs_mask,
                true_ore_map=true_ore,
                predicted_ore_map=pred_ore,
                predicted_uncertainty_map=pred_unc,
                save_path=seq_dir / f"step_{step:03d}.png",
                title=f"Map {map_id} / Seq {seq_id} / Step {step} ({step} drills)",
                timestamp=timestamp,
            )
            panel_rows.append((sparse_ore, obs_mask, true_ore, pred_ore, pred_unc))

        global_vmax = max(float(max(t.max(), p.max())) for _, _, t, p, _ in panel_rows)
        global_vmax = max(global_vmax, 1e-3)

        n_cols = 5 if has_unc else 4
        fig, axes = plt.subplots(n_steps, n_cols, figsize=(4 * n_cols + 2, 4 * n_steps))
        if n_steps == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Belief evolution — Map {map_id} / Seq {seq_id}"
            f"  (colour scale max = {global_vmax:.3f})",
            fontsize=12,
        )

        col_titles = ["Observations", "True ore map", "Predicted ore map"]
        if has_unc:
            col_titles.append("Uncertainty")
        col_titles.append("Abs error")
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        im_unc: matplotlib.image.AxesImage | None = None

        for row_idx, ((step, _), (sparse_ore, obs_mask, true_ore, pred_ore, pred_unc)) in enumerate(
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

            next_col = 3
            if has_unc and pred_unc is not None:
                ax = axes[row_idx, next_col]
                im_unc = ax.imshow(pred_unc.T, origin="lower", cmap="hot_r")
                ax.scatter(drill_rows, drill_cols, c="blue", s=8, marker="x", linewidths=0.6)
                ax.set_xticks([])
                ax.set_yticks([])
                next_col = 4

            ax = axes[row_idx, next_col]
            abs_err = np.abs(pred_ore - true_ore)
            im_err = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=global_vmax, cmap="Reds")
            ax.set_xlabel(f"max err = {abs_err.max():.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_pred, ax=axes[:, 2], shrink=0.6, label="ore value")
        if has_unc and im_unc is not None:
            fig.colorbar(im_unc, ax=axes[:, 3], shrink=0.6, label="uncertainty")
            fig.colorbar(im_err, ax=axes[:, 4], shrink=0.6, label="abs error")
        else:
            fig.colorbar(im_err, ax=axes[:, 3], shrink=0.6, label="abs error")

        fig.tight_layout()
        fig.savefig(seq_dir / "evolution.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  sequential plots -> {base_dir}")


# Backward-compatible alias
save_e2e_map_sequential_val_plots = save_sequential_val_plots
