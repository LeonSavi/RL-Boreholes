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
from ....utils import TargetNormalizer


def save_e2e_map_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_plots: int = 20,
) -> None:
    """Save n_plots 5-panel belief-map figures to a timestamped subdirectory."""
    matplotlib.use("Agg")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(plot_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    indices = np.linspace(0, len(val_ds) - 1, n_plots, dtype=int)

    for plot_k, idx in enumerate(indices):
        sample = val_ds.samples[int(idx)]

        bh = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
        ov = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
        pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm, unc_norm = model(bh, ov, pos)
        pred_ore_map = normalizer.inverse(pred_norm.squeeze().cpu().numpy())
        pred_unc_map = unc_norm.squeeze().cpu().numpy()

        n_x, n_y = sample["target_map"].shape
        sparse_ore_map = np.zeros((n_x, n_y), dtype=np.float32)
        observation_mask = np.zeros((n_x, n_y), dtype=np.float32)
        for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
            i = int(round(float(px) * (n_x - 1)))
            j = int(round(float(py) * (n_y - 1)))
            sparse_ore_map[i, j] = float(ov_val)
            observation_mask[i, j] = 1.0

        true_ore_map = normalizer.inverse(sample["target_map"])

        plot_belief_sample(
            sparse_ore_map=sparse_ore_map,
            observation_mask=observation_mask,
            true_ore_map=true_ore_map,
            predicted_ore_map=pred_ore_map,
            predicted_uncertainty_map=pred_unc_map,
            save_path=out_dir / f"val_sample_{plot_k:02d}.png",
            title=f"Val sample {plot_k}  ({sample['drill_count']} drills)",
            timestamp=timestamp,
        )

    print(f"  plots saved -> {out_dir}")


def save_e2e_map_sequential_val_plots(
    model: nn.Module,
    val_ds: object,
    normalizer: TargetNormalizer,
    plot_dir: Path,
    device: str,
    n_sequences: int = 3,
) -> None:
    """Save per-step belief-evolution plots for sequential validation sequences.

    For each selected (map, sequence) pair, saves:
    * individual PNGs per step: ``step_001.png``, ``step_003.png``, …
    * a combined ``evolution.png`` grid (rows=steps, cols=5 panels)

    Falls back to ``save_e2e_map_val_plots`` when ``sequence_id`` is not
    present in the dataset samples (i.e. non-sequential datasets).
    """
    if not val_ds.samples or "sequence_id" not in val_ds.samples[0]:
        save_e2e_map_val_plots(model, val_ds, normalizer, plot_dir, device, n_plots=n_sequences)
        return

    matplotlib.use("Agg")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(plot_dir) / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for idx, s in enumerate(val_ds.samples):
        key = (int(s["map_idx"]), int(s["sequence_id"]))
        groups[key].append((int(s["drill_count"]), idx))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    all_keys = list(groups.keys())
    n_select = min(n_sequences, len(all_keys))
    sel_indices = np.linspace(0, len(all_keys) - 1, n_select, dtype=int)
    selected_keys = [all_keys[i] for i in sel_indices]

    model.eval()

    for map_idx, seq_id in selected_keys:
        seq_dir = base_dir / f"seq_{map_idx:04d}_{seq_id:02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        step_pairs = groups[(map_idx, seq_id)]
        n_steps = len(step_pairs)

        panel_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for step, sample_idx in step_pairs:
            sample = val_ds.samples[sample_idx]

            bh = torch.from_numpy(sample["boreholes"]).unsqueeze(0).to(device)
            ov = torch.from_numpy(sample["ore_vals"]).unsqueeze(0).to(device)
            pos = torch.from_numpy(sample["positions"]).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm, unc_norm = model(bh, ov, pos)
            pred_ore = normalizer.inverse(pred_norm.squeeze().cpu().numpy())
            pred_unc = unc_norm.squeeze().cpu().numpy()

            n_x, n_y = sample["target_map"].shape
            sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
            obs_mask = np.zeros((n_x, n_y), dtype=np.float32)
            for (px, py), ov_val in zip(sample["positions"], sample["ore_vals"]):
                i = int(round(float(px) * (n_x - 1)))
                j = int(round(float(py) * (n_y - 1)))
                sparse_ore[i, j] = float(ov_val)
                obs_mask[i, j] = 1.0

            true_ore = normalizer.inverse(sample["target_map"])

            plot_belief_sample(
                sparse_ore_map=sparse_ore,
                observation_mask=obs_mask,
                true_ore_map=true_ore,
                predicted_ore_map=pred_ore,
                predicted_uncertainty_map=pred_unc,
                save_path=seq_dir / f"step_{step:03d}.png",
                title=f"Map {map_idx} / Seq {seq_id} / Step {step} ({step} drills)",
                timestamp=timestamp,
            )

            panel_rows.append((sparse_ore, obs_mask, true_ore, pred_ore, pred_unc))

        global_vmax = max(float(max(t.max(), p.max())) for _, _, t, p, _ in panel_rows)
        global_vmax = max(global_vmax, 1e-3)

        fig, axes = plt.subplots(n_steps, 5, figsize=(22, 4 * n_steps))
        if n_steps == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"Belief evolution — Map {map_idx} / Seq {seq_id}"
            f"  (colour scale max = {global_vmax:.3f})",
            fontsize=12,
        )

        col_titles = ["Observations", "True ore map", "Predicted ore map", "Uncertainty", "Abs error"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

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

            ax = axes[row_idx, 3]
            im_unc = ax.imshow(pred_unc.T, origin="lower", cmap="hot_r")
            ax.scatter(drill_rows, drill_cols, c="blue", s=8, marker="x", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[row_idx, 4]
            abs_err = np.abs(pred_ore - true_ore)
            im_err = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=global_vmax, cmap="Reds")
            ax.set_xlabel(f"max err = {abs_err.max():.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(im_pred, ax=axes[:, 2], shrink=0.6, label="ore value")
        fig.colorbar(im_unc, ax=axes[:, 3], shrink=0.6, label="uncertainty")
        fig.colorbar(im_err, ax=axes[:, 4], shrink=0.6, label="abs error")
        fig.tight_layout()

        evo_path = seq_dir / "evolution.png"
        fig.savefig(evo_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  sequential plots -> {base_dir}")
