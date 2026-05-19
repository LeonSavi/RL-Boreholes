from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_belief_sample(
    sparse_ore_map: np.ndarray,
    observation_mask: np.ndarray,
    true_ore_map: np.ndarray,
    predicted_ore_map: np.ndarray,
    save_path: Path | None = None,
    title: str = "",
    timestamp: str = "",
) -> None:
    """Four-panel validation plot: observations / truth / prediction / error.

    Parameters
    ----------
    sparse_ore_map    : (n_x, n_y) raw ore values at drilled cells, 0 elsewhere
    observation_mask  : (n_x, n_y) binary, 1 = drilled
    true_ore_map      : (n_x, n_y) ground-truth max-pooled ore field
    predicted_ore_map : (n_x, n_y) model output in ore-value space
    save_path         : if given, save to this path (PNG); directory is created
    title             : optional figure super-title
    timestamp         : if given, printed at the bottom of the figure
    """
    import matplotlib.pyplot as plt

    vmax = float(max(true_ore_map.max(), predicted_ore_map.max(), 1e-3))
    drill_rows, drill_cols = np.where(observation_mask > 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if title:
        fig.suptitle(title, fontsize=11)

    # imshow with origin="lower" so axis-0 → x (horizontal), axis-1 → y (vertical)
    kw = dict(origin="lower", cmap="viridis")

    ax = axes[0]
    ax.set_title(f"Observations ({int(observation_mask.sum())} drills)")
    im = ax.imshow(sparse_ore_map.T, vmin=0, vmax=vmax, **kw)
    ax.scatter(drill_rows, drill_cols, c="red", s=10, marker="x", linewidths=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.set_title("True ore map")
    im = ax.imshow(true_ore_map.T, vmin=0, vmax=vmax, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ax.set_title("Predicted ore map")
    im = ax.imshow(predicted_ore_map.T, vmin=0, vmax=vmax, **kw)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[3]
    ax.set_title("Absolute error")
    abs_err = np.abs(predicted_ore_map - true_ore_map)
    im = ax.imshow(abs_err.T, origin="lower", vmin=0, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()

    if timestamp:
        fig.text(0.5, 0.01, timestamp, ha="center", va="bottom",
                 fontsize=8, color="gray")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)
