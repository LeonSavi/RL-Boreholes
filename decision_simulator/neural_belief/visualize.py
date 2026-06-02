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
    # vmax matches prediction scale: "light red" and "dark purple" represent the
    # same numerical value, so the error panel is directly comparable to the others
    im = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=vmax, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()

    if timestamp:
        fig.text(0.5, 0.01, timestamp, ha="center", va="bottom",
                 fontsize=8, color="gray")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)


def plot_belief_step_with_uncertainty(
    sparse_ore_map: np.ndarray,
    observation_mask: np.ndarray,
    true_ore_map: np.ndarray,
    predicted_ore_map: np.ndarray,
    predicted_uncertainty_map: np.ndarray | None = None,
    save_path: Path | None = None,
    title: str = "",
    timestamp: str = "",
) -> None:
    """Per-step POMDP plot: observations / truth / prediction / uncertainty / error.

    Extends ``plot_belief_sample`` with an optional uncertainty panel. When
    ``predicted_uncertainty_map`` is None the layout falls back to 4 panels.

    Parameters
    ----------
    sparse_ore_map            : (n_x, n_y) ore values at drilled cells, 0 elsewhere
    observation_mask          : (n_x, n_y) bool, True = drilled
    true_ore_map              : (n_x, n_y) ground-truth ore field
    predicted_ore_map         : (n_x, n_y) model output in ore-value space
    predicted_uncertainty_map : (n_x, n_y) per-cell uncertainty in normalised model
                                space, or None
    save_path                 : if given, save PNG here; parent dir is created
    title                     : optional figure super-title
    timestamp                 : if given, printed at the bottom of the figure
    """
    import matplotlib.pyplot as plt

    has_unc = predicted_uncertainty_map is not None
    n_panels = 5 if has_unc else 4
    vmax = float(max(true_ore_map.max(), predicted_ore_map.max(), 1e-3))
    drill_rows, drill_cols = np.where(observation_mask)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if title:
        fig.suptitle(title, fontsize=11)

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
    ax.scatter(drill_rows, drill_cols, c="red", s=10, marker="x", linewidths=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046)

    if has_unc:
        ax = axes[3]
        ax.set_title("Predicted uncertainty")
        im = ax.imshow(predicted_uncertainty_map.T, origin="lower", cmap="hot_r")
        ax.scatter(drill_rows, drill_cols, c="blue", s=10, marker="x", linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[-1]
    ax.set_title("Absolute error")
    abs_err = np.abs(predicted_ore_map - true_ore_map)
    im = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=vmax, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()

    if timestamp:
        fig.text(0.5, 0.01, timestamp, ha="center", va="bottom",
                 fontsize=8, color="gray")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)


def plot_policy_map_evolution(
    policy_name: str,
    step_plot_dir: Path,
    map_idx: int,
    selected_steps: list[int],
    timestamp: str,
    save_path: Path | None = None,
) -> None:
    """Evolution grid for one policy on one map: rows = selected steps.

    Loads previously saved per-step PNG files and stacks them vertically.
    Each row is the full multi-panel belief plot for that step.

    Parameters
    ----------
    policy_name   : policy identifier matching the subdirectory name
    step_plot_dir : base plot directory; step PNGs are expected at
                    ``step_plot_dir / policy_name / timestamp / map_{idx:02d}_step_{step:02d}.png``
    map_idx       : map index used in the file paths and figure title
    selected_steps: list of step indices (1-based) to include as rows
    timestamp     : timestamp string matching the saved per-step plots
    save_path     : if given, save PNG here; parent dir is created
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n_rows = len(selected_steps)
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows), squeeze=False)
    fig.suptitle(f"{policy_name} | Map {map_idx:02d}", fontsize=13, y=1.01)

    for row, step in enumerate(selected_steps):
        ax = axes[row][0]
        img_path = (
            step_plot_dir / policy_name / timestamp
            / f"map_{map_idx:02d}_step_{step:02d}.png"
        )
        if img_path.exists():
            ax.imshow(mpimg.imread(str(img_path)))
        else:
            ax.text(0.5, 0.5, f"Step {step} — missing", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=12)
        ax.axis("off")
        ax.set_ylabel(f"Step {step}", fontsize=10, rotation=90, labelpad=4)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)


def plot_policy_evolution_grid(
    policy_names: list[str],
    step_plot_dir: Path,
    map_idx: int,
    n_steps: int,
    timestamp: str,
    save_path: Path | None = None,
) -> None:
    """Grid figure: rows = policies, columns = drill steps, cells = per-step plots.

    Loads the PNG files previously saved by the per-step callback and arranges
    them in a (n_policies × n_steps) grid so the full belief evolution for every
    policy is visible in one figure.

    Parameters
    ----------
    policy_names  : ordered list of policy names (one row each)
    step_plot_dir : base plot directory; step PNGs are expected at
                    ``step_plot_dir / policy / timestamp / map_{idx:02d}_step_{step:02d}.png``
    map_idx       : map index used in the file paths and figure title
    n_steps       : number of policy-driven steps (columns)
    timestamp     : timestamp string matching the saved per-step plots
    save_path     : if given, save PNG here; parent dir is created
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n_policies = len(policy_names)
    fig, axes = plt.subplots(
        n_policies, n_steps,
        figsize=(4 * n_steps, 4 * n_policies),
        squeeze=False,
    )
    fig.suptitle(f"Policy drill evolution — Map {map_idx:02d}", fontsize=13, y=1.01)

    for row, policy in enumerate(policy_names):
        for col, step in enumerate(range(1, n_steps + 1)):
            ax = axes[row][col]
            img_path = (
                step_plot_dir / policy / timestamp
                / f"map_{map_idx:02d}_step_{step:02d}.png"
            )
            if img_path.exists():
                ax.imshow(mpimg.imread(str(img_path)))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="red")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Step {step}", fontsize=9)
        axes[row][0].set_ylabel(policy, fontsize=9, rotation=90, labelpad=4)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)


def plot_policy_evolution(
    results: dict[str, dict],
    map_idx: int,
    save_path: Path | None = None,
) -> None:
    """Line plot of best-ore-found-so-far over drill steps for all policies on one map.

    Parameters
    ----------
    results  : {policy_name: episode_result_dict} as returned by
               ``run_fixed_budget_episode``
    map_idx  : map index used in the title
    save_path: if given, save PNG here; parent dir is created
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    total_true = None
    for name, result in results.items():
        history = result["step_history"]
        if not history:
            continue
        steps = [row["step"] for row in history]
        total_pred = [row["total_predicted_ore"] for row in history]
        ax.plot(steps, total_pred, marker="o", label=name)
        if total_true is None:
            total_true = result.get("total_true_ore")

    if total_true is not None:
        ax.axhline(total_true, color="black", linestyle="--", linewidth=1.2, label="true total ore")

    ax.set_xlabel("Policy step")
    ax.set_ylabel("Total predicted ore (map sum)")
    ax.set_title(f"Policy comparison — Map {map_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    plt.close(fig)
