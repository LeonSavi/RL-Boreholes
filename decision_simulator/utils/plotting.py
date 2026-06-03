from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from decision_simulator.pomdp.beliefs.belief_state import BeliefState


def plot_trajectory(
    seed: int,
    method: str,
    observations: list[dict],
    true_map: dict,
    out_dir: Path,
) -> None:
    """Save a trajectory plot for one seed to out_dir/plots/."""
    ore_map = true_map["yield_field"].max(axis=2)  # (n_x, n_y) peak yield

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        ore_map,
        origin="lower",
        cmap="YlOrRd",
        aspect="equal",
        extent=[-0.5, ore_map.shape[1] - 0.5, -0.5, ore_map.shape[0] - 0.5],
    )
    plt.colorbar(im, ax=ax, label="Peak ore yield", fraction=0.046, pad=0.04)

    steps = [o["step"] for o in observations]
    xs = [o["location"][0] for o in observations]
    ys = [o["location"][1] for o in observations]

    sc = ax.scatter(
        ys,
        xs,
        c=steps,
        cmap="cool",
        s=120,
        zorder=3,
        edgecolors="white",
        linewidths=0.7,
        vmin=1,
        vmax=len(steps),
    )
    plt.colorbar(sc, ax=ax, label="Drill step", fraction=0.046, pad=0.04)

    for obs in observations:
        i, j = obs["location"]
        ax.text(
            j,
            i,
            str(obs["step"]),
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    best_obs = max(observations, key=lambda o: o["ore_value"])
    bi, bj = best_obs["location"]
    ax.scatter(
        [bj],
        [bi],
        marker="*",
        s=350,
        c="gold",
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )

    best_ore = max(o["ore_value"] for o in observations)
    decision = observations[0].get("decision", "?")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.set_title(f"{method}  seed={seed}  best_ore={best_ore:.3f}  decision={decision}")
    fig.tight_layout()

    out_path = out_dir / "plots" / f"trajectory_seed_{seed}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_belief_sample(
    sparse_ore_map: np.ndarray,
    observation_mask: np.ndarray,
    true_ore_map: np.ndarray,
    predicted_ore_map: np.ndarray,
    predicted_uncertainty_map: np.ndarray | None = None,
    save_path: Path | None = None,
    title: str = "",
    timestamp: str = "",
) -> None:
    """Five-panel belief plot: observations / truth / prediction / uncertainty / error.

    Parameters
    ----------
    sparse_ore_map            : (n_x, n_y) ore values at drilled cells, 0 elsewhere
    observation_mask          : (n_x, n_y) bool/float, True/1 = drilled
    true_ore_map              : (n_x, n_y) ground-truth ore field
    predicted_ore_map         : (n_x, n_y) model output in ore-value space
    predicted_uncertainty_map : (n_x, n_y) per-cell uncertainty; shows placeholder if None
    save_path                 : if given, save PNG here; parent dir is created
    title                     : optional figure super-title
    timestamp                 : if given, printed at the bottom of the figure
    """
    vmax = float(max(true_ore_map.max(), predicted_ore_map.max(), 1e-3))
    drill_rows, drill_cols = np.where(observation_mask > 0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
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

    ax = axes[3]
    ax.set_title("Predicted uncertainty")
    if predicted_uncertainty_map is not None:
        im = ax.imshow(predicted_uncertainty_map.T, origin="lower", cmap="hot_r")
        ax.scatter(drill_rows, drill_cols, c="blue", s=10, marker="x", linewidths=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "No uncertainty", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.axis("off")

    ax = axes[4]
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


def plot_step_belief_grid(
    step_beliefs: dict[str, dict[int, BeliefState]],
    map_idx: int,
    drill_counts: tuple[int, int] = (2, 3),
    timestamp: str = "",
    save_path: Path | None = None,
) -> None:
    """4-row × 4-col belief grid comparing policies at two drill counts.

    Layout
    ------
    Rows  : one per policy (in insertion order of ``step_beliefs``)
    Cols  : ore@drill_counts[0] | uncertainty@drill_counts[0]
             | ore@drill_counts[1] | uncertainty@drill_counts[1]

    Parameters
    ----------
    step_beliefs : {policy_name: {n_drills: BeliefState}}
                   BeliefState at step k was computed from k observations.
    map_idx      : map index used in the figure title
    drill_counts : the two drill-count snapshots to compare (default 2 and 3)
    timestamp    : optional timestamp string printed at the bottom
    save_path    : if given, save PNG here; parent dir is created
    """
    policy_names = list(step_beliefs.keys())
    n_rows = len(policy_names)
    d0, d1 = drill_counts

    col_titles = [
        f"Predicted ore map\n({d0} drills)",
        f"Predicted uncertainty\n({d0} drills)",
        f"Predicted ore map\n({d1} drills)",
        f"Predicted uncertainty\n({d1} drills)",
    ]

    _POLICY_LABELS: dict[str, tuple[str, str]] = {
        "random":       ("Random",      "randomly selects a cell"),
        "greedy_yield": ("Greedy",      "Highest predicted ore"),
        "uncertainty":  ("Uncertainty", "Highest uncertainty"),
        "hybrid":       ("Hybrid",      "Combination of Ore + Uncertainty"),
    }

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4 * n_rows))
    fig.suptitle(f"Belief comparison — Map {map_idx:02d}", fontsize=13, y=1.01)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9)

    for row, name in enumerate(policy_names):
        beliefs = step_beliefs[name]

        display_name, description = _POLICY_LABELS.get(name, (name, ""))
        ax0 = axes[row, 0]
        ax0.set_ylabel("")
        ax0.text(-0.18, 0.58, display_name,
                 transform=ax0.transAxes, ha="right", va="bottom",
                 fontsize=12, fontweight="bold", clip_on=False)
        ax0.text(-0.18, 0.52, description,
                 transform=ax0.transAxes, ha="right", va="top",
                 fontsize=9, clip_on=False)

        for col_pair, n_drills in enumerate([d0, d1]):
            belief = beliefs.get(n_drills)
            col_ore = col_pair * 2
            col_unc = col_pair * 2 + 1

            ax_ore = axes[row, col_ore]
            ax_unc = axes[row, col_unc]

            if belief is None:
                for ax in (ax_ore, ax_unc):
                    ax.text(0.5, 0.5, f"n/a ({n_drills} drills)",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=9, color="gray")
                    ax.axis("off")
                continue

            pred_ore = belief.predicted_ore_map
            pred_unc = belief.predicted_uncertainty_map
            drill_rows, drill_cols = np.where(belief.observed_mask)
            vmax_ore = float(max(pred_ore.max(), 1e-3))

            im_ore = ax_ore.imshow(pred_ore.T, origin="lower", cmap="viridis",
                                   vmin=0, vmax=vmax_ore)
            ax_ore.scatter(drill_rows, drill_cols, c="white", s=14, marker="o",
                           linewidths=0.6, edgecolors="black", zorder=3)
            fig.colorbar(im_ore, ax=ax_ore, fraction=0.046)

            if pred_unc is not None:
                im_unc = ax_unc.imshow(pred_unc.T, origin="lower", cmap="hot_r")
                ax_unc.scatter(drill_rows, drill_cols, c="cyan", s=14, marker="o",
                               linewidths=0.6, edgecolors="black", zorder=3)
                fig.colorbar(im_unc, ax=ax_unc, fraction=0.046)
            else:
                ax_unc.text(0.5, 0.5, "No uncertainty", ha="center", va="center",
                            transform=ax_unc.transAxes, fontsize=9, color="gray")
                ax_unc.axis("off")

    fig.tight_layout()

    if timestamp:
        fig.text(0.5, 0.0, timestamp, ha="center", va="bottom",
                 fontsize=8, color="gray")

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
