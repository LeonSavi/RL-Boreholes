from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


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
