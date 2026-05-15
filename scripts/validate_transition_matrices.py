"""Render fitted Markov transition matrices to plots/validation/ (Task 2).

For each formation that has a fitted transition matrix attached to the
FormationGeometry pickle, write a heatmap PNG and a CSV of the matrix
values, plus a summary table of mean run lengths per rock.

Run:
    python scripts/validate_transition_matrices.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.formation_geometry import (
    FormationGeometry,
    MIN_TRANSITIONS_FOR_FIT,
    TRANSITION_BIN_STEP_M,
)


GEOM_PATH = Path("data/clean/formation_geometry.pkl")
OUT_DIR = Path("plots/validation/transition_matrices")
RUNLENGTH_CAP_CELLS = 1000.0  # cap display value; "infinite" runs are real


def _plot_matrix(fm: str, P: pd.DataFrame, n_trans: int,
                 out_path: Path) -> None:
    rocks = list(P.index)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(rocks) + 2),
                                    max(4, 0.7 * len(rocks) + 1.5)))
    im = ax.imshow(P.values, cmap="viridis", vmin=0.0, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(rocks)))
    ax.set_yticks(range(len(rocks)))
    ax.set_xticklabels(rocks, rotation=45, ha="right")
    ax.set_yticklabels(rocks)
    ax.set_xlabel("to rock (t+1)")
    ax.set_ylabel("from rock (t)")
    ax.set_title(f"{fm}: transition matrix "
                 f"({n_trans:,} transitions @ {TRANSITION_BIN_STEP_M:.0f}m)")
    for i in range(len(rocks)):
        for j in range(len(rocks)):
            v = P.values[i, j]
            color = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="P(t+1 | t)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    geom = FormationGeometry.load(GEOM_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fm in geom.formation_order:
        stats = geom.formations.get(fm)
        if stats is None or stats.transition_matrix is None:
            continue
        P = stats.transition_matrix
        n_trans = stats.n_transitions
        fit_mode = ("matrix" if n_trans >= MIN_TRANSITIONS_FOR_FIT
                    else "persistence-fallback")

        # heatmap + csv
        _plot_matrix(fm, P, n_trans, OUT_DIR / f"{fm}_transition.png")
        P.to_csv(OUT_DIR / f"{fm}_transition.csv")

        # summary row
        rl = stats.mean_run_lengths()
        rl_display = sorted(rl.items(), key=lambda kv: -kv[1])
        rl_str = ", ".join(
            f"{r}={min(v, RUNLENGTH_CAP_CELLS):.1f}"
            + ("+" if v > RUNLENGTH_CAP_CELLS else "")
            for r, v in rl_display[:4]
        )
        summary_rows.append({
            "fm": fm,
            "n_transitions": n_trans,
            "fit": fit_mode,
            "n_rocks": len(P.index),
            "max_diag": f"{np.diag(P.values).max():.3f}",
            "top_run_lengths_cells": rl_str,
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nWrote {len(summary_rows)} formation matrices to {OUT_DIR}/")


if __name__ == "__main__":
    main()
