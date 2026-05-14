from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import pandas as pd

from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.utils.experiment_results import (
    build_step_rows,
    build_summary_row,
    print_aggregate,
)
from decision_simulator.utils.plotting import plot_trajectory


def run_many_experiments(
    seeds: list[int],
    cfg,
    out_dir: Path,
    resources: DecisionSimulationResources,
    device: str | None,
    method: str,
    simulation_fn: Callable,
) -> pd.DataFrame:
    """Generic multi-seed experiment runner.

    Parameters
    ----------
    seeds         : list of integer seeds to run
    cfg           : policy config dataclass (GreedyConfig or ParticleBeliefConfig)
    out_dir       : directory for parquet files and plots
    resources     : shared DecisionSimulationResources
    device        : torch device string
    method        : method name constant (METHOD_GREEDY or METHOD_PARTICLE_BELIEF)
    simulation_fn : callable(seed, cfg, resources, device, verbose=False)
                    -> (observations, true_map, decision)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    all_step_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    total_runs = len(seeds)

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[{run_idx}/{total_runs}] seed={seed}", flush=True)

        observations, true_map, decision = simulation_fn(
            seed=seed,
            cfg=cfg,
            resources=resources,
            device=device,
            verbose=False,
        )

        best_ore = max(o["ore_value"] for o in observations)
        print(f"  best_ore={best_ore:.4f}  decision={decision}")

        for obs in observations:
            obs["decision"] = decision

        all_step_rows.extend(build_step_rows(seed, method, cfg, observations, decision))
        all_summary_rows.append(
            build_summary_row(seed, method, cfg, observations, true_map, decision)
        )
        plot_trajectory(
            seed=seed,
            method=method,
            observations=observations,
            true_map=true_map,
            out_dir=out_dir,
        )

    steps_df = pd.DataFrame(all_step_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    steps_path = out_dir / "steps.parquet"
    summary_path = out_dir / "summary.parquet"
    steps_df.to_parquet(steps_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    print(f"\nSaved {len(steps_df)} step rows    -> {steps_path}")
    print(f"Saved {len(summary_df)} summary rows -> {summary_path}")

    print_aggregate(summary_df)
    return summary_df
