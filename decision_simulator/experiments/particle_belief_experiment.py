from __future__ import annotations

import torch
from pathlib import Path

import pandas as pd

from decision_simulator.config_decision_experiments import (
    ParticleBeliefConfig,
    METHOD_PARTICLE_BELIEF,
)
from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.pomdp.policies.particle_belief_policy import (
    run_particle_belief_simulation,
)
from decision_simulator.utils.experiment_results import (
    build_step_rows,
    build_summary_row,
    print_aggregate,
)
from decision_simulator.utils.plotting import plot_trajectory


def run_many_particle_belief_experiments(
    seeds: list[int],
    cfg: ParticleBeliefConfig,
    out_dir: Path,
    resources: DecisionSimulationResources,
    device: str | None = None,
) -> pd.DataFrame:
    """Run particle-belief simulation over many seeds and persist results.

    Writes
    ------
    out_dir/steps.parquet
    out_dir/summary.parquet
    out_dir/plots/trajectory_seed_{seed}.png

    Returns
    -------
    summary_df : pd.DataFrame  - one row per seed
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    all_step_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    total_runs = len(seeds)

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[{run_idx}/{total_runs}] seed={seed}", flush=True)

        observations, true_map, decision = run_particle_belief_simulation(
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

        all_step_rows.extend(
            build_step_rows(seed, METHOD_PARTICLE_BELIEF, cfg, observations, decision)
        )
        all_summary_rows.append(
            build_summary_row(
                seed, METHOD_PARTICLE_BELIEF, cfg, observations, true_map, decision
            )
        )
        plot_trajectory(
            seed=seed,
            method=METHOD_PARTICLE_BELIEF,
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
