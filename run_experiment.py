"""
run_experiment.py - CLI entry point for drilling decision experiments.

Usage
-----
    # greedy, single run (verbose, no files written)
    python run_experiment.py --policy greedy --mode single --seed 42

    # greedy, multi-seed experiment
    python run_experiment.py --policy greedy --mode many --n-seeds 100

    # particle belief, single run
    python run_experiment.py --policy particle_belief --mode single --seed 42

    # particle belief, multi-seed experiment
    python run_experiment.py --policy particle_belief --mode many --n-seeds 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from decision_simulator.config_decision_experiments import (
    GreedyConfig,
    ParticleBeliefConfig,
    METHOD_GREEDY,
    METHOD_PARTICLE_BELIEF,
    RESULTS_BASE,
    JEPA_CHECKPOINT,
    DISTRIBUTIONS,
    FORMATION_GEOMETRY,
    DISCOVERY_PRIOR,
)
from decision_simulator.resources import load_resources
from decision_simulator.pomdp.policies.greedy_policy import run_greedy_simulation
from decision_simulator.pomdp.policies.particle_belief_policy import (
    run_particle_belief_simulation,
)
from decision_simulator.experiments.greedy_experiment import run_many_greedy_experiments
from decision_simulator.experiments.particle_belief_experiment import (
    run_many_particle_belief_experiments,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Drilling decision experiment runner")
    p.add_argument(
        "--policy",
        choices=["greedy", "particle_belief"],
        default="greedy",
        help="decision policy to run",
    )
    p.add_argument(
        "--mode",
        choices=["single", "many"],
        default="single",
        help="single: one seed (verbose); many: multi-seed experiment",
    )
    # single-run option
    p.add_argument("--seed", type=int, default=42, help="seed for --mode single")
    # multi-run option
    p.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="number of seeds for --mode many (uses seeds 0..n-1)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for parquet files and plots (default: decision_simulator/results/<method>)",
    )
    # shared config
    p.add_argument("--drilling-budget", type=int, default=10)
    p.add_argument("--initial-random-drills", type=int, default=3)
    p.add_argument("--mine-threshold", type=float, default=0.7)
    p.add_argument("--device", default=None)
    # greedy-specific
    p.add_argument("--k-neighbors", type=int, default=5)
    # particle-belief-specific
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.1)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    resources = load_resources(
        jepa_path=JEPA_CHECKPOINT,
        distributions_path=DISTRIBUTIONS,
        formation_geometry_path=FORMATION_GEOMETRY,
        discovery_prior_path=DISCOVERY_PRIOR,
        device=device,
    )

    dispatch = {
        "greedy": {
            "single": _run_greedy_single,
            "many": _run_greedy_many,
        },
        "particle_belief": {
            "single": _run_particle_belief_single,
            "many": _run_particle_belief_many,
        },
    }

    dispatch[args.policy][args.mode](args, resources, device)


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def _run_greedy_single(args, resources, device: str) -> None:
    cfg = GreedyConfig(
        drilling_budget=args.drilling_budget,
        initial_random_drills=args.initial_random_drills,
        mine_threshold=args.mine_threshold,
        k_neighbors=args.k_neighbors,
    )
    run_greedy_simulation(
        seed=args.seed,
        cfg=cfg,
        resources=resources,
        device=device,
        verbose=True,
    )


def _run_greedy_many(args, resources, device: str) -> None:
    cfg = GreedyConfig(
        drilling_budget=args.drilling_budget,
        initial_random_drills=args.initial_random_drills,
        mine_threshold=args.mine_threshold,
        k_neighbors=args.k_neighbors,
    )
    out_dir = args.out_dir or (RESULTS_BASE / METHOD_GREEDY)
    run_many_greedy_experiments(
        seeds=list(range(args.n_seeds)),
        cfg=cfg,
        out_dir=out_dir,
        resources=resources,
        device=device,
    )


# ---------------------------------------------------------------------------
# Particle belief
# ---------------------------------------------------------------------------

def _run_particle_belief_single(args, resources, device: str) -> None:
    cfg = ParticleBeliefConfig(
        drilling_budget=args.drilling_budget,
        initial_random_drills=args.initial_random_drills,
        mine_threshold=args.mine_threshold,
        n_particles=args.n_particles,
        temperature=args.temperature,
    )
    run_particle_belief_simulation(
        seed=args.seed,
        cfg=cfg,
        resources=resources,
        device=device,
        verbose=True,
    )


def _run_particle_belief_many(args, resources, device: str) -> None:
    cfg = ParticleBeliefConfig(
        drilling_budget=args.drilling_budget,
        initial_random_drills=args.initial_random_drills,
        mine_threshold=args.mine_threshold,
        n_particles=args.n_particles,
        temperature=args.temperature,
    )
    out_dir = args.out_dir or (RESULTS_BASE / METHOD_PARTICLE_BELIEF)
    run_many_particle_belief_experiments(
        seeds=list(range(args.n_seeds)),
        cfg=cfg,
        out_dir=out_dir,
        resources=resources,
        device=device,
    )


if __name__ == "__main__":
    main()
