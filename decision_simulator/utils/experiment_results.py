import numpy as np
import pandas as pd


def build_step_rows(
    seed: int,
    method: str,
    cfg,
    observations: list[dict],
    decision: str,
) -> list[dict]:
    """One parquet row per drilled location, in step order."""
    rows = []
    best_so_far = -np.inf
    for obs in sorted(observations, key=lambda o: o["step"]):
        best_so_far = max(best_so_far, obs["ore_value"])
        step = obs["step"]
        rows.append(
            {
                "seed": seed,
                "method": method,
                "step": step,
                "x": obs["location"][0],
                "y": obs["location"][1],
                "selection_type": (
                    "random" if step <= cfg.initial_random_drills else "greedy"
                ),
                "predicted_ore": obs["predicted_ore"],  # None -> NaN in pandas
                "true_ore": obs["ore_value"],
                "best_so_far": best_so_far,
                "decision": decision,
                "mine_threshold": cfg.mine_threshold,
                "drilling_budget": cfg.drilling_budget,
            }
        )
    return rows


def build_summary_row(
    seed: int,
    method: str,
    cfg,
    observations: list[dict],
    true_map: dict,
    decision: str,
) -> dict:
    """One parquet row per seed."""
    best_observed_ore = max(o["ore_value"] for o in observations)
    true_best_ore = float(true_map["yield_field"].max())
    return {
        "seed": seed,
        "method": method,
        "best_observed_ore": best_observed_ore,
        "final_decision": decision,
        "mine_threshold": cfg.mine_threshold,
        "drilling_budget": cfg.drilling_budget,
        "n_drills": len(observations),
        "true_best_ore": true_best_ore,
        "regret": true_best_ore - best_observed_ore,
        "success": best_observed_ore >= cfg.mine_threshold,
    }


def print_aggregate(summary_df: pd.DataFrame) -> None:
    n = len(summary_df)
    mean_best = summary_df["best_observed_ore"].mean()
    std_best = summary_df["best_observed_ore"].std()
    mean_regret = summary_df["regret"].mean()
    std_regret = summary_df["regret"].std()
    success_rate = summary_df["success"].mean() * 100
    mine_rate = (summary_df["final_decision"] == "MINE").mean() * 100

    print("\n" + "=" * 52)
    print("EXPERIMENT SUMMARY")
    print("=" * 52)
    print(f"  Number of runs          : {n}")
    print(f"  Mean best observed ore  : {mean_best:.4f} +/- {std_best:.4f}")
    print(f"  Std best observed ore   : {std_best:.4f}")
    print(f"  Mean regret             : {mean_regret:.4f} +/- {std_regret:.4f}")
    print(f"  Success rate            : {success_rate:.1f}%")
    print(f"  Mine decision rate      : {mine_rate:.1f}%")
    print("=" * 52)
