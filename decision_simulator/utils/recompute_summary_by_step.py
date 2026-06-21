"""Standalone script to recompute summary_by_step.csv for existing run directories.

Reads all per-map CSVs from prediction_summaries/*/maps/ inside each given run
directory, then writes summary_by_step.csv and total_predicted_ore_evolution.png
to each orebody subdirectory.

Usage
-----
python decision_simulator/utils/recompute_summary_by_step.py <run_dir> [<run_dir> ...]

Example
-------
python decision_simulator/utils/recompute_summary_by_step.py \
    "decision_simulator/results/single_orebody/cat_var/20260608_222813" \
    "decision_simulator/results/single_orebody/only_ore/20260608_231952"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from decision_simulator.utils.evaluate_policies import write_summary_by_step


def _load_maps_dir(maps_dir: Path) -> pd.DataFrame:
    """Read all map_XX.csv files from *maps_dir* into a single DataFrame."""
    csv_files = sorted(maps_dir.glob("map_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No map_*.csv files found in {maps_dir}")

    frames = []
    for path in csv_files:
        match = re.search(r"map_(\d+)\.csv$", path.name)
        map_idx = int(match.group(1)) if match else -1
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if "map_idx" not in df.columns:
            df["map_idx"] = map_idx
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["policy"] = combined["policy"].str.strip()
    combined["top_ore"] = combined["top_ore"].astype(bool)
    return combined


def _process_run_dir(run_dir: Path) -> None:
    pred_summaries = run_dir / "prediction_summaries"
    if not pred_summaries.is_dir():
        print(f"  [skip] no prediction_summaries/ in {run_dir}")
        return

    orebody_dirs = [d for d in sorted(pred_summaries.iterdir())
                    if d.is_dir() and (d / "maps").is_dir()]

    if not orebody_dirs:
        print(f"  [skip] no orebody subdirectories with maps/ found in {pred_summaries}")
        return

    for orebody_dir in orebody_dirs:
        maps_dir = orebody_dir / "maps"
        print(f"  Processing {orebody_dir.name} ({maps_dir}) ...")
        try:
            df = _load_maps_dir(maps_dir)
            write_summary_by_step(df, orebody_dir)
            print(f"    -> {orebody_dir / 'summary_by_step.csv'}")
            print(f"    -> {orebody_dir / 'total_predicted_ore_evolution.png'}")
        except Exception as exc:
            print(f"    [error] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute summary_by_step.csv for existing run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories containing prediction_summaries/*/maps/.",
    )
    args = parser.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        print(f"\nRun directory: {run_dir}")
        if not run_dir.is_dir():
            print(f"  [error] directory not found: {run_dir}")
            continue
        _process_run_dir(run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
