"""Generate combined ore-estimation-error plots across all models and policies.

For each orebody context (single_orebody, multiple_orebodies) this script:
  - Finds the latest timestamped run for each model (cat_var, only_ore)
  - Reads prediction_summaries/<orebody_subdir>/summary_by_step.csv
  - Overlays all model × policy combinations on one chart
  - Saves to results/<orebody_context>/combined_plots/ore_estimation_evolution.png

Usage
-----
python -m decision_simulator.utils.generate_combined_plots
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from decision_simulator.utils.evaluate_policies import (
    plot_combined_ore_evolution,
    plot_combined_per_map_metrics,
    plot_combined_top_ore_rate,
)

# Relative to repo root
_RESULTS_DIR = Path(__file__).parent.parent / "results"

_OREBODY_CONTEXTS = {
    "single_orebody":    "one_orebody",
    "multiple_orebodies": "two_orebodies",
}
_MODELS = ["cat_var", "only_ore"]


def _latest_run(model_dir: Path) -> Path | None:
    """Return the most-recently-timestamped run directory under *model_dir*."""
    candidates = [d for d in sorted(model_dir.iterdir()) if d.is_dir()]
    return candidates[-1] if candidates else None


def _load_summary(run_dir: Path, orebody_subdir: str) -> pd.DataFrame | None:
    csv_path = run_dir / "prediction_summaries" / orebody_subdir / "summary_by_step.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["policy"] = df["policy"].str.strip()
    return df


def _load_per_map_metrics(run_dir: Path, orebody_subdir: str) -> pd.DataFrame | None:
    csv_path = run_dir / "prediction_summaries" / orebody_subdir / "per_map_metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in [c for c in df.columns if df[c].dtype == object]:
        df[col] = df[col].str.strip()
        if col != "policy":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _generate_for_context(context: str, orebody_subdir: str) -> None:
    context_dir = _RESULTS_DIR / context
    if not context_dir.is_dir():
        print(f"  [skip] {context_dir} does not exist")
        return

    frames: list[pd.DataFrame] = []
    per_map_frames: list[pd.DataFrame] = []
    for model in _MODELS:
        model_dir = context_dir / model
        if not model_dir.is_dir():
            print(f"  [skip] {model_dir} does not exist")
            continue

        run_dir = _latest_run(model_dir)
        if run_dir is None:
            print(f"  [skip] no run dirs found in {model_dir}")
            continue

        df = _load_summary(run_dir, orebody_subdir)
        if df is None:
            print(f"  [skip] no summary_by_step.csv in {run_dir}/prediction_summaries/{orebody_subdir}/")
            continue

        df["model"] = model
        frames.append(df)

        per_map_df = _load_per_map_metrics(run_dir, orebody_subdir)
        if per_map_df is not None:
            per_map_df["model"] = model
            per_map_frames.append(per_map_df)

        print(f"  Loaded {model}: {run_dir.name} ({len(df)} rows)")

    if not frames:
        print(f"  [skip] no data found for {context}, skipping plot.")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_dir = context_dir / "combined_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "ore_estimation_evolution.png"
    plot_combined_ore_evolution(combined, save_path)
    print(f"  -> {save_path}")
    ore_ci_path = out_dir / "ore_estimation_evolution_ci.png"
    plot_combined_ore_evolution(combined, ore_ci_path, band="ci")
    print(f"  -> {ore_ci_path}")

    top_ore_path = out_dir / "top_ore_rate_evolution.png"
    plot_combined_top_ore_rate(combined, top_ore_path)
    print(f"  -> {top_ore_path}")
    top_ore_ci_path = out_dir / "top_ore_rate_evolution_ci.png"
    plot_combined_top_ore_rate(combined, top_ore_ci_path, band="ci")
    print(f"  -> {top_ore_ci_path}")

    if per_map_frames:
        combined_per_map = pd.concat(per_map_frames, ignore_index=True)
        plot_combined_per_map_metrics(combined_per_map, out_dir)
        print(f"  -> {out_dir / 'first_top_ore_step_distribution.png'}")
        print(f"  -> {out_dir / 'n_top_ore_found_distribution.png'}")
        print(f"  -> {out_dir / 'n_top_ore_found_distribution_violin.png'}")
        print(f"  -> {out_dir / 'n_top_ore_found_split_violin.png'}")
        print(f"  -> {out_dir / 'first_top_ore_step_pyramid.png'}")


def main() -> None:
    for context, orebody_subdir in _OREBODY_CONTEXTS.items():
        print(f"\n{context} ({orebody_subdir})")
        _generate_for_context(context, orebody_subdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
