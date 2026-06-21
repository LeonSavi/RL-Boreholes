"""Policy evaluation metrics computed from in-memory step data.

Called from pomdp.py after all maps have been evaluated. Writes five CSV files
to the prediction_summaries directory:
  - per_map_metrics.csv      : 10 metrics per (map_idx, policy)
  - aggregate_metrics.csv    : mean / std / n_maps per policy
  - cumulative_ore_by_step.csv
  - summary_by_step.csv      : top-ore hit rate + total predicted ore, per (policy, step)
And one PNG:
  - total_predicted_ore_evolution.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_ROUND = 3

_OREBODY_NAMES = {0: "no_orebodies", 1: "one_orebody", 2: "two_orebodies"}


def _orebody_dir_name(n: int) -> str:
    return _OREBODY_NAMES.get(n, f"{n}_orebodies")


# ── Output helpers ───────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame, *secondary_keys: str) -> pd.DataFrame:
    """Sort alphabetically by policy (then secondary keys) and round floats."""
    df = df.sort_values(["policy", *secondary_keys]).reset_index(drop=True)
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(_ROUND)
    return df


# ── Correlation helpers ───────────────────────────────────────────────────────

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


# ── Per-map metrics ───────────────────────────────────────────────────────────

def _compute_one_map(df: pd.DataFrame) -> dict:
    """Compute all metrics for a single (map_idx, policy) group."""
    n_steps = len(df)
    true_ore = df["true_ore"].to_numpy(dtype=float)
    pred_ore = df["predicted_ore"].to_numpy(dtype=float)
    pred_unc = df["predicted_uncertainty"].to_numpy(dtype=float)
    top_mask = df["top_ore"].to_numpy(dtype=bool)
    abs_err = np.abs(pred_ore - true_ore)

    top_ore_steps = df.loc[top_mask, "step"].to_numpy()
    first_top = float(top_ore_steps.min()) if len(top_ore_steps) > 0 else float("nan")

    return {
        "top_ore_hit_rate":    float(top_mask.sum()) / n_steps,
        "first_top_ore_step":  first_top,
        "n_top_ore_found":     int(top_mask.sum()),
        "avg_true_ore_drilled":float(true_ore.mean()),
        "cumulative_true_ore": float(true_ore.sum()),
        "mae":                 float(abs_err.mean()),
        "pearson_r":           _safe_pearson(pred_ore, true_ore),
        "spearman_r":          _safe_spearman(pred_ore, true_ore),
        "unc_abs_error_corr":  _safe_pearson(pred_unc, abs_err),
        "unc_true_ore_corr":   _safe_pearson(pred_unc, true_ore),
    }


def _build_per_map_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (map_idx, policy), grp in df.groupby(["map_idx", "policy"], sort=True):
        metrics = _compute_one_map(grp.sort_values("step"))
        rows.append({"map_idx": map_idx, "policy": policy, **metrics})
    return pd.DataFrame(rows)


# ── Aggregate metrics ─────────────────────────────────────────────────────────

_METRIC_COLS = [
    "top_ore_hit_rate", "first_top_ore_step", "n_top_ore_found",
    "mae", "pearson_r", "spearman_r",
    "unc_abs_error_corr", "unc_true_ore_corr",
]


def _aggregate_metrics(per_map_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, grp in per_map_df.groupby("policy", sort=True):
        row: dict = {"policy": policy, "n_maps": len(grp)}
        for col in _METRIC_COLS:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Step-level curve tables ───────────────────────────────────────────────────

def _cumulative_ore_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, step): mean and std of cumulative true ore across maps."""
    rows = []
    for policy, pol_grp in df.groupby("policy", sort=True):
        maps = sorted(pol_grp["map_idx"].unique())
        steps = sorted(pol_grp["step"].unique())
        # Build (n_maps × n_steps) matrix of cumulative ore
        matrix = []
        for map_idx in maps:
            m = pol_grp[pol_grp["map_idx"] == map_idx].sort_values("step")
            matrix.append(np.cumsum(m["true_ore"].to_numpy(dtype=float)))
        mat = np.array(matrix)  # (n_maps, n_steps)
        for step_i, step in enumerate(steps):
            col = mat[:, step_i]
            rows.append({
                "policy":             policy,
                "step":               step,
                "mean_cumulative_ore":float(col.mean()),
                "std_cumulative_ore": float(col.std(ddof=1)) if len(col) > 1 else float("nan"),
                "n_maps":             len(col),
            })
    return pd.DataFrame(rows)


def _top_ore_rate_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, step): mean and std of top-ore hit (0/1) across maps."""
    rows = []
    for policy, pol_grp in df.groupby("policy", sort=True):
        maps = sorted(pol_grp["map_idx"].unique())
        steps = sorted(pol_grp["step"].unique())
        matrix = []
        for map_idx in maps:
            m = pol_grp[pol_grp["map_idx"] == map_idx].sort_values("step")
            matrix.append(m["top_ore"].astype(float).to_numpy())
        mat = np.array(matrix)
        for step_i, step in enumerate(steps):
            col = mat[:, step_i]
            rows.append({
                "policy":            policy,
                "step":              step,
                "mean_top_ore_rate": float(col.mean()),
                "std_top_ore_rate":  float(col.std(ddof=1)) if len(col) > 1 else float("nan"),
                "n_maps":            len(col),
            })
    return pd.DataFrame(rows)


def _total_predicted_ore_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Per (policy, step): mean and std of relative predicted-ore error across maps.

    relative_error = (total_predicted_ore - total_true_ore) / total_true_ore
    """
    rows = []
    for policy, pol_grp in df.groupby("policy", sort=True):
        maps = sorted(pol_grp["map_idx"].unique())
        steps = sorted(pol_grp["step"].unique())
        matrix = []
        for map_idx in maps:
            m = pol_grp[pol_grp["map_idx"] == map_idx].sort_values("step")
            pred = m["total_predicted_ore"].to_numpy(dtype=float)
            true_total = m["total_true_ore"].to_numpy(dtype=float)
            rel = (pred - true_total) / np.where(true_total == 0, np.nan, true_total)
            matrix.append(rel)
        mat = np.array(matrix)  # (n_maps, n_steps)
        for step_i, step in enumerate(steps):
            col = mat[:, step_i]
            rows.append({
                "policy":                           policy,
                "step":                             step,
                "mean_relative_ore_difference":     float(np.nanmean(col)),
                "std_relative_ore_difference":      float(np.nanstd(col, ddof=1)) if np.sum(~np.isnan(col)) > 1 else float("nan"),
            })
    return pd.DataFrame(rows)


def _band_half_width(std: np.ndarray, n_maps_col: np.ndarray | None, band: str) -> np.ndarray:
    if band == "ci" and n_maps_col is not None:
        return 1.96 * std / np.sqrt(np.maximum(n_maps_col, 1))
    return std


def _plot_total_predicted_ore(df: pd.DataFrame, save_path: Path, band: str = "std") -> None:
    n_maps = int(df["n_maps"].iloc[0]) if "n_maps" in df.columns else None
    n_label = f"  (n = {n_maps} maps)" if n_maps is not None else ""
    band_label = "±1 std" if band == "std" else "95% CI"
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, grp in df.groupby("policy"):
        grp = grp.sort_values("step")
        steps = grp["step"].to_numpy()
        mean = grp["mean_relative_ore_difference"].to_numpy(dtype=float)
        std = grp["std_relative_ore_difference"].to_numpy(dtype=float)
        n_col = grp["n_maps"].to_numpy(dtype=float) if "n_maps" in grp.columns else None
        hw = _band_half_width(std, n_col, band)
        line, = ax.plot(steps, mean, marker="o", label=policy)
        ax.fill_between(steps, mean - hw, mean + hw, alpha=0.2, color=line.get_color())
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Drilling step")
    ax.set_ylabel("Relative ore estimation error  (predicted − true) / true")
    ax.set_title(f"Relative ore estimation error over drilling steps  ({band_label}){n_label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_top_ore_rate(df: pd.DataFrame, save_path: Path, band: str = "std") -> None:
    n_maps = int(df["n_maps"].iloc[0]) if "n_maps" in df.columns else None
    n_label = f"  (n = {n_maps} maps)" if n_maps is not None else ""
    band_label = "±1 std" if band == "std" else "95% CI"
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, grp in df.groupby("policy"):
        grp = grp.sort_values("step")
        steps = grp["step"].to_numpy()
        mean = grp["mean_top_ore_rate"].to_numpy(dtype=float)
        std = grp["std_top_ore_rate"].to_numpy(dtype=float)
        n_col = grp["n_maps"].to_numpy(dtype=float) if "n_maps" in grp.columns else None
        hw = _band_half_width(std, n_col, band)
        line, = ax.plot(steps, mean, marker="o", label=policy.capitalize())
        ax.fill_between(steps, mean - hw, mean + hw, alpha=0.2, color=line.get_color())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Drilling step", fontsize=12)
    ax.set_ylabel("Probability of selecting a top-ore cell", fontsize=12)
    ax.set_title(f"Top-ore selection rate by drilling step  ({band_label}){n_label}", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Public helpers (reused by standalone script) ──────────────────────────────

def build_summary_by_step(df: pd.DataFrame) -> pd.DataFrame:
    """Merge top-ore-rate and total-predicted-ore tables into summary_by_step."""
    top_df = _top_ore_rate_by_step(df)
    pred_df = _total_predicted_ore_by_step(df)
    merged = top_df.merge(pred_df, on=["policy", "step"], how="outer")
    return merged


def write_summary_by_step(df: pd.DataFrame, group_dir: Path) -> None:
    """Write summary_by_step.csv and per-metric evolution PNGs (std and 95% CI variants)."""
    merged = build_summary_by_step(df)
    _prepare(merged, "step").to_csv(group_dir / "summary_by_step.csv", index=False)
    _plot_total_predicted_ore(merged, group_dir / "total_predicted_ore_evolution.png")
    _plot_total_predicted_ore(merged, group_dir / "total_predicted_ore_evolution_ci.png", band="ci")
    _plot_top_ore_rate(merged, group_dir / "top_ore_rate_evolution.png")
    _plot_top_ore_rate(merged, group_dir / "top_ore_rate_evolution_ci.png", band="ci")


_MODEL_COLORS = {"cat_var": "tab:blue", "only_ore": "tab:orange"}
_POLICY_LINESTYLES = {"uncertainty": "-", "greedy": "--", "random": ":"}
_MODEL_DISPLAY = {"cat_var": "Full model", "only_ore": "Map-only"}
_MODEL_MARKERS = {"cat_var": "o", "only_ore": "s"}
_POLICY_VISUAL = {
    "uncertainty": dict(linewidth=1.8, markersize=6, band_alpha=0.15, line_alpha=1.0),
    "greedy":      dict(linewidth=1.8, markersize=6, band_alpha=0.15, line_alpha=1.0),
    "random":      dict(linewidth=0.9, markersize=3, band_alpha=0.08, line_alpha=0.5),
}
_POLICY_VISUAL_DEFAULT = dict(linewidth=1.5, markersize=5, band_alpha=0.15, line_alpha=1.0)


def _combined_n_maps_subtitle(combined_df: pd.DataFrame) -> str:
    if "n_maps" not in combined_df.columns:
        return ""
    parts = [
        f"{model}: {int(grp['n_maps'].iloc[0])} maps"
        for model, grp in combined_df.groupby("model")
    ]
    return "\n" + " | ".join(parts)


def plot_combined_ore_evolution(combined_df: pd.DataFrame, save_path: Path, band: str = "std") -> None:
    """Combined line plot of relative ore estimation error for all model × policy combinations."""
    band_label = "±1 std" if band == "std" else "95% CI"
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, policy), grp in combined_df.groupby(["model", "policy"]):
        grp = grp.sort_values("step")
        steps = grp["step"].to_numpy()
        mean = grp["mean_relative_ore_difference"].to_numpy(dtype=float)
        std = grp["std_relative_ore_difference"].to_numpy(dtype=float)
        n_col = grp["n_maps"].to_numpy(dtype=float) if "n_maps" in grp.columns else None
        hw = _band_half_width(std, n_col, band)
        color = _MODEL_COLORS.get(model, None)
        ls = _POLICY_LINESTYLES.get(policy, "-")
        marker = _MODEL_MARKERS.get(model, "o")
        model_label = _MODEL_DISPLAY.get(model, model)
        label = f"{model_label} – {policy.capitalize()}"
        line, = ax.plot(steps, mean, marker=marker, color=color, linestyle=ls, label=label)
        ax.fill_between(steps, mean - hw, mean + hw, alpha=0.15, color=line.get_color())
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Drilling step", fontsize=12)
    ax.set_ylabel("Relative ore estimation error  (predicted − true) / true", fontsize=12)
    ax.set_title(f"Relative ore estimation error by drilling step  ({band_label})", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_top_ore_rate(combined_df: pd.DataFrame, save_path: Path, band: str = "std") -> None:
    """Combined line plot of top-ore selection rate for all model × policy combinations."""
    band_label = "±1 std" if band == "std" else "95% CI"
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, policy), grp in combined_df.groupby(["model", "policy"]):
        grp = grp.sort_values("step")
        steps = grp["step"].to_numpy()
        mean = grp["mean_top_ore_rate"].to_numpy(dtype=float)
        std = grp["std_top_ore_rate"].to_numpy(dtype=float)
        n_col = grp["n_maps"].to_numpy(dtype=float) if "n_maps" in grp.columns else None
        hw = _band_half_width(std, n_col, band)
        color = _MODEL_COLORS.get(model, None)
        ls = _POLICY_LINESTYLES.get(policy, "-")
        marker = _MODEL_MARKERS.get(model, "o")
        model_label = _MODEL_DISPLAY.get(model, model)
        label = f"{model_label} – {policy.capitalize()}"
        line, = ax.plot(steps, mean, marker=marker, color=color, linestyle=ls, label=label)
        ax.fill_between(steps, mean - hw, mean + hw, alpha=0.15, color=line.get_color())
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Drilling step", fontsize=12)
    ax.set_ylabel("Probability of selecting a top-ore cell", fontsize=12)
    ax.set_title(f"Top-ore selection rate by drilling step  ({band_label})", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        borderaxespad=0, fontsize=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _group_positions(policies: list, models: list) -> dict[tuple, float]:
    """Map (policy, model) to x-position; policy groups separated by a gap of 1."""
    pos = {}
    for p_idx, policy in enumerate(policies):
        base = p_idx * (len(models) + 1)
        for m_idx, model in enumerate(models):
            pos[(policy, model)] = base + m_idx
    return pos


def plot_combined_per_map_metrics(combined_per_map_df: pd.DataFrame, save_dir: Path) -> None:
    """Boxplot (first_top_ore_step) and violin (n_top_ore_found) saved to save_dir."""
    # Strip all string columns and coerce numeric target columns
    combined_per_map_df["policy"] = combined_per_map_df["policy"].astype(str).str.strip()
    combined_per_map_df["model"]  = combined_per_map_df["model"].astype(str).str.strip()
    for _col in ("first_top_ore_step", "n_top_ore_found"):
        combined_per_map_df[_col] = pd.to_numeric(
            combined_per_map_df[_col].astype(str).str.strip(), errors="coerce"
        )
    policies = sorted(combined_per_map_df["policy"].unique())
    models = sorted(combined_per_map_df["model"].unique())
    positions = _group_positions(policies, models)

    tick_positions = [
        (p_idx * (len(models) + 1)) + (len(models) - 1) / 2
        for p_idx in range(len(policies))
    ]
    tick_labels = [p.capitalize() for p in policies]

    # ── Figure 1: boxplot of first_top_ore_step ───────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    legend_handles: list = []
    for model in models:
        color = _MODEL_COLORS.get(model, None)
        label = _MODEL_DISPLAY.get(model, model)
        handle = plt.matplotlib.patches.Patch(facecolor=color, label=label)
        legend_handles.append(handle)
        for policy in policies:
            data = (
                combined_per_map_df.loc[
                    (combined_per_map_df["policy"] == policy)
                    & (combined_per_map_df["model"] == model),
                    "first_top_ore_step",
                ]
                .dropna()
                .to_numpy(dtype=float)
            )
            pos = positions[(policy, model)]
            bp = ax1.boxplot(
                data,
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                markerfacecolor=color, markeredgewidth=0),
            )

    n_dropped = int(combined_per_map_df["first_top_ore_step"].isna().sum())
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.set_ylabel("Drilling step", fontsize=12)
    ax1.set_title("Step at which the first top-ore cell was discovered", fontsize=14)
    if n_dropped:
        ax1.set_xlabel(f"NaN excluded (no top-ore found): {n_dropped} map–policy pairs", fontsize=10)
    ax1.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
               borderaxespad=0, fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    fig1.savefig(save_dir / "first_top_ore_step_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: boxplot of n_top_ore_found ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for model in models:
        color = _MODEL_COLORS.get(model, None)
        for policy in policies:
            data = (
                combined_per_map_df.loc[
                    (combined_per_map_df["policy"] == policy)
                    & (combined_per_map_df["model"] == model),
                    "n_top_ore_found",
                ]
                .dropna()
                .to_numpy(dtype=float)
            )
            pos = positions[(policy, model)]
            ax2.boxplot(
                data,
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                markerfacecolor=color, markeredgewidth=0),
            )

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Number of top-ore cells found per run", fontsize=14)
    ax2.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
               borderaxespad=0, fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.savefig(save_dir / "n_top_ore_found_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: violin of n_top_ore_found ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for model in models:
        color = _MODEL_COLORS.get(model, None)
        for policy in policies:
            data = (
                combined_per_map_df.loc[
                    (combined_per_map_df["policy"] == policy)
                    & (combined_per_map_df["model"] == model),
                    "n_top_ore_found",
                ]
                .dropna()
                .to_numpy(dtype=float)
            )
            pos = positions[(policy, model)]
            if len(np.unique(data)) < 2:
                ax3.boxplot(data, positions=[pos], widths=0.5, patch_artist=True,
                            manage_ticks=False,
                            boxprops=dict(facecolor=color, alpha=0.7),
                            medianprops=dict(color="black", linewidth=1.5))
                continue
            parts = ax3.violinplot(data, positions=[pos], widths=0.7,
                                   showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax3.boxplot(
                data,
                positions=[pos],
                widths=0.12,
                manage_ticks=False,
                patch_artist=True,
                boxprops=dict(facecolor="white", linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0),
                capprops=dict(linewidth=0),
                flierprops=dict(marker=""),
            )

    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, fontsize=11)
    ax3.tick_params(axis="y", labelsize=11)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Number of top-ore cells found per run", fontsize=14)
    ax3.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
               borderaxespad=0, fontsize=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    fig3.savefig(save_dir / "n_top_ore_found_distribution_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # ── Figure 4: population-pyramid split per policy ────────────────────────
    max_val = int(combined_per_map_df["n_top_ore_found"].max())
    values = np.arange(max_val + 1)
    label_min_pct = 0.005   # skip bar labels below this proportion

    # Pre-compute proportions and means for all (policy, model) pairs
    prop_store:  dict[tuple, np.ndarray] = {}
    mean_store:  dict[tuple, float] = {}
    raw_store:   dict[tuple, np.ndarray] = {}
    for policy in policies:
        for model in models:
            data = (
                combined_per_map_df.loc[
                    (combined_per_map_df["policy"] == policy)
                    & (combined_per_map_df["model"] == model),
                    "n_top_ore_found",
                ]
                .dropna()
                .to_numpy(dtype=float)
            )
            raw_store[(policy, model)] = data
            n_total = len(data)
            prop_store[(policy, model)] = np.array(
                [np.sum(data == v) / n_total if n_total else 0.0 for v in values]
            )
            mean_store[(policy, model)] = float(data.mean()) if n_total else 0.0

    global_max_prop = max(p.max() for p in prop_store.values()) if prop_store else 0.1
    # Extra space: room for percentage labels + mean annotation text
    x_lim = global_max_prop * 1.55

    fig4, axes4 = plt.subplots(
        1, len(policies), sharey=True,
        figsize=(5.5 * len(policies), 7),
        gridspec_kw={"wspace": 0.35},
    )
    if len(policies) == 1:
        axes4 = [axes4]

    bar_height = 0.65
    cat_model, only_model = models[0], models[1]   # cat_var left, only_ore right
    cat_color  = _MODEL_COLORS.get(cat_model,  "tab:blue")
    only_color = _MODEL_COLORS.get(only_model, "tab:orange")

    for ax, policy in zip(axes4, policies):
        cat_props  = prop_store[(policy, cat_model)]
        only_props = prop_store[(policy, only_model)]

        ax.barh(values, -cat_props,  height=bar_height, color=cat_color,  alpha=0.75)
        ax.barh(values,  only_props, height=bar_height, color=only_color, alpha=0.75)

        # Percentage labels next to each bar
        for v, p in enumerate(cat_props):
            if p >= label_min_pct:
                ax.text(-p - 0.004, v, f"{p:.0%}", ha="right", va="center", fontsize=8)
        for v, p in enumerate(only_props):
            if p >= label_min_pct:
                ax.text(p + 0.004, v, f"{p:.0%}", ha="left", va="center", fontsize=8)

        # Mean lines and text annotations in top-right corner
        cat_mean  = mean_store[(policy, cat_model)]
        only_mean = mean_store[(policy, only_model)]
        cat_label  = f"{_MODEL_DISPLAY.get(cat_model,  cat_model)}  mean = {cat_mean:.1f}"
        only_label = f"{_MODEL_DISPLAY.get(only_model, only_model)}  mean = {only_mean:.1f}"
        ax.axhline(cat_mean,  color=cat_color,  linewidth=0.9, linestyle="--", alpha=0.5)
        ax.axhline(only_mean, color=only_color, linewidth=0.9, linestyle="--", alpha=0.5)
        ax.text(0.98, 0.97, cat_label,  color=cat_color,  transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, style="italic")
        ax.text(0.98, 0.93, only_label, color=only_color, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, style="italic")

        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_title(policy.capitalize(), fontsize=13, pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        # Symmetric absolute-value x-tick labels
        step = 0.1 if global_max_prop > 0.15 else 0.05
        ticks_pos = np.arange(0, global_max_prop * 1.1 + step, step)
        all_ticks = np.concatenate([-ticks_pos[1:][::-1], ticks_pos])
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([f"{abs(t):.0%}" for t in all_ticks], fontsize=9,
                           rotation=45, ha="right")
        ax.tick_params(axis="x", labelsize=9)

    axes4[0].set_yticks(values)
    axes4[0].set_yticklabels(values, fontsize=11)
    axes4[0].set_ylabel("Number of top-ore cells found", fontsize=12)

    handles = [
        plt.matplotlib.patches.Patch(facecolor=cat_color,  alpha=0.75,
                                     label=_MODEL_DISPLAY.get(cat_model,  cat_model)),
        plt.matplotlib.patches.Patch(facecolor=only_color, alpha=0.75,
                                     label=_MODEL_DISPLAY.get(only_model, only_model)),
    ]
    axes4[-1].legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                     borderaxespad=0, fontsize=10)
    fig4.suptitle(
        "Distribution of discovered top-ore locations after 10 drilling steps",
        fontsize=14, y=1.02,
    )
    fig4.savefig(save_dir / "n_top_ore_found_split_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # ── Figure 5: population-pyramid for first_top_ore_step ──────────────────
    max_step5 = 10
    steps_range5 = np.arange(1, max_step5 + 1)
    # y=0: No discovery; y=1..10: steps 1..10
    y_no_disc = 0
    y_positions5 = np.arange(max_step5 + 1)           # [0, 1, ..., 10]
    y_tick_labels5 = ["No discovery"] + [str(int(s)) for s in steps_range5]

    prop5_store: dict[tuple, np.ndarray] = {}
    mean5_store: dict[tuple, float] = {}
    for policy in policies:
        for model in models:
            raw5 = combined_per_map_df.loc[
                (combined_per_map_df["policy"] == policy)
                & (combined_per_map_df["model"] == model),
                "first_top_ore_step",
            ]
            n_total5 = len(raw5)
            p_no_disc5 = float(raw5.isna().sum()) / n_total5 if n_total5 else 0.0
            raw5_num = raw5.dropna().to_numpy(dtype=float)
            p_steps5 = np.array(
                [float(np.sum(raw5_num == s)) / n_total5 if n_total5 else 0.0
                 for s in steps_range5]
            )
            prop5_store[(policy, model)] = np.concatenate([[p_no_disc5], p_steps5])
            mean5_store[(policy, model)] = float(raw5_num.mean()) if len(raw5_num) > 0 else float("nan")

    global_max_prop5 = max(p.max() for p in prop5_store.values()) if prop5_store else 0.1
    x_lim5 = global_max_prop5 * 1.55
    label_min_pct5 = 0.005

    fig5, axes5 = plt.subplots(
        1, len(policies), sharey=True,
        figsize=(5.5 * len(policies), 7),
        gridspec_kw={"wspace": 0.35},
    )
    if len(policies) == 1:
        axes5 = [axes5]

    bar_height5 = 0.65

    for ax, policy in zip(axes5, policies):
        cat_props5  = prop5_store[(policy, cat_model)]
        only_props5 = prop5_store[(policy, only_model)]

        ax.barh(y_positions5, -cat_props5,  height=bar_height5, color=cat_color,  alpha=0.75)
        ax.barh(y_positions5,  only_props5, height=bar_height5, color=only_color, alpha=0.75)

        for yi, p in zip(y_positions5, cat_props5):
            if p >= label_min_pct5:
                ax.text(-p - 0.004, yi, f"{p:.0%}", ha="right", va="center", fontsize=8)
        for yi, p in zip(y_positions5, only_props5):
            if p >= label_min_pct5:
                ax.text(p + 0.004, yi, f"{p:.0%}", ha="left", va="center", fontsize=8)

        cat_mean5  = mean5_store[(policy, cat_model)]
        only_mean5 = mean5_store[(policy, only_model)]
        cat_label5  = f"{_MODEL_DISPLAY.get(cat_model,  cat_model)}  mean = {cat_mean5:.1f}"
        only_label5 = f"{_MODEL_DISPLAY.get(only_model, only_model)}  mean = {only_mean5:.1f}"
        if not np.isnan(cat_mean5):
            ax.axhline(cat_mean5,  color=cat_color,  linewidth=0.9, linestyle="--", alpha=0.5)
            ax.text(0.98, 0.97, cat_label5,  color=cat_color,  transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, style="italic")
        if not np.isnan(only_mean5):
            ax.axhline(only_mean5, color=only_color, linewidth=0.9, linestyle="--", alpha=0.5)
            ax.text(0.98, 0.93, only_label5, color=only_color, transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, style="italic")

        # Dotted separator between "No discovery" and steps
        ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlim(-x_lim5, x_lim5)
        ax.set_title(policy.capitalize(), fontsize=13, pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        step5 = 0.1 if global_max_prop5 > 0.15 else 0.05
        ticks_pos5 = np.arange(0, global_max_prop5 * 1.1 + step5, step5)
        all_ticks5 = np.concatenate([-ticks_pos5[1:][::-1], ticks_pos5])
        ax.set_xticks(all_ticks5)
        ax.set_xticklabels([f"{abs(t):.0%}" for t in all_ticks5], fontsize=9,
                           rotation=45, ha="right")

    axes5[0].set_yticks(y_positions5)
    axes5[0].set_yticklabels(y_tick_labels5, fontsize=10)
    axes5[0].set_ylabel("First top-ore discovery step", fontsize=12)

    handles5 = [
        plt.matplotlib.patches.Patch(facecolor=cat_color,  alpha=0.75,
                                     label=_MODEL_DISPLAY.get(cat_model,  cat_model)),
        plt.matplotlib.patches.Patch(facecolor=only_color, alpha=0.75,
                                     label=_MODEL_DISPLAY.get(only_model, only_model)),
    ]
    axes5[-1].legend(handles=handles5, bbox_to_anchor=(1.02, 1), loc="upper left",
                     borderaxespad=0, fontsize=10)
    fig5.suptitle(
        "Distribution of first top-ore discovery step",
        fontsize=14, y=1.02,
    )
    fig5.savefig(save_dir / "first_top_ore_step_pyramid.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)


# ── Public entry point ────────────────────────────────────────────────────────

def _write_group(df: pd.DataFrame, group_dir: Path) -> None:
    """Write all evaluation CSVs and plots for one orebody group into *group_dir*."""
    group_dir.mkdir(parents=True, exist_ok=True)

    per_map_df = _build_per_map_metrics(df)
    _prepare(per_map_df, "map_idx").to_csv(group_dir / "per_map_metrics.csv", index=False)

    agg_df = _aggregate_metrics(per_map_df)
    _prepare(agg_df).to_csv(group_dir / "aggregate_metrics.csv", index=False)

    cum_df = _cumulative_ore_by_step(df)
    _prepare(cum_df, "step").to_csv(group_dir / "cumulative_ore_by_step.csv", index=False)

    write_summary_by_step(df, group_dir)


def run_evaluation(step_rows: list[dict], out_dir: Path) -> None:
    """Compute all policy evaluation metrics and write CSVs to *out_dir*.

    Outputs are split by ore-body count into subdirectories (no_orebodies/,
    one_orebody/, two_orebodies/). The n_bodies column is used for routing only
    and is dropped before writing any CSV.

    Parameters
    ----------
    step_rows:
        Accumulated list of per-step dicts from the POMDP loop. Each dict must
        contain: policy, step, loc_i, loc_j, true_ore, predicted_ore,
        predicted_uncertainty, total_predicted_ore, total_true_ore, top_ore,
        map_idx, n_bodies.
    out_dir:
        prediction_summaries/ directory.
    """
    if not step_rows:
        print("evaluate_policies: no step data, skipping.")
        return

    df = pd.DataFrame(step_rows)

    if "map_idx" not in df.columns or df["map_idx"].isna().all():
        print("evaluate_policies: map_idx missing from step rows, skipping.")
        return

    df["top_ore"] = df["top_ore"].astype(bool)
    df = df.sort_values(["map_idx", "policy", "step"]).reset_index(drop=True)

    groups = sorted(df["n_bodies"].unique()) if "n_bodies" in df.columns else [None]

    for nb in groups:
        if nb is None:
            group_df = df.drop(columns=["n_bodies"], errors="ignore")
            group_dir = out_dir
        else:
            group_df = df[df["n_bodies"] == nb].drop(columns=["n_bodies"])
            group_dir = out_dir / _orebody_dir_name(int(nb))

        print(f"  [{_orebody_dir_name(int(nb)) if nb is not None else 'all'}] Computing metrics ({len(group_df['map_idx'].unique())} maps)...")
        _write_group(group_df, group_dir)
        print(f"    -> {group_dir}")
