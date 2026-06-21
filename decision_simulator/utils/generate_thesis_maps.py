"""Generate thesis-quality evolution maps and supporting data for 10 one-orebody maps.

For each map × policy the script:
  - Runs the full POMDP episode
  - Captures full 2D spatial arrays at SELECTED_STEPS = {1, 2, 3, 5, 10}
  - Saves a modified evolution PNG  (Obs | Pred ore | Uncertainty | Abs error | True ore)
  - Saves a .npz data file with all 2D arrays so the figure can be reproduced later

Output layout
-------------
decision_simulator/results/thesis/{model}/{policy}/
  map_{idx:02d}_evolution.png
  map_{idx:02d}_data.npz

Usage
-----
python -m decision_simulator.utils.generate_thesis_maps
python -m decision_simulator.utils.generate_thesis_maps --model ore_only_null
python -m decision_simulator.utils.generate_thesis_maps --n-maps 5 --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SELECTED_STEPS = {1, 2, 3, 5, 10}
_BELIEFS_DIR = Path(__file__).parent.parent / "pomdp" / "beliefs"
_RESULTS_DIR = Path(__file__).parent.parent / "results" / "thesis"


# ── Plot and data functions (numpy / matplotlib only — no circular imports) ───

def _scatter_drills(ax, drill_rows, drill_cols, newest_ij) -> None:
    """Scatter drill markers: previous in blue, newest in red."""
    ni, nj = newest_ij
    newest_mask = (drill_rows == ni) & (drill_cols == nj)
    prev_mask   = ~newest_mask
    if prev_mask.any():
        ax.scatter(drill_rows[prev_mask], drill_cols[prev_mask],
                   c="steelblue", s=20, marker="x", linewidths=1.0, zorder=5,
                   label="Previous drill")
    ax.scatter([ni], [nj], c="red", s=90, marker="o", zorder=6,
               edgecolors="darkred", linewidths=0.8, label="Newest drill")


def plot_thesis_evolution(steps: list[dict], save_path: Path, title: str = "") -> None:
    """Evolution grid: Obs | Belief map | Uncertainty | Abs error | True ore.

    Previous drill locations are shown in blue; the newest selected location
    is highlighted in red so the reader can follow the decision sequence:
    Uncertainty -> Selected location -> Updated belief map.
    """
    if not steps:
        return

    vmax = float(max(
        max(s["true_ore_map"].max() for s in steps),
        max(s["predicted_ore_map"].max() for s in steps),
        1e-3,
    ))

    n_rows = len(steps)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=22)

    col_titles = [
        "Observations (a)",
        "Belief map (b)",
        "Predicted uncertainty (c)",
        "Absolute error (d)",
        "True ore map (e)",
    ]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=16)

    kw_viridis = dict(origin="lower", cmap="viridis")

    for row, s in enumerate(steps):
        step         = s["step"]
        sparse       = s["sparse_ore_map"]
        mask         = s["observation_mask"]
        true_ore     = s["true_ore_map"]
        pred_ore     = s["predicted_ore_map"]
        pred_unc     = s["predicted_uncertainty_map"]
        newest_ij    = s["newest_drill"]
        drill_rows, drill_cols = np.where(mask > 0)
        n_drills     = int(mask.sum())

        # Col 0 — observations (all drills; newest highlighted)
        ax = axes[row, 0]
        im = ax.imshow(sparse.T, vmin=0, vmax=vmax, **kw_viridis)
        _scatter_drills(ax, drill_rows, drill_cols, newest_ij)
        fig.colorbar(im, ax=ax, fraction=0.046)
        drill_label = f"{step} drill" if step == 1 else f"{step} drills"
        ax.set_ylabel(drill_label, fontsize=22)

        # Col 1 — belief map (all drills; newest highlighted)
        ax = axes[row, 1]
        im = ax.imshow(pred_ore.T, vmin=0, vmax=vmax, **kw_viridis)
        _scatter_drills(ax, drill_rows, drill_cols, newest_ij)
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Col 2 — predicted uncertainty (newest drill as focal point)
        ax = axes[row, 2]
        if pred_unc is not None:
            im = ax.imshow(pred_unc.T, origin="lower", cmap="hot_r")
            _scatter_drills(ax, drill_rows, drill_cols, newest_ij)
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "No uncertainty", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            ax.axis("off")

        # Col 3 — absolute error (no markers)
        ax = axes[row, 3]
        abs_err = np.abs(pred_ore - true_ore)
        im = ax.imshow(abs_err.T, origin="lower", vmin=0, vmax=vmax, cmap="Reds")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # Col 4 — true ore map (no markers)
        ax = axes[row, 4]
        im = ax.imshow(true_ore.T, vmin=0, vmax=vmax, **kw_viridis)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_step_data(steps: list[dict], save_path: Path) -> None:
    """Save all 2D arrays for the selected steps to a compressed .npz file."""
    arrays: dict[str, np.ndarray] = {}
    step_nums: list[int] = []
    for s in steps:
        n = s["step"]
        step_nums.append(n)
        arrays[f"step{n:02d}_sparse_ore_map"]   = s["sparse_ore_map"]
        arrays[f"step{n:02d}_observation_mask"]  = s["observation_mask"]
        arrays[f"step{n:02d}_true_ore_map"]      = s["true_ore_map"]
        arrays[f"step{n:02d}_predicted_ore_map"] = s["predicted_ore_map"]
        arrays[f"step{n:02d}_newest_drill"]      = np.array(s["newest_drill"], dtype=np.int32)
        if s["predicted_uncertainty_map"] is not None:
            arrays[f"step{n:02d}_predicted_uncertainty_map"] = s["predicted_uncertainty_map"]
    arrays["selected_steps"] = np.array(step_nums, dtype=np.int32)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **arrays)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # All neural_belief / torch imports are deferred here to avoid the circular
    # import that triggers when they are loaded at module level.
    import h5py
    import torch
    from collections import defaultdict

    from decision_simulator.neural_belief.map_hdf5 import HDF5MapStore, MapPool
    from decision_simulator.neural_belief.models.belief_models.borehole_encoders.autoencoder import (
        standardise,
    )
    from decision_simulator.neural_belief.training.belief_models.end_to_end.train_cat_var_encoder import (
        load_cat_var_checkpoint,
    )
    from decision_simulator.neural_belief.training.belief_models.end_to_end.train_ore_only_null_encoder import (
        load_ore_only_null_checkpoint,
    )
    from decision_simulator.pomdp.beliefs.belief_state import BeliefState
    from decision_simulator.pomdp.observations.borehole_observations import (
        BoreholeObservationState,
    )
    from decision_simulator.pomdp.policies.greedy_yield_policy import GreedyYieldPolicy
    from decision_simulator.pomdp.policies.random_policy import RandomPolicy
    from decision_simulator.pomdp.policies.uncertainty_policy import UncertaintyPolicy
    from decision_simulator.pomdp.pomdp import run_fixed_budget_episode
    from decision_simulator.resources import load_decision_resources

    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["cat_var", "ore_only_null"], default="cat_var")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path,
                        default=Path("C:/validation_datasets/One_orebody"))
    parser.add_argument("--n-maps", type=int, default=20,
                        help="Number of one-orebody maps to process (default: 20).")
    parser.add_argument("--budget", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpt_name = ("cat_var_best_500.pt" if args.model == "cat_var"
                     else "ore_only_null_best_500.pt")
        args.checkpoint = _BELIEFS_DIR / ckpt_name

    model_label = "cat_var" if args.model == "cat_var" else "only_ore"
    out_dir = _RESULTS_DIR / model_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Helper classes (defined here to access deferred imports) ─────────────

    def _decode_belief(output, normalizer, obs_state, step):
        if isinstance(output, tuple) and len(output) >= 2:
            pred_ore_t, pred_unc_t = output[0], output[1]
        elif isinstance(output, tuple):
            pred_ore_t, pred_unc_t = output[0], None
        else:
            pred_ore_t, pred_unc_t = output, None
        pred_ore_np = normalizer.inverse_tensor(pred_ore_t).squeeze().cpu().numpy()
        pred_unc_np = pred_unc_t.squeeze().cpu().numpy() if pred_unc_t is not None else None
        return BeliefState(
            predicted_ore_map=pred_ore_np,
            predicted_uncertainty_map=pred_unc_np,
            observed_mask=obs_state.get_observed_mask(),
            step=step,
        )

    class _HDF5DrillEnv:
        def __init__(self, bh_raw, target, rocks, n_x, n_y, norm_stats, variable_names):
            self._bh             = bh_raw
            self._target         = target
            self._rocks          = rocks
            self.n_x             = n_x
            self.n_y             = n_y
            self._norm_stats     = norm_stats
            self._variable_names = variable_names

        def get_true_ore_map(self):
            return self._target

        def drill(self, i, j):
            flat   = i * self.n_y + j
            bh_std = standardise(self._bh[flat], self._norm_stats, self._variable_names)
            bh_std = np.nan_to_num(bh_std, nan=0.0).astype(np.float32)
            return bh_std, float(self._target[i, j])

        def get_rock_ids(self, i, j):
            if self._rocks is None:
                return np.zeros(self._bh.shape[-1], dtype=np.int64)
            return self._rocks[i * self.n_y + j].astype(np.int64)

    class _CatVarUpdater:
        def __init__(self, model, normalizer, device, env):
            self.model = model; self.normalizer = normalizer
            self.device = device; self._env = env
            model.eval()

        def update(self, obs_state, step=None):
            inputs      = obs_state.to_model_inputs()
            ore_norm    = self.normalizer.transform(inputs["ore_vals"])
            rock_ids_np = np.stack([self._env.get_rock_ids(i, j)
                                    for i, j in obs_state._positions])
            bh_t  = torch.from_numpy(inputs["boreholes"]).unsqueeze(0).to(self.device)
            rid_t = torch.from_numpy(rock_ids_np).unsqueeze(0).to(self.device)
            ov_t  = torch.from_numpy(ore_norm).unsqueeze(0).to(self.device)
            pos_t = torch.from_numpy(inputs["positions"]).unsqueeze(0).to(self.device)
            pm_t  = torch.from_numpy(inputs["padding_mask"]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(bh_t, rid_t, ov_t, pos_t, pm_t)
            return _decode_belief(output, self.normalizer, obs_state, step)

    class _OreOnlyNullUpdater:
        def __init__(self, model, normalizer, device, env):
            self.model = model; self.normalizer = normalizer
            self.device = device; self._env = env
            model.eval()

        def update(self, obs_state, step=None):
            inputs   = obs_state.to_model_inputs()
            ore_norm = self.normalizer.transform(inputs["ore_vals"])
            K, V, D  = inputs["boreholes"].shape
            bh_t   = torch.from_numpy(inputs["boreholes"]).unsqueeze(0).to(self.device)
            zero_r = torch.zeros(1, K, D, dtype=torch.long, device=self.device)
            zero_f = torch.zeros(1, K, D, dtype=torch.long, device=self.device)
            ov_t   = torch.from_numpy(ore_norm).unsqueeze(0).to(self.device)
            pos_t  = torch.from_numpy(inputs["positions"]).unsqueeze(0).to(self.device)
            pm_t   = torch.from_numpy(inputs["padding_mask"]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(bh_t, zero_r, zero_f, ov_t, pos_t, pm_t)
            return _decode_belief(output, self.normalizer, obs_state, step)

    def _make_step_callback(store, true_ore_map):
        def callback(step, belief_state, obs_state):
            if step not in SELECTED_STEPS:
                return
            # obs_state._positions[-1] is the newest drill (added after policy decision)
            newest = obs_state._positions[-1]
            store.append(dict(
                step                      = step,
                newest_drill              = newest,
                sparse_ore_map            = obs_state.get_sparse_ore_map(),
                observation_mask          = obs_state.get_observed_mask(),
                true_ore_map              = true_ore_map.copy(),
                predicted_ore_map         = belief_state.predicted_ore_map.copy(),
                predicted_uncertainty_map = (
                    belief_state.predicted_uncertainty_map.copy()
                    if belief_state.predicted_uncertainty_map is not None else None
                ),
            ))
        return callback

    # ── Load resources and model ──────────────────────────────────────────────
    print("Loading resources...")
    resources, _ = load_decision_resources(borehole_encoder="jepa", device=args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    if args.model == "cat_var":
        model, _, normalizer, _ = load_cat_var_checkpoint(args.checkpoint, device=args.device)
    else:
        model, _, normalizer, _ = load_ore_only_null_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    # ── Scan shards for one-orebody maps ──────────────────────────────────────
    print(f"Scanning {args.map_dir} for one-orebody maps...")
    shard_files = (
        sorted(args.map_dir.glob("maps_0_and_1_orebodies_*.h5"))
        or sorted(args.map_dir.glob("maps_stratified_*.h5"))
        or sorted(args.map_dir.glob("maps_[0-9][0-9][0-9][0-9][0-9]_*.h5"))
    )
    if not shard_files:
        raise FileNotFoundError(f"No HDF5 shards found in {args.map_dir}.")

    selected: list[tuple[Path, int]] = []
    for sf in shard_files:
        if len(selected) >= args.n_maps:
            break
        with h5py.File(sf, "r") as hf:
            for local_idx, nb in enumerate(hf["n_bodies"][:]):
                if int(nb) == 1:
                    selected.append((sf, local_idx))
                    if len(selected) >= args.n_maps:
                        break

    if len(selected) < args.n_maps:
        print(f"  Warning: only {len(selected)} one-orebody maps found (requested {args.n_maps}).")
    print(f"  Selected {len(selected)} maps.")

    # Load selected maps grouped by shard
    by_shard: dict[Path, list[int]] = defaultdict(list)
    for sf, local_idx in selected:
        by_shard[sf].append(local_idx)

    pools = [HDF5MapStore(sf).load_subset(idxs) for sf, idxs in by_shard.items()]
    if len(pools) == 1:
        map_pool = pools[0]
    else:
        p0 = pools[0]
        map_pool = MapPool(
            borehole_arrays=[b for p in pools for b in p.borehole_arrays],
            targets        =[t for p in pools for t in p.targets],
            drill_patterns =[d for p in pools for d in p.drill_patterns],
            cfg            =dataclasses.replace(p0.cfg, n_maps=len(selected)),
            n_x            =p0.n_x,
            n_y            =p0.n_y,
            rocks_arrays   =[r for p in pools if p.rocks_arrays for r in p.rocks_arrays] or None,
        )

    print(f"  Pool: {map_pool.pool_size} maps  "
          f"(n_x={map_pool.n_x}, n_y={map_pool.n_y}, "
          f"rocks={'yes' if map_pool.rocks_arrays else 'no'})")

    # ── Episode loop ──────────────────────────────────────────────────────────
    policy_list = [
        ("uncertainty", UncertaintyPolicy()),
        ("greedy",      GreedyYieldPolicy()),
        ("random",      RandomPolicy(seed=42)),
    ]

    for map_idx in range(map_pool.pool_size):
        print(f"\n{'-' * 50}")
        print(f"Map {map_idx:02d} / {map_pool.pool_size - 1}")

        rocks_arr = map_pool.rocks_arrays[map_idx] if map_pool.rocks_arrays else None
        env = _HDF5DrillEnv(
            bh_raw        =map_pool.borehole_arrays[map_idx],
            target        =map_pool.targets[map_idx],
            rocks         =rocks_arr,
            n_x           =map_pool.n_x,
            n_y           =map_pool.n_y,
            norm_stats    =resources.norm_stats,
            variable_names=resources.variable_names,
        )
        center = (env.n_x // 2, env.n_y // 2)

        for policy_name, policy in policy_list:
            print(f"  Policy: {policy_name}")
            policy_out = out_dir / policy_name
            policy_out.mkdir(parents=True, exist_ok=True)

            updater = (_CatVarUpdater(model, normalizer, args.device, env)
                       if args.model == "cat_var"
                       else _OreOnlyNullUpdater(model, normalizer, args.device, env))

            collected: list[dict] = []
            run_fixed_budget_episode(
                environment      =env,
                belief_updater   =updater,
                policy           =policy,
                drilling_budget  =args.budget,
                initial_locations=[center],
                step_callback    =_make_step_callback(collected, env.get_true_ore_map()),
            )

            png_path = policy_out / f"map_{map_idx:02d}_evolution.png"
            npz_path = policy_out / f"map_{map_idx:02d}_data.npz"

            plot_thesis_evolution(collected, png_path,
                                  title=f"{policy_name} | Map {map_idx:02d}")
            save_step_data(collected, npz_path)
            print(f"    -> {png_path}")
            print(f"    -> {npz_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
