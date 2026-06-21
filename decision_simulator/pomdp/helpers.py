"""Shared helpers for POMDP evaluation scripts.

Contains the thin drilling environment, belief decoders, and belief updaters
that were originally defined inside pomdp.py's __main__ block. Extracted here
so that standalone evaluation scripts can import them without running pomdp.py.
"""

from __future__ import annotations

import numpy as np
import torch

from decision_simulator.neural_belief.models.belief_models.borehole_encoders.autoencoder import (
    standardise,
)
from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.observations.borehole_observations import (
    BoreholeObservationState,
)


def decode_belief(
    output,
    normalizer,
    obs_state: BoreholeObservationState,
    step: int | None,
) -> BeliefState:
    """Convert raw model output tuple/tensor into a BeliefState."""
    if isinstance(output, tuple) and len(output) >= 2:
        pred_ore_t, pred_unc_t = output[0], output[1]
    elif isinstance(output, tuple):
        pred_ore_t, pred_unc_t = output[0], None
    else:
        pred_ore_t, pred_unc_t = output, None

    pred_ore_np: np.ndarray = (
        normalizer.inverse_tensor(pred_ore_t).squeeze().cpu().numpy()
    )
    pred_unc_np: np.ndarray | None = (
        pred_unc_t.squeeze().cpu().numpy() if pred_unc_t is not None else None
    )
    return BeliefState(
        predicted_ore_map=pred_ore_np,
        predicted_uncertainty_map=pred_unc_np,
        observed_mask=obs_state.get_observed_mask(),
        step=step,
    )


class HDF5DrillEnv:
    """Thin drilling environment backed by one map entry from a MapPool."""

    def __init__(
        self,
        bh_raw: np.ndarray,
        target: np.ndarray,
        rocks: np.ndarray | None,
        n_x: int,
        n_y: int,
        norm_stats: dict,
        variable_names: list[str],
    ) -> None:
        self._bh = bh_raw          # (n_x*n_y, V, D) float32, raw
        self._target = target      # (n_x, n_y) float32
        self._rocks = rocks        # (n_x*n_y, D) int8, or None
        self.n_x = n_x
        self.n_y = n_y
        self._norm_stats = norm_stats
        self._variable_names = variable_names

    def get_true_ore_map(self) -> np.ndarray:
        return self._target

    def drill(self, i: int, j: int) -> tuple[np.ndarray, float]:
        flat = i * self.n_y + j
        raw = self._bh[flat]  # (V, D)
        bh_std = standardise(raw, self._norm_stats, self._variable_names)
        bh_std = np.nan_to_num(bh_std, nan=0.0).astype(np.float32)
        return bh_std, float(self._target[i, j])

    def get_rock_ids(self, i: int, j: int) -> np.ndarray:
        if self._rocks is None:
            return np.zeros(self._bh.shape[-1], dtype=np.int64)
        flat = i * self.n_y + j
        return self._rocks[flat].astype(np.int64)


class CatVarUpdater:
    """Belief updater for CatVarEncoder.

    Reads rock-type labels from the HDF5 environment to match the
    categorical inputs the model was trained on.
    """

    def __init__(self, model, normalizer, device: str, env: HDF5DrillEnv) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self._env = env
        model.eval()

    def update(
        self, obs_state: BoreholeObservationState, step: int | None = None
    ) -> BeliefState:
        inputs = obs_state.to_model_inputs()
        ore_vals_norm = self.normalizer.transform(inputs["ore_vals"])

        rock_ids_np = np.stack(
            [self._env.get_rock_ids(i, j) for i, j in obs_state._positions]
        )  # (K, D) int64

        bh_t  = torch.from_numpy(inputs["boreholes"]).unsqueeze(0).to(self.device)
        rid_t = torch.from_numpy(rock_ids_np).unsqueeze(0).to(self.device)
        ov_t  = torch.from_numpy(ore_vals_norm).unsqueeze(0).to(self.device)
        pos_t = torch.from_numpy(inputs["positions"]).unsqueeze(0).to(self.device)
        pm_t  = torch.from_numpy(inputs["padding_mask"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(bh_t, rid_t, ov_t, pos_t, pm_t)
        return decode_belief(output, self.normalizer, obs_state, step)


class OreOnlyNullUpdater:
    """Belief updater for OreOnlyNullEncoder.

    The null encoder ignores borehole logs, rock_ids, and formation_ids;
    zero tensors are passed for all three to satisfy the API.
    """

    def __init__(self, model, normalizer, device: str, env: HDF5DrillEnv) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self._env = env
        model.eval()

    def update(
        self, obs_state: BoreholeObservationState, step: int | None = None
    ) -> BeliefState:
        inputs = obs_state.to_model_inputs()
        ore_vals_norm = self.normalizer.transform(inputs["ore_vals"])

        K, V, D = inputs["boreholes"].shape

        bh_t   = torch.from_numpy(inputs["boreholes"]).unsqueeze(0).to(self.device)
        zero_r = torch.zeros(1, K, D, dtype=torch.long, device=self.device)
        zero_f = torch.zeros(1, K, D, dtype=torch.long, device=self.device)
        ov_t   = torch.from_numpy(ore_vals_norm).unsqueeze(0).to(self.device)
        pos_t  = torch.from_numpy(inputs["positions"]).unsqueeze(0).to(self.device)
        pm_t   = torch.from_numpy(inputs["padding_mask"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(bh_t, zero_r, zero_f, ov_t, pos_t, pm_t)
        return decode_belief(output, self.normalizer, obs_state, step)


def make_updater(
    model_name: str,
    model,
    normalizer,
    device: str,
    env: HDF5DrillEnv,
) -> CatVarUpdater | OreOnlyNullUpdater:
    """Factory: return the correct updater class for the given model name."""
    if model_name == "cat_var":
        return CatVarUpdater(model, normalizer, device, env)
    return OreOnlyNullUpdater(model, normalizer, device, env)
