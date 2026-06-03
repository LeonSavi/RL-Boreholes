from __future__ import annotations

import numpy as np
import torch

from decision_simulator.neural_belief.training_utils import TargetNormalizer
from decision_simulator.pomdp.beliefs.belief_state import BeliefState
from decision_simulator.pomdp.observations.borehole_observations import (
    BoreholeObservationState,
)


class NeuralBeliefUpdater:
    """Converts a BoreholeObservationState into a BeliefState via a pretrained model.

    Supports models that return:
    - a single tensor   (pred_ore only)
    - a 2-tuple         (pred_ore, pred_uncertainty)
    - a 3-tuple         (pred_ore, pred_uncertainty, latent)

    Normalisation contract
    ----------------------
    Borehole profiles in BoreholeObservationState are already standardised
    (z-scored per geological variable) — they are passed to the model as-is.

    Ore values in BoreholeObservationState are raw (ore-value space).
    This class applies normalizer.transform() before passing them to the model,
    matching the normalisation used during training.

    The predicted ore map output is denormalised via normalizer.inverse_tensor()
    so that BeliefState.predicted_ore_map is in ore-value space.

    The predicted uncertainty map is left in normalised model space because
    there is no meaningful inverse transform for uncertainty estimates; policies
    use it only in a relative sense (higher = more uncertain).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer: TargetNormalizer,
        device: str,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.model.eval()

    def update(
        self,
        obs_state: BoreholeObservationState,
        step: int | None = None,
    ) -> BeliefState:
        """Run the belief model on current observations and return a BeliefState.

        Parameters
        ----------
        obs_state : BoreholeObservationState
            Current set of drilled observations.
        step : int, optional
            Drill step index to embed in the returned BeliefState.

        Returns
        -------
        BeliefState
            predicted_ore_map in ore-value space; uncertainty in normalised space.
        """
        inputs = obs_state.to_model_inputs()

        # Normalise ore values to match training distribution
        ore_vals_norm = self.normalizer.transform(inputs["ore_vals"])

        boreholes_t = (
            torch.from_numpy(inputs["boreholes"]).unsqueeze(0).to(self.device)
        )  # (1, K, V, D)
        ore_vals_t = (
            torch.from_numpy(ore_vals_norm).unsqueeze(0).to(self.device)
        )  # (1, K)
        positions_t = (
            torch.from_numpy(inputs["positions"]).unsqueeze(0).to(self.device)
        )  # (1, K, 2)
        padding_mask_t = (
            torch.from_numpy(inputs["padding_mask"]).unsqueeze(0).to(self.device)
        )  # (1, K)

        with torch.no_grad():
            output = self.model(boreholes_t, ore_vals_t, positions_t, padding_mask_t)

        # Dispatch on output format
        if isinstance(output, tuple):
            if len(output) == 3:
                pred_ore_t, pred_unc_t, latent_t = output
                latent: np.ndarray | None = latent_t.squeeze(0).cpu().numpy()
            else:
                pred_ore_t, pred_unc_t = output
                latent = None
        else:
            pred_ore_t = output
            pred_unc_t = None
            latent = None

        # Denormalise ore map to ore-value space
        pred_ore_np: np.ndarray = (
            self.normalizer.inverse_tensor(pred_ore_t).squeeze().cpu().numpy()
        )  # (n_x, n_y)

        # Uncertainty stays in normalised space (no meaningful inverse)
        pred_unc_np: np.ndarray | None = None
        if pred_unc_t is not None:
            pred_unc_np = pred_unc_t.squeeze().cpu().numpy()  # (n_x, n_y)

        return BeliefState(
            predicted_ore_map=pred_ore_np,
            predicted_uncertainty_map=pred_unc_np,
            observed_mask=obs_state.get_observed_mask(),
            latent=latent,
            step=step,
        )
