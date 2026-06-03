from __future__ import annotations

import numpy as np
import torch

from .models import UNetBelief
from .training_utils import TargetNormalizer


def predict_ore_map(
    observed_ore_map: np.ndarray,
    observation_mask: np.ndarray,
    jepa_latent_map: np.ndarray,
    model: UNetBelief,
    device: str,
    normalizer: TargetNormalizer | None = None,
) -> np.ndarray:
    """Predict the full spatial ore distribution from partial observations.

    Parameters
    ----------
    observed_ore_map : (1, n_x, n_y) or (n_x, n_y) float32
        Ore values at drilled locations; 0 at undrilled locations.
    observation_mask : (1, n_x, n_y) or (n_x, n_y) float32
        Binary mask: 1 = observed, 0 = unobserved.
    jepa_latent_map  : (latent_dim, n_x, n_y) or (n_x, n_y, latent_dim) float32
        JEPA latent vectors at drilled locations; zero vectors elsewhere.
    model            : trained UNetBelief
    device           : torch device string
    normalizer       : if provided, apply inverse transform to return predictions
                       in ore-value space; otherwise return raw model output

    Returns
    -------
    np.ndarray of shape (n_x, n_y) float32 — predicted ore map
    """
    ore  = np.asarray(observed_ore_map, dtype=np.float32)
    mask = np.asarray(observation_mask, dtype=np.float32)
    lat  = np.asarray(jepa_latent_map,  dtype=np.float32)

    if ore.ndim == 2:
        ore = ore[np.newaxis]
    if mask.ndim == 2:
        mask = mask[np.newaxis]
    # Accept (n_x, n_y, latent_dim) → convert to (latent_dim, n_x, n_y)
    if lat.ndim == 3 and lat.shape[2] not in (lat.shape[0], lat.shape[1]):
        lat = lat.transpose(2, 0, 1)

    inp = np.concatenate([ore, mask, lat], axis=0)  # (2 + latent_dim, n_x, n_y)
    x   = torch.from_numpy(inp).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()  # (n_x, n_y) in model output space

    if normalizer is not None:
        pred = normalizer.inverse(pred)

    return pred


def build_latent_map_from_observations(
    observations: list[dict],
    n_x: int,
    n_y: int,
    latent_dim: int = 128,
) -> np.ndarray:
    """Build (latent_dim, n_x, n_y) from a list of DrillObservation dicts.

    Observation dicts must contain:
      - ``location`` : (i, j) grid coordinates
      - ``latent``   : (latent_dim,) JEPA embedding

    Undrilled cells are zero vectors.

    Returns
    -------
    np.ndarray of shape (latent_dim, n_x, n_y) float32
    """
    jepa_map = np.zeros((n_x, n_y, latent_dim), dtype=np.float32)
    for obs in observations:
        i, j = obs["location"]
        jepa_map[i, j] = np.asarray(obs["latent"], dtype=np.float32)
    return jepa_map.transpose(2, 0, 1)  # (latent_dim, n_x, n_y)


def build_ore_map_from_observations(
    observations: list[dict],
    n_x: int,
    n_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse ore map and observation mask from DrillObservation dicts.

    Returns
    -------
    observed_ore_map : (1, n_x, n_y) float32
    observation_mask : (1, n_x, n_y) float32
    """
    sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
    mask       = np.zeros((n_x, n_y), dtype=np.float32)
    for obs in observations:
        i, j = obs["location"]
        sparse_ore[i, j] = float(obs["ore_value"])
        mask[i, j] = 1.0
    return sparse_ore[np.newaxis], mask[np.newaxis]


def predict_from_observations(
    observations: list[dict],
    model: UNetBelief,
    device: str,
    n_x: int = 32,
    n_y: int = 32,
    latent_dim: int = 128,
    normalizer: TargetNormalizer | None = None,
) -> np.ndarray:
    """Predict ore map directly from a list of DrillObservation dicts.

    Parameters
    ----------
    observations : list of DrillObservation dicts (from drill_at)
    model        : trained UNetBelief
    device       : torch device string
    n_x, n_y     : grid dimensions
    latent_dim   : JEPA latent dimension
    normalizer   : if provided, predictions are returned in ore-value space

    Returns
    -------
    np.ndarray of shape (n_x, n_y) float32 — predicted ore map
    """
    observed_ore_map, observation_mask = build_ore_map_from_observations(observations, n_x, n_y)
    jepa_latent_map = build_latent_map_from_observations(observations, n_x, n_y, latent_dim)
    return predict_ore_map(
        observed_ore_map, observation_mask, jepa_latent_map, model, device, normalizer
    )
