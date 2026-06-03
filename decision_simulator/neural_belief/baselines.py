from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from .training_utils import TargetNormalizer


def _mean_predict(sparse_ore: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Predict the mean of observed ore values uniformly across the grid.

    Falls back to 0.0 when no cells are observed.
    """
    n_observed = mask.sum()
    if n_observed == 0:
        return np.zeros_like(sparse_ore)
    observed_mean = float(sparse_ore[mask.astype(bool)].mean())
    return np.full_like(sparse_ore, observed_mean)


def _nearest_neighbor_predict(sparse_ore: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill each cell with the ore value of the nearest observed cell (Euclidean).

    Falls back to 0.0 when no cells are observed.
    """
    from scipy.ndimage import distance_transform_edt

    if mask.sum() == 0:
        return np.zeros_like(sparse_ore)

    # distance_transform_edt assigns each background cell (unobserved, False)
    # its distance to the nearest foreground cell (observed, True).
    # With return_indices=True it also returns those nearest cell coordinates.
    _, nearest = distance_transform_edt(~mask.astype(bool), return_indices=True)
    return sparse_ore[nearest[0], nearest[1]]


def _sample_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """MSE, MAE, and Pearson correlation for one (pred, truth) pair."""
    err = pred - truth
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    p_c = pred - pred.mean()
    t_c = truth - truth.mean()
    denom = float(np.linalg.norm(p_c) * np.linalg.norm(t_c))
    corr = float((p_c * t_c).sum() / denom) if denom > 1e-8 else 0.0

    return {"mse": mse, "mae": mae, "corr": corr}


def evaluate_baselines(
    val_loader: DataLoader,
    normalizer: TargetNormalizer,
) -> dict[str, dict[str, float]]:
    """Evaluate mean and nearest-neighbor baselines on the validation set.

    Inputs are never normalized, so baselines work directly in ore-value space.
    Targets come from the loader in normalized form and are inverted before comparison.

    Returns
    -------
    dict mapping baseline name to {"mse", "mae", "corr"} — all in ore-value space
    """
    accum: dict[str, dict[str, list[float]]] = {
        "mean_predictor":   {"mse": [], "mae": [], "corr": []},
        "nearest_neighbor": {"mse": [], "mae": [], "corr": []},
    }

    predictors = {
        "mean_predictor":   _mean_predict,
        "nearest_neighbor": _nearest_neighbor_predict,
    }

    for x_batch, y_batch in val_loader:
        sparse_ore_batch = x_batch[:, 0, :, :].numpy()                    # (B, n_x, n_y)
        mask_batch       = x_batch[:, 1, :, :].numpy()                    # (B, n_x, n_y)
        target_batch     = normalizer.inverse(y_batch.squeeze(1).numpy()) # (B, n_x, n_y)

        for b in range(x_batch.shape[0]):
            for name, fn in predictors.items():
                pred = fn(sparse_ore_batch[b], mask_batch[b])
                m = _sample_metrics(pred, target_batch[b])
                for k, v in m.items():
                    accum[name][k].append(v)

    return {
        name: {k: float(np.mean(vs)) for k, vs in vals.items()}
        for name, vals in accum.items()
    }
