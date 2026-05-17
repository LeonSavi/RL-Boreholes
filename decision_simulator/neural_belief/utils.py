from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TargetNormalizer:
    """Reversible normalization for ore-map targets.

    Supported modes
    ---------------
    "log1p"  : y' = log(1 + y)  — compresses the heavy right tail of ore values
    "zscore" : y' = (y - mean) / std  — zero-mean unit-variance
    "none"   : identity
    """

    mode: str = "log1p"
    mean: float = 0.0  # fitted; used by zscore only
    std: float = 1.0   # fitted; used by zscore only

    def fit(self, values: np.ndarray) -> "TargetNormalizer":
        """Fit statistics from any array of raw ore values."""
        if self.mode == "zscore":
            flat = values.ravel().astype(np.float64)
            self.mean = float(flat.mean())
            self.std = max(float(flat.std()), 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.mode == "log1p":
            return np.log1p(x)
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.mode == "log1p":
            return np.expm1(x)
        if self.mode == "zscore":
            return x * self.std + self.mean
        return x

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "log1p":
            return torch.log1p(x)
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        return x

    def inverse_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "log1p":
            return torch.expm1(x)
        if self.mode == "zscore":
            return x * self.std + self.mean
        return x


def build_ore_target(true_map: dict) -> np.ndarray:
    """Max-pool yield_field over depth.

    Returns
    -------
    np.ndarray of shape (n_x, n_y) float32
    """
    return true_map["yield_field"].max(axis=2).astype(np.float32)


def encode_full_latent_map(
    borehole_array: np.ndarray,
    n_x: int,
    n_y: int,
    resources,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode every borehole in a map in one batched pass.

    Parameters
    ----------
    borehole_array
        Pre-computed borehole tensor of shape ``(n_x * n_y, V, D)`` float32.
        Must already be standardised and nan-zeroed before calling.
    n_x, n_y
        Spatial grid dimensions.
    resources
        Shared experiment resources; encoder is resolved as:
        1. ``resources.borehole_encoder_fn`` if set
        2. ``resources.jepa_model.embed`` as fallback
        3. Neither set → returns empty ``(n_x, n_y, 0)`` array (no-encoder mode)
    device
        Torch device string.
    batch_size
        Boreholes encoded per forward pass.

    Returns
    -------
    np.ndarray of shape ``(n_x, n_y, latent_dim)`` float32
    """
    encoder_fn = getattr(resources, "borehole_encoder_fn", None)
    if encoder_fn is None:
        jepa = getattr(resources, "jepa_model", None)
        if jepa is not None:
            encoder_fn = jepa.embed

    if encoder_fn is None:
        return np.zeros((n_x, n_y, 0), dtype=np.float32)

    bh_tensor = torch.from_numpy(borehole_array.astype(np.float32)).to(device)

    chunks: list[np.ndarray] = []
    for start in range(0, bh_tensor.shape[0], batch_size):
        with torch.no_grad():
            lat = encoder_fn(bh_tensor[start : start + batch_size])
        chunks.append(lat.cpu().numpy())

    latents = np.concatenate(chunks, axis=0)  # (n_x*n_y, latent_dim)
    return latents.reshape(n_x, n_y, -1).astype(np.float32)


def build_sample_input(
    drill_locations: list[tuple[int, int]],
    ore_values: list[float],
    full_latent_map: np.ndarray,  # (n_x, n_y, latent_dim)
) -> np.ndarray:
    """Construct the (2 + latent_dim, n_x, n_y) model input from a drill pattern.

    Channel layout
    --------------
    [0]       : sparse observed ore map  (0 at unobserved cells)
    [1]       : binary observation mask  (1 = observed, 0 = unobserved)
    [2 ...]   : JEPA latent vectors      (zero vector at unobserved cells)
    """
    n_x, n_y, latent_dim = full_latent_map.shape

    sparse_ore = np.zeros((n_x, n_y), dtype=np.float32)
    mask = np.zeros((n_x, n_y), dtype=np.float32)
    jepa_map = np.zeros((n_x, n_y, latent_dim), dtype=np.float32)

    for (i, j), ore_val in zip(drill_locations, ore_values):
        sparse_ore[i, j] = float(ore_val)
        mask[i, j] = 1.0
        jepa_map[i, j] = full_latent_map[i, j]

    return np.concatenate(
        [
            sparse_ore[np.newaxis],        # (1, n_x, n_y)
            mask[np.newaxis],              # (1, n_x, n_y)
            jepa_map.transpose(2, 0, 1),   # (latent_dim, n_x, n_y)
        ],
        axis=0,
    )  # (2 + latent_dim, n_x, n_y)
