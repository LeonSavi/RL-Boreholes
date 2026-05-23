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
    std: float = 1.0  # fitted; used by zscore only

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


def make_coordinate_grid(n_x: int, n_y: int) -> np.ndarray:
    """Return normalized (x, y) coordinate channels of shape ``(2, n_x, n_y)`` float32.

    x varies along axis 0 (rows), y along axis 1 (columns), both in ``[0, 1]``.
    """
    x = np.linspace(0.0, 1.0, n_x, dtype=np.float32)
    y = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
    x_grid = np.broadcast_to(x[:, None], (n_x, n_y)).copy()
    y_grid = np.broadcast_to(y[None, :], (n_x, n_y)).copy()
    return np.stack([x_grid, y_grid], axis=0)  # (2, n_x, n_y)


