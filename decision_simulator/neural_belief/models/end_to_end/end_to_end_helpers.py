"""Shared helpers for end-to-end borehole transformer experiments."""

from __future__ import annotations

import math

import torch


def _sinusoidal_pe_1d(
    n_tokens: int, d_model: int, device: torch.device
) -> torch.Tensor:
    """Standard 1D sinusoidal positional encoding.

    Returns
    -------
    pe : (n_tokens, d_model)
    """
    pos = torch.arange(n_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(n_tokens, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
