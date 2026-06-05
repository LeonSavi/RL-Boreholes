"""Shared helpers for end-to-end borehole transformer experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def encode_boreholes(
    bh_encoder: nn.Module,
    boreholes: torch.Tensor,
    padding_mask: torch.Tensor | None,
    latent_dim: int,
) -> torch.Tensor:
    """Encode raw boreholes to latent vectors, skipping padded rows.

    Parameters
    ----------
    bh_encoder   : callable borehole encoder module
    boreholes    : (B, K, V, D)
    padding_mask : (B, K) bool — True at padded positions; None if no padding
    latent_dim   : output embedding size

    Returns
    -------
    latents : (B, K, latent_dim)
    """
    B, K = boreholes.shape[:2]
    if padding_mask is not None:
        not_padded = ~padding_mask
        bh_valid = boreholes[not_padded]
        lat_valid = bh_encoder(bh_valid)
        lat = torch.zeros(B, K, latent_dim, device=boreholes.device, dtype=lat_valid.dtype)
        lat[not_padded] = lat_valid
    else:
        bh_flat = boreholes.reshape(B * K, *boreholes.shape[2:])
        lat = bh_encoder(bh_flat).reshape(B, K, latent_dim)
    return lat


def scatter_to_map(
    ore_vals: torch.Tensor,
    positions: torch.Tensor,
    latents: torch.Tensor,
    padding_mask: torch.Tensor | None,
    n_x: int,
    n_y: int,
    latent_dim: int,
) -> torch.Tensor:
    """Build a dense (B, 2 + latent_dim, n_x, n_y) map from sparse observations.

    Drilled-cell positions (normalised [0,1]) are converted to flat grid indices.
    Padded positions are routed to a scratch slot at index N so they never
    overwrite real observations.  The scratch slot is discarded before the final
    reshape.

    Parameters
    ----------
    ore_vals     : (B, K) — observed ore at drilled cells
    positions    : (B, K, 2) — normalised [0,1] (x, y)
    latents      : (B, K, latent_dim) — borehole embeddings
    padding_mask : (B, K) bool — True at padded positions
    n_x, n_y     : grid dimensions
    latent_dim   : embedding size

    Returns
    -------
    x : (B, 2 + latent_dim, n_x, n_y)
        ch 0  : sparse observed ore value
        ch 1  : binary drilled mask (1 = observed)
        ch 2+ : borehole latent embedding
    """
    B, K = ore_vals.shape
    N = n_x * n_y
    C = 2 + latent_dim
    device = ore_vals.device
    dtype = ore_vals.dtype

    i_idx = (positions[..., 0] * (n_x - 1)).round().long().clamp(0, n_x - 1)
    j_idx = (positions[..., 1] * (n_y - 1)).round().long().clamp(0, n_y - 1)
    flat_idx = i_idx * n_y + j_idx  # (B, K)

    if padding_mask is not None:
        flat_idx = flat_idx.masked_fill(padding_mask, N)
        valid = (~padding_mask).to(dtype)
    else:
        valid = torch.ones(B, K, device=device, dtype=dtype)

    ore_ch = (ore_vals * valid).unsqueeze(1)                        # (B, 1, K)
    mask_ch = valid.unsqueeze(1)                                     # (B, 1, K)
    lat_ch = (latents * valid.unsqueeze(-1)).transpose(1, 2)        # (B, latent_dim, K)
    values = torch.cat([ore_ch, mask_ch, lat_ch], dim=1)            # (B, C, K)

    flat_idx_exp = flat_idx.unsqueeze(1).expand(B, C, K)
    out_flat = torch.zeros(B, C, N + 1, device=device, dtype=dtype)
    out_flat.scatter_(2, flat_idx_exp, values)

    return out_flat[:, :, :N].reshape(B, C, n_x, n_y)


def encode_categorical_boreholes(
    bh_encoder: nn.Module,
    boreholes: torch.Tensor,
    rock_ids: torch.Tensor,
    formation_ids: torch.Tensor,
    padding_mask: torch.Tensor | None,
    latent_dim: int,
) -> torch.Tensor:
    """Encode boreholes with categorical rock/formation context, skipping padded rows.

    Companion to encode_boreholes() for use with CatVarBoreholeTransformerEncoder,
    which requires three inputs per borehole: the continuous variable tensor plus
    per-depth rock-type and formation integer IDs.

    Parameters
    ----------
    bh_encoder    : CatVarBoreholeTransformerEncoder (or compatible)
    boreholes     : (B, K, V, D)
    rock_ids      : (B, K, D) int64 rock-type vocab indices
    formation_ids : (B, K, D) int64 formation vocab indices
    padding_mask  : (B, K) bool — True at padded positions; None if no padding
    latent_dim    : output embedding size

    Returns
    -------
    latents : (B, K, latent_dim)
    """
    B, K = boreholes.shape[:2]
    if padding_mask is not None:
        not_padded = ~padding_mask                        # (B, K) bool
        bh_valid   = boreholes[not_padded]               # (N_valid, V, D)
        rock_valid = rock_ids[not_padded]                # (N_valid, D)
        form_valid = formation_ids[not_padded]           # (N_valid, D)
        lat_valid  = bh_encoder(bh_valid, rock_valid, form_valid)
        lat = torch.zeros(B, K, latent_dim, device=boreholes.device, dtype=lat_valid.dtype)
        lat[not_padded] = lat_valid
    else:
        bh_flat   = boreholes.reshape(B * K, *boreholes.shape[2:])
        rock_flat = rock_ids.reshape(B * K, rock_ids.shape[2])
        form_flat = formation_ids.reshape(B * K, formation_ids.shape[2])
        lat = bh_encoder(bh_flat, rock_flat, form_flat).reshape(B, K, latent_dim)
    return lat


def sinusoidal_pe_1d(n_tokens: int, d_model: int, device: torch.device) -> torch.Tensor:
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
