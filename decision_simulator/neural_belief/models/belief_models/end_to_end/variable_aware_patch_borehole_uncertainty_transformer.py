"""Variable-aware patch borehole belief model with spatial uncertainty head.

Experimental variant of VariableAwarePatchBoreholeEndToEndMapBeliefTransformer
that predicts both a reconstructed ore map and a per-cell spatial uncertainty map.

Architecture
------------
Identical to the base variable-aware variant except an UncertaintyHead runs in
parallel to OreReconstructionHead, sharing the same spatial transformer outputs.
Uncertainty outputs are passed through softplus, guaranteeing strictly non-negative
predictions.

Shared components (unchanged from base model)
---------------------------------------------
  VariableAwarePatchBoreholeTransformerEncoder
  SpatialTokenEmbedding
  MapBeliefEncoder
  OreReconstructionHead

New component
-------------
  UncertaintyHead  — same MLP structure as OreReconstructionHead + softplus output

Training objective (implemented in the companion training script)
-----------------------------------------------------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + uncertainty_weight * uncertainty_loss

Public interface
----------------
forward(boreholes, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty)   both (B, 1, n_x, n_y)

encode(boreholes, ore_vals, positions, padding_mask=None)
    -> latent  (B, d_model)

forward_with_latent(boreholes, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty, latent)

Input (forward)
---------------
boreholes    : (B, K, V, D)  padded standardised raw boreholes
ore_vals     : (B, K)        observed ore values  (0 at padding)
positions    : (B, K, 2)     normalised [0,1] (x, y) of drilled cells
padding_mask : (B, K) bool   True at zero-padded rows

Output
------
pred_ore         : (B, 1, n_x, n_y)  reconstructed ore map (normalised space)
pred_uncertainty : (B, 1, n_x, n_y)  spatial uncertainty   (>= 0, normalised space)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..map_encoders.map_belief_transformer import (
    MapBeliefConfig,
    MapBeliefEncoder,
    OreReconstructionHead,
    SpatialTokenEmbedding,
)
from .variable_aware_patch_borehole_transformer import (
    VariableAwarePatchBoreholeEndToEndConfig,
    VariableAwarePatchBoreholeTransformerEncoder,
)


# ---------------------------------------------------------------------------
# Uncertainty head
# ---------------------------------------------------------------------------


class UncertaintyHead(nn.Module):
    """Per-cell MLP head predicting spatial prediction uncertainty.

    Structurally identical to OreReconstructionHead, with softplus applied to
    the final output to enforce strictly non-negative uncertainty estimates.

    Input
    -----
    spatial_tokens  : (B, n_x * n_y, d_model)
    n_x, n_y        : int — passed at runtime to support variable grid sizes

    Output
    ------
    uncertainty_map : (B, 1, n_x, n_y)  — strictly non-negative (softplus)
    """

    def __init__(self, cfg: MapBeliefConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.trunc_normal_(self.mlp[-1].weight, std=0.02)

    def forward(
        self, spatial_tokens: torch.Tensor, n_x: int, n_y: int
    ) -> torch.Tensor:
        B = spatial_tokens.shape[0]
        unc_flat = self.mlp(spatial_tokens)           # (B, N, 1)
        unc_flat = F.softplus(unc_flat.squeeze(-1))   # (B, N) — strictly positive
        return unc_flat.reshape(B, 1, n_x, n_y)      # (B, 1, n_x, n_y)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer(nn.Module):
    """End-to-end map belief transformer with variable-aware encoding and uncertainty head.

    Extends VariableAwarePatchBoreholeEndToEndMapBeliefTransformer by adding a
    parallel UncertaintyHead that estimates per-cell prediction uncertainty from
    the same spatial transformer representations used by the ore reconstruction head.

    The model config type is identical to the base variant
    (VariableAwarePatchBoreholeEndToEndConfig): no new architecture dimensions are
    needed because the uncertainty head reuses head_hidden_dim and dropout.

    For downstream tasks use encode() to obtain the map belief latent (CLS token),
    or forward_with_latent() to get all outputs in one pass.
    """

    def __init__(self, cfg: VariableAwarePatchBoreholeEndToEndConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.bh_encoder = VariableAwarePatchBoreholeTransformerEncoder(
            cfg.to_encoder_config()
        )

        map_cfg = cfg.to_map_belief_config()
        self.token_embed = SpatialTokenEmbedding(map_cfg)
        self.map_encoder = MapBeliefEncoder(map_cfg)
        self.ore_head = OreReconstructionHead(map_cfg)
        self.uncertainty_head = UncertaintyHead(map_cfg)

    # ------------------------------------------------------------------
    # Internal helpers (identical to base model)
    # ------------------------------------------------------------------

    def _encode_boreholes(
        self,
        boreholes: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode raw boreholes to latent vectors, skipping padded rows."""
        B, K = boreholes.shape[:2]
        if padding_mask is not None:
            not_padded = ~padding_mask
            bh_valid = boreholes[not_padded]
            lat_valid = self.bh_encoder(bh_valid)
            lat = torch.zeros(
                B, K, self.cfg.latent_dim,
                device=boreholes.device, dtype=lat_valid.dtype,
            )
            lat[not_padded] = lat_valid
        else:
            bh_flat = boreholes.reshape(B * K, *boreholes.shape[2:])
            lat = self.bh_encoder(bh_flat).reshape(B, K, self.cfg.latent_dim)
        return lat

    def _scatter_to_map(
        self,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        latents: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a dense (B, 2 + latent_dim, n_x, n_y) map from sparse observations."""
        B, K = ore_vals.shape
        n_x, n_y = self.cfg.n_x, self.cfg.n_y
        N = n_x * n_y
        C = 2 + self.cfg.latent_dim
        device = ore_vals.device
        dtype = ore_vals.dtype

        i_idx = (positions[..., 0] * (n_x - 1)).round().long().clamp(0, n_x - 1)
        j_idx = (positions[..., 1] * (n_y - 1)).round().long().clamp(0, n_y - 1)
        flat_idx = i_idx * n_y + j_idx

        if padding_mask is not None:
            flat_idx = flat_idx.masked_fill(padding_mask, N)
            valid = (~padding_mask).to(dtype)
        else:
            valid = torch.ones(B, K, device=device, dtype=dtype)

        ore_ch = (ore_vals * valid).unsqueeze(1)
        mask_ch = valid.unsqueeze(1)
        lat_ch = (latents * valid.unsqueeze(-1)).transpose(1, 2)
        values = torch.cat([ore_ch, mask_ch, lat_ch], dim=1)

        flat_idx_exp = flat_idx.unsqueeze(1).expand(B, C, K)
        out_flat = torch.zeros(B, C, N + 1, device=device, dtype=dtype)
        out_flat.scatter_(2, flat_idx_exp, values)

        return out_flat[:, :, :N].reshape(B, C, n_x, n_y)

    def _forward_all(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, uncertainty_map, map_belief_latent)."""
        latents = self._encode_boreholes(boreholes, padding_mask)
        x = self._scatter_to_map(ore_vals, positions, latents, padding_mask)
        tokens = self.token_embed(x)
        cls_out, spatial_out = self.map_encoder(tokens)
        ore_map = self.ore_head(spatial_out, self.cfg.n_x, self.cfg.n_y)
        uncertainty_map = self.uncertainty_head(spatial_out, self.cfg.n_x, self.cfg.n_y)
        return ore_map, uncertainty_map, cls_out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ore map and uncertainty from raw drilled boreholes.

        Parameters
        ----------
        boreholes    : (B, K, V, D)
        ore_vals     : (B, K)
        positions    : (B, K, 2)
        padding_mask : (B, K) bool

        Returns
        -------
        pred_ore         : (B, 1, n_x, n_y)
        pred_uncertainty : (B, 1, n_x, n_y)  — non-negative (softplus)
        """
        ore_map, uncertainty_map, _ = self._forward_all(
            boreholes, ore_vals, positions, padding_mask
        )
        return ore_map, uncertainty_map

    def encode(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the map-level belief latent (CLS token output).

        Returns
        -------
        latent : (B, d_model)
        """
        _, _, latent = self._forward_all(boreholes, ore_vals, positions, padding_mask)
        return latent

    def forward_with_latent(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ore reconstruction, uncertainty map, and map belief latent.

        Returns
        -------
        pred_ore         : (B, 1, n_x, n_y)
        pred_uncertainty : (B, 1, n_x, n_y)
        latent           : (B, d_model)
        """
        return self._forward_all(boreholes, ore_vals, positions, padding_mask)
