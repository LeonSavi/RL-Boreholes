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

from ..BH_to_map_encoder_components.spatial_projection_layer import SpatialProjectionLayer
from ..map_encoder_components.components import (
    MapBeliefEncoder,
    OreReconstructionHead,
    UncertaintyHead,
)
from .utils.end_to_end_helpers import encode_boreholes
from .variable_aware_patch_borehole_transformer import (
    VariableAwarePatchBoreholeEndToEndConfig,
    VariableAwarePatchBoreholeTransformerEncoder,
)


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
        self.spatial_projection = SpatialProjectionLayer(map_cfg)
        self.map_encoder = MapBeliefEncoder(map_cfg)
        self.ore_head = OreReconstructionHead(map_cfg)
        self.uncertainty_head = UncertaintyHead(map_cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_all(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, uncertainty_map, map_belief_latent)."""
        latents = encode_boreholes(self.bh_encoder, boreholes, padding_mask, self.cfg.latent_dim)
        tokens = self.spatial_projection(ore_vals, positions, latents, padding_mask)
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

    def encode_borehole_with_attention(
        self,
        borehole: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a single borehole (or batch) and return its latent plus attention weights.

        Runs only the borehole encoder with attention capture enabled; the rest of
        the pipeline (scatter, map transformer, heads) is not executed.

        Parameters
        ----------
        borehole : (V, D) or (B, V, D) — standardised raw borehole tensor

        Returns
        -------
        latent       : (B, latent_dim)
        attn_weights : (n_layers, B, n_heads, seq_len, seq_len) or None
                       seq_len = 1 + n_variables * n_patches (CLS token first).
                       None when bh_n_layers == 0.

        Example
        -------
        >>> latent, attn_weights = model.encode_borehole_with_attention(borehole)
        >>> from decision_simulator.neural_belief.models.belief_models.borehole_encoder_components import (
        ...     extract_cls_attention, plot_cls_attention,
        ... )
        >>> enc = model.bh_encoder
        >>> cls_attn = extract_cls_attention(attn_weights, enc.cfg.n_variables, enc.n_patches)
        >>> plot_cls_attention(cls_attn, var_names=["Fe", "Al", ...])
        """
        if borehole.dim() == 2:
            borehole = borehole.unsqueeze(0)
        return self.bh_encoder(borehole, return_attention=True)
