"""CatVarEncoder — variable-aware patch borehole belief model with categorical labels.

Extends the uncertainty-head variant of the variable-aware patch borehole transformer
by also accepting per-depth rock-type categorical labels.

Why a soft one-hot projection instead of a raw numeric channel?
---------------------------------------------------------------
Rock types are *categorical* — their integer IDs carry no ordinal meaning (ID 3 is not
"more rock" than ID 2).  Inserting them as an extra numeric channel would impose a false
ordering and distort the encoder.  Instead, each vocab index is converted to a soft
one-hot distribution over a patch window, then projected via a learned linear layer to
the same bh_d_model space as the continuous tokens.

How rock patch tokens are computed
-----------------------------------
Each depth position carries one rock-type ID.  The ID is one-hot encoded, mean-pooled
over each depth-patch window, and projected to bh_d_model, giving one rock token per
patch.  That token is assembled alongside the V continuous variable tokens:

    token(v, p) = continuous_patch_proj(v, p)
                + variable_embedding(v)
                + depth_position_embedding(p)

    token(rock, p) = rock_proj(mean_pool_one_hot(rock_ids, p))
                   + rock_type_embedding   (token type V)
                   + depth_position_embedding(p)

How this model differs from VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer
---------------------------------------------------------------------------------------------
* Config type is CatVarEndToEndConfig (adds n_rock_types; inferred from labels_vocab.pkl).
* Borehole encoder is CatVarBoreholeTransformerEncoder (adds rock_proj).
* encode_categorical_boreholes() is used instead of encode_boreholes().
* All public methods take rock_ids in addition to boreholes.
* The map-transformer pipeline (SpatialProjectionLayer, MapBeliefEncoder,
  OreReconstructionHead, UncertaintyHead) is unchanged.

Architecture
------------
  CatVarBoreholeTransformerEncoder  — (B,K,V,D)+(B,K,D)+(B,K,D) → (B,K,latent_dim)
                                      seq_len = 1 + (n_variables+2) * n_patches
  SpatialProjectionLayer            — scatter latents onto grid tokens
  MapBeliefEncoder                  — transformer over grid tokens
  OreReconstructionHead             — ore map (B,1,n_x,n_y)
  UncertaintyHead                   — uncertainty map (B,1,n_x,n_y), softplus ≥ 0

Training objective (implemented in train_cat_var_encoder.py)
------------------------------------------------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + uncertainty_weight * uncertainty_loss

Public interface
----------------
forward(boreholes, rock_ids, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty)   both (B, 1, n_x, n_y)

encode(boreholes, rock_ids, ore_vals, positions, padding_mask=None)
    -> latent  (B, d_model)

forward_with_latent(boreholes, rock_ids, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty, latent)

encode_borehole_with_attention(borehole, rock_ids)
    -> (latent, attn_weights)

Input (forward)
---------------
boreholes    : (B, K, V, D)  padded standardised raw boreholes
rock_ids     : (B, K, D)     int64 rock-type vocab indices (0 = other/unknown)
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
from .utils.end_to_end_helpers import encode_categorical_boreholes
from ..borehole_encoder_components.components import CatVarBoreholeTransformerEncoder
from ..model_configs import CatVarEndToEndConfig


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CatVarEncoder(nn.Module):
    """End-to-end map belief transformer with categorical rock/formation label support.

    Extends the variable-aware uncertainty model by embedding per-depth rock-type and
    formation labels and injecting them into the borehole encoder's token stream.  The
    map-level pipeline (spatial projection, map transformer, ore head, uncertainty head)
    is identical to the base uncertainty variant.

    See module docstring for full architecture description.
    """

    def __init__(self, cfg: CatVarEndToEndConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.bh_encoder = CatVarBoreholeTransformerEncoder(cfg.to_encoder_config())

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
        rock_ids: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, uncertainty_map, map_belief_latent)."""
        latents = encode_categorical_boreholes(
            self.bh_encoder, boreholes, rock_ids,
            padding_mask, self.cfg.latent_dim,
        )
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
        rock_ids: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ore map and uncertainty from raw drilled boreholes + rock-type labels.

        Parameters
        ----------
        boreholes    : (B, K, V, D)
        rock_ids     : (B, K, D)  int64 rock-type vocab indices
        ore_vals     : (B, K)
        positions    : (B, K, 2)
        padding_mask : (B, K) bool

        Returns
        -------
        pred_ore         : (B, 1, n_x, n_y)
        pred_uncertainty : (B, 1, n_x, n_y)  — non-negative (softplus)
        """
        ore_map, uncertainty_map, _ = self._forward_all(
            boreholes, rock_ids, ore_vals, positions, padding_mask
        )
        return ore_map, uncertainty_map

    def encode(
        self,
        boreholes: torch.Tensor,
        rock_ids: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the map-level belief latent (CLS token output).

        Returns
        -------
        latent : (B, d_model)
        """
        _, _, latent = self._forward_all(
            boreholes, rock_ids, ore_vals, positions, padding_mask
        )
        return latent

    def forward_with_latent(
        self,
        boreholes: torch.Tensor,
        rock_ids: torch.Tensor,
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
        return self._forward_all(
            boreholes, rock_ids, ore_vals, positions, padding_mask
        )

    def encode_borehole_with_attention(
        self,
        borehole: torch.Tensor,
        rock_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a single borehole (or batch) with rock-type labels and return
        latent plus per-layer attention weights.

        Runs only the borehole encoder with attention capture enabled; the rest of
        the pipeline (scatter, map transformer, heads) is not executed.

        Parameters
        ----------
        borehole  : (V, D) or (B, V, D) — standardised raw borehole tensor
        rock_ids  : (D,)   or (B, D)    — int64 rock-type vocab indices

        Returns
        -------
        latent       : (B, latent_dim)
        attn_weights : (n_layers, B, n_heads, seq_len, seq_len) or None
                       seq_len = 1 + (n_variables + 1) * n_patches (CLS token first).
                       None when bh_n_layers == 0.

        Example
        -------
        >>> latent, attn_weights = model.encode_borehole_with_attention(borehole, rock_ids)
        >>> from decision_simulator.neural_belief.models.belief_models.borehole_encoder_components import (
        ...     extract_cls_attention_patch_major, plot_cls_attention,
        ... )
        >>> enc = model.bh_encoder
        >>> cls_attn = extract_cls_attention_patch_major(attn_weights, enc.cfg.n_variables, enc.n_patches)
        >>> plot_cls_attention(cls_attn, var_names=["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log", "rock"])
        """
        if borehole.dim() == 2:
            borehole = borehole.unsqueeze(0)
            rock_ids = rock_ids.unsqueeze(0)
        return self.bh_encoder(borehole, rock_ids, return_attention=True)


# ---------------------------------------------------------------------------
# Sanity / shape test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

    from decision_simulator.neural_belief.models.belief_models.model_configs import (
        CatVarEndToEndConfig,
    )

    cfg = CatVarEndToEndConfig(
        n_variables=5,
        n_depth=440,
        bh_patch_size=20,
        n_rock_types=12,
        n_x=32,
        n_y=32,
    )
    model = CatVarEncoder(cfg)
    model.eval()

    B, K, V, D = 2, 4, 5, 440
    boreholes = torch.randn(B, K, V, D)
    rock_ids  = torch.randint(0, 12, (B, K, D))
    ore_vals  = torch.rand(B, K)
    positions = torch.rand(B, K, 2)

    with torch.no_grad():
        pred_ore, pred_unc = model(boreholes, rock_ids, ore_vals, positions)

    assert pred_ore.shape == (B, 1, 32, 32), f"pred_ore shape: {pred_ore.shape}"
    assert pred_unc.shape == (B, 1, 32, 32), f"pred_unc shape: {pred_unc.shape}"

    with torch.no_grad():
        latent = model.encode(boreholes, rock_ids, ore_vals, positions)

    assert latent.shape == (B, cfg.d_model), f"latent shape: {latent.shape}"

    # Verify borehole encoder sequence length: 1 + (V+1) * n_patches
    enc = model.bh_encoder
    n_patches = enc.n_patches                          # 22 with patch_size=20, n_depth=440
    expected_seq = 1 + (5 + 1) * n_patches             # 133
    with torch.no_grad():
        _, attn = enc(boreholes[0], rock_ids[0], return_attention=True)
    assert attn[0].shape[-1] == expected_seq, (
        f"seq_len mismatch: got {attn[0].shape[-1]}, expected {expected_seq}"
    )

    print("CatVarEncoder shape test passed.")
    print(f"  pred_ore:  {tuple(pred_ore.shape)}")
    print(f"  pred_unc:  {tuple(pred_unc.shape)}")
    print(f"  latent:    {tuple(latent.shape)}")
    print(f"  bh seq_len: {attn[0].shape[-1]}  (1 + (5+1)×{n_patches})")
