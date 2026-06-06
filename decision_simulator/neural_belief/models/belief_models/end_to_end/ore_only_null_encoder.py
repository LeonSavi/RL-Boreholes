"""OreOnlyNullEncoder — null-test model for the end-to-end belief map architecture.

Null-test hypothesis
--------------------
This model tests whether the borehole encoder in CatVarEncoder adds useful
predictive information beyond what is already observable from:

  * the ore yield at each drilled location  (ore_vals)
  * the binary drill mask                   (which cells have been sampled)
  * the spatial positions of drilled cells  (positions)

It does NOT use:
  * continuous borehole depth logs          (boreholes tensor)
  * rock-type labels                        (rock_ids tensor)
  * formation labels                        (formation_ids tensor)

If CatVarEncoder does not outperform OreOnlyNullEncoder, the borehole encoder is
probably not contributing useful predictive information and map reconstruction is
driven primarily by the observed ore values and their spatial pattern.

Architecture
------------
OreOnlyNullEncoder is structurally identical to CatVarEncoder's map-level pipeline:

  SpatialProjectionLayer  — scatter ore + positions + ZERO latents onto grid tokens
  MapBeliefEncoder        — transformer over grid tokens
  OreReconstructionHead   — ore map (B, 1, n_x, n_y)
  UncertaintyHead         — uncertainty map (B, 1, n_x, n_y), softplus >= 0

The only difference from CatVarEncoder is that the borehole latent vectors are
replaced with zeros of the same shape:

    zero_latents = torch.zeros(B, K, cfg.latent_dim)

This keeps raw_token_dim and the map encoder architecture *identical* to
CatVarEncoder, making the comparison fair — the only experimental variable is
whether the borehole encoder latents carry useful information.

Training objective (implemented in train_ore_only_null_encoder.py)
------------------------------------------------------------------
  ore_loss         = MSE(pred_ore, target)
  uncertainty_loss = MSE(pred_uncertainty, |pred_ore.detach() - target|)
  total_loss       = ore_loss + uncertainty_weight * uncertainty_loss

Public interface
----------------
forward(boreholes, rock_ids, formation_ids, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty)   both (B, 1, n_x, n_y)

encode(boreholes, rock_ids, formation_ids, ore_vals, positions, padding_mask=None)
    -> latent  (B, d_model)

forward_with_latent(boreholes, rock_ids, formation_ids, ore_vals, positions, padding_mask=None)
    -> (pred_ore, pred_uncertainty, latent)

Input (forward)
---------------
boreholes     : (B, K, V, D)  — accepted for dataloader compatibility; ignored
rock_ids      : (B, K, D)     — accepted for dataloader compatibility; ignored
formation_ids : (B, K, D)     — accepted for dataloader compatibility; ignored
ore_vals      : (B, K)        observed ore values (0 at padding)
positions     : (B, K, 2)     normalised [0,1] (x, y) of drilled cells
padding_mask  : (B, K) bool   True at zero-padded rows

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
from ..model_configs import CatVarEndToEndConfig


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class OreOnlyNullEncoder(nn.Module):
    """Null-test belief model: uses only observed ore values and drill positions.

    Structurally identical to CatVarEncoder's map-level pipeline, but replaces
    all borehole latents with zeros.  Accepts the same inputs as CatVarEncoder
    (boreholes, rock_ids, formation_ids) for dataloader compatibility, but
    silently ignores them.

    See module docstring for full null-test rationale.
    """

    def __init__(self, cfg: CatVarEndToEndConfig) -> None:
        super().__init__()
        self.cfg = cfg

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
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pipeline returning (ore_map, uncertainty_map, map_belief_latent)."""
        B, K = ore_vals.shape
        zero_latents = torch.zeros(
            B, K, self.cfg.latent_dim,
            device=ore_vals.device,
            dtype=ore_vals.dtype,
        )
        tokens = self.spatial_projection(ore_vals, positions, zero_latents, padding_mask)
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
        formation_ids: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ore map and uncertainty from drill positions and observed ore only.

        boreholes, rock_ids, and formation_ids are accepted for dataloader
        compatibility but are not used.

        Parameters
        ----------
        boreholes     : (B, K, V, D)  — ignored
        rock_ids      : (B, K, D)     — ignored
        formation_ids : (B, K, D)     — ignored
        ore_vals      : (B, K)
        positions     : (B, K, 2)
        padding_mask  : (B, K) bool

        Returns
        -------
        pred_ore         : (B, 1, n_x, n_y)
        pred_uncertainty : (B, 1, n_x, n_y)  — non-negative (softplus)
        """
        ore_map, uncertainty_map, _ = self._forward_all(ore_vals, positions, padding_mask)
        return ore_map, uncertainty_map

    def encode(
        self,
        boreholes: torch.Tensor,
        rock_ids: torch.Tensor,
        formation_ids: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the map-level belief latent (CLS token output).

        Returns
        -------
        latent : (B, d_model)
        """
        _, _, latent = self._forward_all(ore_vals, positions, padding_mask)
        return latent

    def forward_with_latent(
        self,
        boreholes: torch.Tensor,
        rock_ids: torch.Tensor,
        formation_ids: torch.Tensor,
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
        return self._forward_all(ore_vals, positions, padding_mask)


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
        n_rock_types=12,  # unused by this model; kept so cfg matches CatVarEncoder
        n_x=32,
        n_y=32,
    )
    model = OreOnlyNullEncoder(cfg)
    model.eval()

    B, K, V, D = 2, 4, 5, 440
    boreholes     = torch.randn(B, K, V, D)
    rock_ids      = torch.randint(0, 12, (B, K, D))
    formation_ids = torch.randint(0, 8,  (B, K, D))
    ore_vals      = torch.rand(B, K)
    positions     = torch.rand(B, K, 2)

    with torch.no_grad():
        pred_ore, pred_unc = model(boreholes, rock_ids, formation_ids, ore_vals, positions)

    assert pred_ore.shape == (B, 1, 32, 32), f"pred_ore shape: {pred_ore.shape}"
    assert pred_unc.shape == (B, 1, 32, 32), f"pred_unc shape: {pred_unc.shape}"

    with torch.no_grad():
        latent = model.encode(boreholes, rock_ids, formation_ids, ore_vals, positions)

    assert latent.shape == (B, cfg.d_model), f"latent shape: {latent.shape}"

    print("OreOnlyNullEncoder shape test passed.")
    print(f"  pred_ore : {tuple(pred_ore.shape)}")
    print(f"  pred_unc : {tuple(pred_unc.shape)}")
    print(f"  latent   : {tuple(latent.shape)}")
