"""Borehole encoder neural network modules.

Four encoder architectures, each mapping a single raw borehole (B, V, D) to a
fixed-size latent vector (B, latent_dim):

  BoreholeTransformerEncoder               — 1D CNN backbone + small transformer, mean-pool
  PatchBoreholeTransformerEncoder          — depth patch tokeniser, mean-pool
  PatchBoreholeCLSTransformerEncoder       — depth patch tokeniser, learned CLS-token pooling
  VariableAwarePatchBoreholeTransformerEncoder — per-(variable, patch) token, CLS-token pooling
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..end_to_end.utils.end_to_end_helpers import sinusoidal_pe_1d
from ..end_to_end.utils.model_configs import (
    E2EConfig,
    PatchBoreholeConfig,
    PatchBoreholeCLSConfig,
    VariableAwarePatchBoreholeConfig,
)


# ---------------------------------------------------------------------------
# CNN + transformer encoder
# ---------------------------------------------------------------------------


class BoreholeTransformerEncoder(nn.Module):
    """1D CNN + transformer borehole encoder.

    Architecture:
      1. 1D CNN backbone reduces depth from D to D/16 tokens.
      2. Small transformer over depth tokens captures long-range relationships.
      3. Mean-pool + linear projection → latent_dim embedding.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: E2EConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_ch = cfg.n_variables
        for out_ch in cfg.bh_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.seq_proj = nn.Linear(cfg.bh_channels[-1], cfg.bh_d_model)

        if cfg.bh_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.bh_d_model,
                nhead=cfg.bh_n_heads,
                dim_feedforward=cfg.bh_d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, cfg.bh_n_layers
            )
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        feat = self.conv(x)              # (B, C, T)
        feat = feat.transpose(1, 2)      # (B, T, C)
        feat = self.seq_proj(feat)       # (B, T, bh_d_model)

        T = feat.shape[1]
        pe = sinusoidal_pe_1d(T, self.cfg.bh_d_model, feat.device)
        feat = feat + pe.unsqueeze(0)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)

        pooled = feat.mean(dim=1)        # (B, bh_d_model)
        return self.out_proj(pooled)     # (B, latent_dim)


# ---------------------------------------------------------------------------
# Patch encoder — mean-pool
# ---------------------------------------------------------------------------


class PatchBoreholeTransformerEncoder(nn.Module):
    """Patch-based borehole encoder with mean-pool aggregation.

    Architecture:
      1. Split depth into non-overlapping patches of size bh_patch_size.
         Depth is zero-padded to the nearest multiple of bh_patch_size if needed.
      2. Each patch flattens variables × depth_interval: token_dim = n_variables * bh_patch_size.
      3. Linear projection to bh_d_model.
      4. 1D sinusoidal PE added per patch position.
      5. Small transformer over patch tokens.
      6. Mean-pool + linear projection → latent_dim embedding.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: PatchBoreholeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        self.padded_depth = self.n_patches * cfg.bh_patch_size

        token_dim = cfg.n_variables * cfg.bh_patch_size
        self.patch_proj = nn.Linear(token_dim, cfg.bh_d_model)

        if cfg.bh_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.bh_d_model,
                nhead=cfg.bh_n_heads,
                dim_feedforward=cfg.bh_d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, cfg.bh_n_layers
            )
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"PatchBoreholeTransformerEncoder: "
            f"V={cfg.n_variables}, D={cfg.n_depth}, "
            f"patch_size={cfg.bh_patch_size}, n_patches={self.n_patches}"
        )
        print(
            f"  bh_d_model={cfg.bh_d_model}, n_layers={cfg.bh_n_layers}, "
            f"n_heads={cfg.bh_n_heads}, latent_dim={cfg.latent_dim}, "
            f"params={n_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        B, V, D = x.shape
        pad_len = self.padded_depth - D
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))  # (B, V, padded_depth)

        x = x.view(B, V, self.n_patches, self.cfg.bh_patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()                    # (B, n_patches, V, patch_size)
        x = x.reshape(B, self.n_patches, V * self.cfg.bh_patch_size)

        feat = self.patch_proj(x)                                  # (B, n_patches, bh_d_model)

        pe = sinusoidal_pe_1d(self.n_patches, self.cfg.bh_d_model, feat.device)
        feat = feat + pe.unsqueeze(0)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)

        pooled = feat.mean(dim=1)                                  # (B, bh_d_model)
        return self.out_proj(pooled)                               # (B, latent_dim)


# ---------------------------------------------------------------------------
# Patch encoder — CLS-token pooling
# ---------------------------------------------------------------------------


class PatchBoreholeCLSTransformerEncoder(nn.Module):
    """Patch-based borehole encoder with learned CLS-token pooling.

    Identical to PatchBoreholeTransformerEncoder except:
      - A learned CLS token is prepended before the transformer.
      - CLS output (index 0) replaces mean pooling.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: PatchBoreholeCLSConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        self.padded_depth = self.n_patches * cfg.bh_patch_size

        token_dim = cfg.n_variables * cfg.bh_patch_size
        self.patch_proj = nn.Linear(token_dim, cfg.bh_d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.bh_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if cfg.bh_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.bh_d_model,
                nhead=cfg.bh_n_heads,
                dim_feedforward=cfg.bh_d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, cfg.bh_n_layers
            )
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"PatchBoreholeCLSTransformerEncoder: "
            f"V={cfg.n_variables}, D={cfg.n_depth}, "
            f"patch_size={cfg.bh_patch_size}, n_patches={self.n_patches}, cls_pooling=True"
        )
        print(
            f"  bh_d_model={cfg.bh_d_model}, n_layers={cfg.bh_n_layers}, "
            f"n_heads={cfg.bh_n_heads}, latent_dim={cfg.latent_dim}, "
            f"params={n_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        B, V, D = x.shape
        pad_len = self.padded_depth - D
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        x = x.view(B, V, self.n_patches, self.cfg.bh_patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(B, self.n_patches, V * self.cfg.bh_patch_size)

        feat = self.patch_proj(x)

        # Positional encoding applied to patch tokens only — CLS gets none
        pe = sinusoidal_pe_1d(self.n_patches, self.cfg.bh_d_model, feat.device)
        feat = feat + pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)  # (B, 1 + n_patches, bh_d_model)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)

        cls_out = feat[:, 0]             # (B, bh_d_model)
        return self.out_proj(cls_out)    # (B, latent_dim)


# ---------------------------------------------------------------------------
# Variable-aware patch encoder — per-(variable, patch) token, CLS pooling
# ---------------------------------------------------------------------------


class VariableAwarePatchBoreholeTransformerEncoder(nn.Module):
    """Variable-aware patch borehole encoder with CLS-token pooling.

    Architecture:
      1. Split depth into non-overlapping patches.
      2. Each (variable, patch) pair becomes one token of size bh_patch_size —
         giving n_variables * n_patches tokens per borehole.
      3. Linear projection: bh_patch_size → bh_d_model.
      4. Learned variable embedding added (distinguishes variable identity).
      5. 1D sinusoidal PE added (encodes depth patch position).
      6. Learned CLS token prepended.
      7. Small transformer over all tokens.
      8. CLS output + linear projection → latent_dim embedding.

    Input:  (B, V, D)
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: VariableAwarePatchBoreholeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_patches = math.ceil(cfg.n_depth / cfg.bh_patch_size)
        self.padded_depth = self.n_patches * cfg.bh_patch_size

        self.patch_proj = nn.Linear(cfg.bh_patch_size, cfg.bh_d_model)

        self.var_embed = nn.Embedding(cfg.n_variables, cfg.bh_d_model)
        nn.init.trunc_normal_(self.var_embed.weight, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.bh_d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if cfg.bh_n_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.bh_d_model,
                nhead=cfg.bh_n_heads,
                dim_feedforward=cfg.bh_d_model * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                enc_layer, cfg.bh_n_layers
            )
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

        n_bh_tokens = cfg.n_variables * self.n_patches
        n_params = sum(p.numel() for p in self.parameters())
        print("VariableAwarePatchBoreholeTransformerEncoder: variable-aware patching enabled")
        print(
            f"  n_variables={cfg.n_variables}, patch_size={cfg.bh_patch_size}, "
            f"n_patches={self.n_patches}"
        )
        print(
            f"  total borehole tokens = {cfg.n_variables} × {self.n_patches} + 1 CLS"
            f" = {n_bh_tokens + 1}"
        )
        print(
            f"  bh_d_model={cfg.bh_d_model}, n_layers={cfg.bh_n_layers}, "
            f"n_heads={cfg.bh_n_heads}, latent_dim={cfg.latent_dim}, "
            f"params={n_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a batch of boreholes to latent vectors.

        Parameters
        ----------
        x                : (B, V, D)
        return_attention : if True, also return per-layer attention weights

        Returns
        -------
        latent       : (B, latent_dim)
        attn_weights : (n_layers, B, n_heads, seq_len, seq_len) — only when
                       return_attention=True; None when bh_n_layers == 0.
                       seq_len = 1 + n_variables * n_patches (CLS first).
        """
        # x: (B, V, D)
        B, V, D = x.shape
        pad_len = self.padded_depth - D
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        x = x.view(B, V, self.n_patches, self.cfg.bh_patch_size)
        x = x.reshape(B, V * self.n_patches, self.cfg.bh_patch_size)

        feat = self.patch_proj(x)  # (B, V * n_patches, bh_d_model)

        # Variable embedding: token v*n_patches+p belongs to variable v
        var_idx = torch.arange(V, device=x.device).repeat_interleave(self.n_patches)
        var_emb = self.var_embed(var_idx)  # (V * n_patches, bh_d_model)
        feat = feat + var_emb.unsqueeze(0)

        # Depth positional encoding: token v*n_patches+p belongs to patch p
        pe_full = sinusoidal_pe_1d(self.n_patches, self.cfg.bh_d_model, feat.device)
        patch_idx = torch.arange(self.n_patches, device=feat.device).repeat(V)
        depth_pe = pe_full[patch_idx]  # (V * n_patches, bh_d_model)
        feat = feat + depth_pe.unsqueeze(0)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)  # (B, 1 + V*n_patches, bh_d_model)

        attn_weights: torch.Tensor | None = None
        if self.cfg.bh_n_layers > 0:
            if return_attention:
                # Manually iterate layers to capture per-layer attention weights.
                # Replicates nn.TransformerEncoderLayer forward with norm_first=True
                # while calling self_attn with need_weights=True.
                all_attn: list[torch.Tensor] = []
                for layer in self.transformer.layers:
                    normed = layer.norm1(feat)
                    attn_out, attn_w = layer.self_attn(
                        normed, normed, normed,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    feat = feat + layer.dropout1(attn_out)
                    feat = feat + layer._ff_block(layer.norm2(feat))
                    all_attn.append(attn_w)  # (B, n_heads, seq, seq)
                attn_weights = torch.stack(all_attn, dim=0)  # (n_layers, B, n_heads, seq, seq)
            else:
                feat = self.transformer(feat)

        cls_out = feat[:, 0]             # (B, bh_d_model)
        latent = self.out_proj(cls_out)  # (B, latent_dim)

        if return_attention:
            return latent, attn_weights
        return latent
