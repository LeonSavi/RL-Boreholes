"""
JEPA encoder — alternative to the reconstruction autoencoder.

Overview
========
Joint Embedding Predictive Architecture (Assran et al., 2023, I-JEPA).
Instead of reconstructing raw variable values from a latent, JEPA learns
to predict the EMBEDDING of a masked region from the embedding of an
unmasked context region.  The key idea: force the encoder to represent
content abstractly enough that one depth range can predict another's
embedding, without needing to reconstruct pixel-level noise.

Architecture
------------
  context_x ─→ context_encoder ──→ z_context ──┐
                                                ├─→ predictor ──→ ẑ_target
  pos_target ─────────────────────────────────→ │
                                                │
  target_x  ─→ target_encoder   ──→ z_target  ──┘  (stopgrad, EMA weights)

  Loss: SmoothL1(ẑ_target, z_target)

The target encoder's weights are an exponential moving average (EMA)
of the context encoder's.  No gradients flow through it.  This prevents
the trivial collapse solution (both encoders output zero → zero loss).

Design choices for borehole data
--------------------------------
* Masking: depth-window.  Context = non-contiguous 60% of depth range,
  target = contiguous 25% window at a random depth.  This simulates
  "we've logged parts of the column, predict what the log signature
  looks like at an unseen interval."
* Position encoding: sinusoidal on depth index, fed to the predictor so
  it knows WHERE in the borehole it's predicting.  Crucial — without
  it the predictor can't tell "predict Zechstein" from "predict Rotliegend."
* Encoder architecture: SAME 1D CNN backbone as the reconstruction
  encoder, for fair comparison.  Only the training signal differs.

Interface
---------
  JEPAEncoder(cfg)(x) -> latent [B, D_latent]    (inference mode)

The latent has the same shape as the reconstruction encoder's output, so
`validate_latents.py` can be adapted with minimal changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class JEPAConfig:
    n_variables: int
    n_depth: int
    latent_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 256)
    # predictor (transformer) params
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_dim_ff: int = 512
    # masking
    context_keep_frac: float = 0.6       # fraction of depth kept as context
    target_window_frac: float = 0.25     # size of the target window
    min_target_dist_from_edge: float = 0.05
    # EMA momentum for target encoder
    ema_momentum: float = 0.996


class BoreholeConvBackbone(nn.Module):
    """Shared 1D CNN backbone — identical to the reconstruction encoder's
    feature extractor. Used by both context and target encoders.

    Input:  (B, V, D) after masking (masked cells zeroed OR left unchanged
                                     by caller — this module is agnostic)
    Output: (B, channels[-1], D / 2^n_pools) — FEATURE MAP, not pooled.

    We return the full feature map (not pooled) so the predictor can
    attend to specific depth positions when asked to predict a target
    window.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_ch = cfg.n_variables
        for out_ch in cfg.channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        return self.conv(x)  # (B, C_last, D/2^n_pools)


class BoreholeTokenEncoder(nn.Module):
    """Wraps the backbone and projects each spatial (depth) position to
    the latent dimension. Output is a SEQUENCE of tokens, one per
    down-sampled depth position.

    This is different from the reconstruction encoder, which pools to a
    single 128-dim vector. Here we keep the sequence structure because
    the predictor needs per-position embeddings to attend over.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = BoreholeConvBackbone(cfg)
        self.proj = nn.Linear(cfg.channels[-1], cfg.latent_dim)
        # post-pool length
        self.n_tokens = max(1, cfg.n_depth // (2 ** len(cfg.channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D) -> feat (B, C, D/16) -> tokens (B, D/16, latent_dim)
        feat = self.backbone(x)
        feat = feat.transpose(1, 2)  # (B, T, C)
        return self.proj(feat)       # (B, T, latent_dim)


def sinusoidal_position_embedding(
    n_tokens: int, dim: int, device: torch.device,
) -> torch.Tensor:
    """Standard sinusoidal position embedding, (T, D)."""
    position = torch.arange(n_tokens, device=device).float().unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float()
        * -(np.log(10000.0) / dim)
    )
    pe = torch.zeros(n_tokens, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LatentPredictor(nn.Module):
    """Small transformer that predicts target-position latents from
    context-position latents.

    Input:  context_tokens (B, T_ctx, D_latent)
            target_positions (B, T_tgt)  — integer indices into the full
                                            sequence of T_tokens positions
    Output: predicted target tokens (B, T_tgt, D_latent)

    We use a trick from I-JEPA: instead of reassembling a full masked
    sequence, concatenate [context tokens, learnable MASK tokens at target
    positions], add position embeddings, run a transformer, then extract
    the output at target positions.
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        # learnable mask token that stands in for target-position inputs
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # transformer encoder layers
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.predictor_heads,
            dim_feedforward=cfg.predictor_dim_ff,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.predictor_depth)

    def forward(
        self,
        context_tokens: torch.Tensor,      # (B, T_ctx, D)
        context_positions: torch.Tensor,   # (B, T_ctx) indices into full seq
        target_positions: torch.Tensor,    # (B, T_tgt) indices into full seq
        n_total_tokens: int,
    ) -> torch.Tensor:
        B, T_ctx, D = context_tokens.shape
        T_tgt = target_positions.size(1)

        device = context_tokens.device
        pos_emb = sinusoidal_position_embedding(n_total_tokens, D, device)  # (T, D)

        # add positions to context tokens
        ctx_pe = pos_emb[context_positions]                      # (B, T_ctx, D)
        ctx_input = context_tokens + ctx_pe

        # build target-position mask tokens and add positions
        mask = self.mask_token.expand(B, T_tgt, D)               # (B, T_tgt, D)
        tgt_pe = pos_emb[target_positions]                       # (B, T_tgt, D)
        tgt_input = mask + tgt_pe

        # concatenate and run transformer
        seq = torch.cat([ctx_input, tgt_input], dim=1)           # (B, T_ctx+T_tgt, D)
        out = self.transformer(seq)                              # (B, T_ctx+T_tgt, D)

        # return only the target-position outputs
        return out[:, T_ctx:, :]                                 # (B, T_tgt, D)


class JEPAModel(nn.Module):
    """Full JEPA training module: context encoder + target encoder (EMA)
    + predictor.

    Usage patterns:
        At training time: model(x, context_mask, target_mask) -> loss
        At inference:     model.embed(x) -> latent (B, D_latent)
                          [pooled across tokens for a single per-borehole vector]
    """

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = BoreholeTokenEncoder(cfg)
        self.target_encoder = BoreholeTokenEncoder(cfg)
        # target encoder starts identical to context, then EMA-updates
        for tgt_p, ctx_p in zip(self.target_encoder.parameters(),
                                 self.context_encoder.parameters()):
            tgt_p.data.copy_(ctx_p.data)
            tgt_p.requires_grad_(False)
        self.predictor = LatentPredictor(cfg)

    @torch.no_grad()
    def ema_update(self) -> None:
        """Update target encoder via EMA of the context encoder."""
        m = self.cfg.ema_momentum
        for tgt_p, ctx_p in zip(self.target_encoder.parameters(),
                                 self.context_encoder.parameters()):
            tgt_p.data.mul_(m).add_(ctx_p.data, alpha=1.0 - m)

    def forward(
        self,
        x_full: torch.Tensor,             # (B, V, D) full borehole
        context_token_mask: torch.Tensor, # (B, T) bool: True where token is CONTEXT
        target_token_mask: torch.Tensor,  # (B, T) bool: True where token is TARGET
    ) -> torch.Tensor:
        """Compute JEPA training loss.

        Both masks operate on the DOWN-SAMPLED token sequence (length T =
        n_depth / 2^n_pools). The masks define which tokens are used as
        context input and which are the prediction targets.

        For encoder input we zero out the target region in the RAW
        borehole (so the encoder can't peek at it), then encode.
        """
        B, V, D = x_full.shape
        T = self.context_encoder.n_tokens
        assert context_token_mask.shape == (B, T)
        assert target_token_mask.shape == (B, T)

        # --- mask the raw input before giving it to the context encoder ---
        # expand target_token_mask from (B, T) back to (B, D) so we can
        # zero out the corresponding raw depth positions.
        scale = D // T
        depth_mask_target = target_token_mask.repeat_interleave(scale, dim=1)
        # ensure shape matches (pad or clip if D/T isn't exact)
        if depth_mask_target.size(1) != D:
            if depth_mask_target.size(1) < D:
                pad = D - depth_mask_target.size(1)
                depth_mask_target = torch.nn.functional.pad(
                    depth_mask_target, (0, pad), value=False)
            else:
                depth_mask_target = depth_mask_target[:, :D]

        x_context_input = x_full.clone()
        x_context_input[depth_mask_target.unsqueeze(1).expand_as(x_full)] = 0.0

        # --- encode context and targets ---
        # context encoder sees masked input, produces token sequence
        ctx_tokens_full = self.context_encoder(x_context_input)  # (B, T, D_lat)

        # target encoder sees UNMASKED input; we stopgrad
        with torch.no_grad():
            tgt_tokens_full = self.target_encoder(x_full)        # (B, T, D_lat)

        # --- gather per-sample context and target token subsets ---
        # We assume the same number of context / target tokens per sample
        # (enforced by the sampler). Use indexing via mask.
        # Result tensors are (B, T_ctx, D_lat) and (B, T_tgt, D_lat).
        T_ctx = context_token_mask.sum(dim=1).min().item()
        T_tgt = target_token_mask.sum(dim=1).min().item()

        context_positions = torch.zeros(B, T_ctx, dtype=torch.long,
                                         device=x_full.device)
        target_positions = torch.zeros(B, T_tgt, dtype=torch.long,
                                        device=x_full.device)
        ctx_tokens = torch.zeros(B, T_ctx, self.cfg.latent_dim,
                                  device=x_full.device)
        tgt_tokens = torch.zeros(B, T_tgt, self.cfg.latent_dim,
                                  device=x_full.device)

        for b in range(B):
            ctx_idx = torch.nonzero(context_token_mask[b], as_tuple=False).squeeze(1)[:T_ctx]
            tgt_idx = torch.nonzero(target_token_mask[b], as_tuple=False).squeeze(1)[:T_tgt]
            context_positions[b] = ctx_idx
            target_positions[b]  = tgt_idx
            ctx_tokens[b] = ctx_tokens_full[b, ctx_idx]
            tgt_tokens[b] = tgt_tokens_full[b, tgt_idx]

        # --- predict target tokens from context ---
        pred_tgt = self.predictor(
            context_tokens=ctx_tokens,
            context_positions=context_positions,
            target_positions=target_positions,
            n_total_tokens=T,
        )

        # --- loss: SmoothL1 on the predicted vs true target tokens ---
        loss = nn.functional.smooth_l1_loss(pred_tgt, tgt_tokens, beta=1.0)
        return loss

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: pool target-encoder tokens to a single latent per
        borehole. We use the target encoder here because the EMA version
        is what downstream consumers (POMDP) should use (stable features).

        x: (B, V, D)
        returns: (B, D_latent)
        """
        self.target_encoder.eval()
        tokens = self.target_encoder(x)         # (B, T, D_latent)
        return tokens.mean(dim=1)               # average pool over tokens


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------
def sample_context_target_masks(
    n_tokens: int, batch_size: int, cfg: JEPAConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-batch context / target token masks.

    Strategy (per sample, independently):
      1. Pick a contiguous target window of size round(target_window_frac * T).
      2. Target positions = those indices.
      3. Context positions = a random subset of non-target positions
         of size round(context_keep_frac * T).
      4. Guarantee target doesn't touch the very top/bottom edges.

    Returns (context_mask, target_mask): (B, T) bool tensors.
    """
    T = n_tokens
    T_tgt = max(2, int(round(cfg.target_window_frac * T)))
    T_ctx = max(2, int(round(cfg.context_keep_frac * T)))
    edge_buffer = max(1, int(cfg.min_target_dist_from_edge * T))

    context_mask = np.zeros((batch_size, T), dtype=bool)
    target_mask = np.zeros((batch_size, T), dtype=bool)
    for b in range(batch_size):
        # pick start index for the contiguous target window
        lo_min = edge_buffer
        lo_max = T - edge_buffer - T_tgt
        if lo_max <= lo_min:
            lo = max(0, (T - T_tgt) // 2)
        else:
            lo = int(rng.integers(lo_min, lo_max + 1))
        target_mask[b, lo:lo + T_tgt] = True
        # context = random subset of non-target
        candidates = np.where(~target_mask[b])[0]
        if len(candidates) < T_ctx:
            # shouldn't happen given our sizes but be safe
            T_ctx_actual = len(candidates)
        else:
            T_ctx_actual = T_ctx
        chosen = rng.choice(candidates, size=T_ctx_actual, replace=False)
        context_mask[b, chosen] = True

    return (torch.from_numpy(context_mask),
            torch.from_numpy(target_mask))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_jepa_checkpoint(
    model: JEPAModel,
    stats: dict,
    variables: list[str],
    path: str | Path,
) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": model.cfg.__dict__,
        "stats": stats,
        "variables": variables,
    }, path)


def load_jepa_checkpoint(
    path: str | Path, device: str = "cpu",
) -> tuple[JEPAModel, dict, list[str]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    # reconstruct JEPAConfig (dict → dataclass)
    cfg = JEPAConfig(**ck["cfg"])
    model = JEPAModel(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["stats"], ck["variables"]