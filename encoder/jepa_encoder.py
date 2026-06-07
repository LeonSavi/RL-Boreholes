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
    # Input-layer augmentation: append a normalised absolute-depth channel
    # (depth_idx / n_depth) as the last input row before the conv backbone.
    # When True, callers must feed (B, n_variables, n_depth) tensors where
    # the *last* row is the depth channel; n_variables therefore equals
    # len(wireline_channels) + 1.
    include_depth: bool = False


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
        self.n_total_tokens = max(1, cfg.n_depth // (2 ** len(cfg.channels)))
        # learnable mask token that stands in for target-position inputs
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # cache sinusoidal position embedding once — constant, deterministic
        # from cfg.  persistent=False keeps it out of state_dict (no need to
        # save derived data; lets old checkpoints load against new code too).
        pe = sinusoidal_position_embedding(self.n_total_tokens, d, torch.device("cpu"))
        self.register_buffer("pos_emb", pe, persistent=False)
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
    ) -> torch.Tensor:
        B, T_ctx, D = context_tokens.shape
        T_tgt = target_positions.size(1)
        pos_emb = self.pos_emb                                    # (T, D)

        # add positions (advanced-indexing fetch is one kernel)
        ctx_input = context_tokens + pos_emb[context_positions]   # (B, T_ctx, D)
        tgt_input = self.mask_token + pos_emb[target_positions]   # broadcasts to (B, T_tgt, D)

        # concatenate and run transformer
        seq = torch.cat([ctx_input, tgt_input], dim=1)            # (B, T_ctx+T_tgt, D)
        out = self.transformer(seq)
        return out[:, T_ctx:, :]                                  # (B, T_tgt, D)


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
        """Update target encoder via EMA of the context encoder.

        Fused via torch._foreach_lerp_ — one launched op covers all
        parameters at once, replacing the per-parameter Python loop.
        target = m*target + (1-m)*context  ==  lerp(target, context, 1-m)
        """
        target_params = list(self.target_encoder.parameters())
        context_params = list(self.context_encoder.parameters())
        torch._foreach_lerp_(target_params, context_params, 1.0 - self.cfg.ema_momentum)

    def forward(
        self,
        x_full: torch.Tensor,             # (B, V, D) full borehole
        context_positions: torch.Tensor,  # (B, T_ctx) long: token indices used as context
        target_positions: torch.Tensor,   # (B, T_tgt) long: token indices to predict
    ) -> torch.Tensor:
        """Compute JEPA training loss.

        The sampler returns explicit token indices (not boolean masks) so
        we never need torch.nonzero or .item() syncs at training time.
        For encoder input we still zero out the target depth range in the
        RAW borehole (so the encoder can't peek at it).
        """
        B, V, D = x_full.shape
        T = self.context_encoder.n_tokens
        D_lat = self.cfg.latent_dim

        # rebuild a (B, T) bool target-token mask from positions via scatter —
        # vectorized, no host-device sync.
        target_token_mask = torch.zeros(B, T, dtype=torch.bool, device=x_full.device)
        target_token_mask.scatter_(1, target_positions, True)

        # expand target token mask to depth resolution and zero the raw input
        # in the target region before handing it to the context encoder.
        scale = D // T
        depth_mask_target = target_token_mask.repeat_interleave(scale, dim=1)
        if depth_mask_target.size(1) != D:
            if depth_mask_target.size(1) < D:
                pad = D - depth_mask_target.size(1)
                depth_mask_target = torch.nn.functional.pad(
                    depth_mask_target, (0, pad), value=False)
            else:
                depth_mask_target = depth_mask_target[:, :D]
        x_context_input = x_full.masked_fill(depth_mask_target.unsqueeze(1), 0.0)

        # encode context (gradient) and target (stopgrad)
        ctx_tokens_full = self.context_encoder(x_context_input)  # (B, T, D_lat)
        with torch.no_grad():
            tgt_tokens_full = self.target_encoder(x_full)        # (B, T, D_lat)

        # vectorized gather of per-sample context / target tokens by position
        ctx_tokens = ctx_tokens_full.gather(
            1, context_positions.unsqueeze(-1).expand(-1, -1, D_lat))
        tgt_tokens = tgt_tokens_full.gather(
            1, target_positions.unsqueeze(-1).expand(-1, -1, D_lat))

        pred_tgt = self.predictor(
            context_tokens=ctx_tokens,
            context_positions=context_positions,
            target_positions=target_positions,
        )
        return nn.functional.smooth_l1_loss(pred_tgt, tgt_tokens, beta=1.0)

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
def sample_context_target_positions(
    n_tokens: int, batch_size: int, cfg: JEPAConfig,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized per-batch sampler.

    Strategy (per sample):
      1. Pick a contiguous target window of size round(target_window_frac*T),
         start drawn uniformly from [edge_buffer, T-edge_buffer-T_tgt].
      2. Context = random T_ctx positions drawn from the (T - T_tgt) tokens
         outside the target window.

    Returns (context_positions, target_positions) as long tensors of
    shape (B, T_ctx) and (B, T_tgt).  No Python per-sample loop, no
    torch.nonzero, no host-device sync at training time.
    """
    T = n_tokens
    T_tgt = max(2, int(round(cfg.target_window_frac * T)))
    T_ctx = max(2, int(round(cfg.context_keep_frac * T)))
    edge_buffer = max(1, int(cfg.min_target_dist_from_edge * T))

    lo_min = edge_buffer
    lo_max = T - edge_buffer - T_tgt
    if lo_max <= lo_min:
        starts = np.full(batch_size, max(0, (T - T_tgt) // 2), dtype=np.int64)
    else:
        starts = rng.integers(lo_min, lo_max + 1, size=batch_size).astype(np.int64)

    target_positions = starts[:, None] + np.arange(T_tgt, dtype=np.int64)[None, :]

    # Vectorized "without-replacement choice from the non-target set":
    # draw uniform keys, set keys at target positions to +inf, take the
    # T_ctx smallest per row (argpartition is O(T) per row).
    keys = rng.random((batch_size, T))
    rows = np.arange(batch_size)[:, None]
    keys[rows, target_positions] = np.inf
    context_positions = np.argpartition(keys, T_ctx - 1, axis=1)[:, :T_ctx].astype(np.int64)

    return (
        torch.from_numpy(context_positions),
        torch.from_numpy(target_positions),
    )


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