"""End-to-end candidate-scoring transformer for geological borehole maps.

Trains a borehole encoder jointly with a candidate-scoring transformer so that
latent embeddings are optimized for ore-discovery inference, not reconstruction.

Architecture
------------
BoreholeTransformerEncoder
    CNN backbone (4 Conv1d + MaxPool, same channel pattern as jepa_encoder.py)
    compresses depth from D=440 to D/16≈27 tokens.  A small transformer then
    captures long-range depth dependencies.  Mean-pool + linear projection
    yields a latent_dim embedding per borehole.

CandidateScoringTransformer
    Each observed borehole becomes a token:
        token = LayerNorm(bh_token_proj(embedding) + ore_proj(ore_val) + pos_emb(x, y))
    The candidate becomes a query token:
        candidate_token = candidate_proj(pos_emb(cx, cy))
    A transformer processes [observed_tokens..., candidate_token] and the
    candidate output is fed through a prediction head to yield a scalar ore value.

Token assembly
--------------
Observed borehole token at position (xi, yi) with ore value vi:
    bh_emb  = bh_token_proj(encoder(raw_borehole))   Linear(latent_dim → d_model)
    ore_emb = ore_proj(vi)                            Linear(1 → d_model)
    pos_emb = sinusoidal_2d_pe(xi, yi)                (d_model,)
    token_i = LayerNorm(bh_emb + ore_emb + pos_emb)

Candidate token at position (cx, cy):
    candidate_token = candidate_proj(sinusoidal_2d_pe(cx, cy))
                      Linear(d_model → d_model), init to identity

Inference helper
----------------
score_candidates(boreholes, ore_vals, positions, candidate_positions)
    Encodes boreholes once, then scores all C candidate positions in a single
    batched transformer forward pass.  Returns (C,) predicted ore scores.

Experiment variants
-------------------
A  No borehole encoder (position + ore only)   E2EConfig(latent_dim=0)
B  End-to-end encoder + candidate transformer  Default
C  Shuffled boreholes sanity check             Handled in training data generation
D  Init from pretrained JEPA weights           Handled in train_end_to_end.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class E2EConfig:
    """All hyperparameters for the end-to-end candidate scoring model.

    Follows the same pattern as MapBeliefConfig so that models can be
    reconstructed from checkpoints without the training config.
    """

    # Borehole dimensions
    n_variables: int = 5
    n_depth: int = 440

    # Borehole encoder: 1D CNN backbone channel sizes
    bh_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256))

    # Borehole encoder: transformer on top of CNN
    bh_d_model: int = 128          # transformer dim inside the borehole encoder
    bh_n_heads: int = 4            # attention heads (must divide bh_d_model)
    bh_n_layers: int = 2           # 0 = CNN only (no transformer)

    # Final borehole embedding dimension (output of borehole encoder)
    # Set to 0 to skip the encoder entirely (variant A: position + ore only)
    latent_dim: int = 128

    # Candidate scoring transformer
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    head_hidden_dim: int = 128     # hidden dim of the per-candidate MLP head

    # Positional encoding
    pe_max_freq: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.bh_n_layers > 0 and self.bh_d_model % self.bh_n_heads != 0:
            raise ValueError(
                f"bh_d_model={self.bh_d_model} must be divisible by "
                f"bh_n_heads={self.bh_n_heads}"
            )
        if self.d_model % 2 != 0:
            raise ValueError(
                "d_model must be even for 2D sinusoidal positional encoding "
                "(d_model/2 dims for x-axis, d_model/2 dims for y-axis)"
            )
        if self.latent_dim == 0 and self.bh_n_layers > 0:
            raise ValueError(
                "latent_dim=0 disables the borehole encoder; "
                "set bh_n_layers=0 to avoid confusion"
            )

    @property
    def use_encoder(self) -> bool:
        """True when borehole embeddings are included in observed tokens."""
        return self.latent_dim > 0

    @property
    def n_depth_tokens(self) -> int:
        """Number of depth tokens after the CNN backbone's max-pooling."""
        return max(1, self.n_depth // (2 ** len(self.bh_channels)))


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

def _sinusoidal_pe_1d(
    n_tokens: int, d_model: int, device: torch.device
) -> torch.Tensor:
    """Standard 1D sinusoidal positional encoding.

    Same formula as jepa_encoder.sinusoidal_position_embedding — used here
    for depth positions inside the borehole encoder.

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


def _sinusoidal_pe_2d(
    positions: torch.Tensor,
    d_model: int,
    pe_max_freq: float = 10000.0,
) -> torch.Tensor:
    """2D sinusoidal positional encoding for arbitrary (x, y) coordinates.

    Adapts the formula in map_belief_transformer._make_2d_sinusoidal_pe to
    work on arbitrary position sets rather than a regular grid.  The first
    d_model//2 dimensions encode x, the last d_model//2 encode y.

    Parameters
    ----------
    positions : (N, 2) float — normalised (x, y) coordinates in [0, 1]

    Returns
    -------
    pe : (N, d_model)
    """
    device = positions.device
    N = positions.shape[0]
    d_half = d_model // 2

    x = positions[:, 0:1]  # (N, 1)
    y = positions[:, 1:2]  # (N, 1)

    div = torch.exp(
        torch.arange(0, d_half, 2, device=device, dtype=torch.float32)
        * -(math.log(pe_max_freq) / d_half)
    )  # (d_half // 2,)

    pe_x = torch.zeros(N, d_half, device=device)
    pe_x[:, 0::2] = torch.sin(x * div)
    pe_x[:, 1::2] = torch.cos(x * div)

    pe_y = torch.zeros(N, d_half, device=device)
    pe_y[:, 0::2] = torch.sin(y * div)
    pe_y[:, 1::2] = torch.cos(y * div)

    return torch.cat([pe_x, pe_y], dim=1)  # (N, d_model)


# ---------------------------------------------------------------------------
# Borehole Encoder: CNN + Transformer hybrid
# ---------------------------------------------------------------------------

class BoreholeTransformerEncoder(nn.Module):
    """Borehole encoder trained end-to-end with the candidate scoring model.

    Architecture:
      1. 1D CNN backbone (same channel pattern as jepa_encoder.BoreholeConvBackbone)
         reduces depth from D to D/16 tokens.
      2. Small transformer over depth tokens captures long-range relationships.
      3. Mean-pool over tokens + linear projection → latent_dim embedding.

    Input:  (B, V, D)   V = n_variables, D = n_depth
    Output: (B, latent_dim)
    """

    def __init__(self, cfg: E2EConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # CNN backbone — identical channel progression to jepa_encoder.py
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

        cnn_out_ch = cfg.bh_channels[-1]
        self.seq_proj = nn.Linear(cnn_out_ch, cfg.bh_d_model)

        # Small transformer over compressed depth tokens
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
            self.transformer: nn.Module = nn.TransformerEncoder(enc_layer, cfg.bh_n_layers)
        else:
            self.transformer = nn.Identity()

        self.out_proj = nn.Linear(cfg.bh_d_model, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        feat = self.conv(x)                        # (B, C, T)  T ≈ D/16
        feat = feat.transpose(1, 2)                # (B, T, C)
        feat = self.seq_proj(feat)                 # (B, T, bh_d_model)

        T = feat.shape[1]
        pe = _sinusoidal_pe_1d(T, self.cfg.bh_d_model, feat.device)
        feat = feat + pe.unsqueeze(0)

        if self.cfg.bh_n_layers > 0:
            feat = self.transformer(feat)          # (B, T, bh_d_model)

        pooled = feat.mean(dim=1)                  # (B, bh_d_model)
        return self.out_proj(pooled)               # (B, latent_dim)


# ---------------------------------------------------------------------------
# Candidate Scoring Transformer
# ---------------------------------------------------------------------------

class CandidateScoringTransformer(nn.Module):
    """End-to-end candidate scoring model.

    The borehole encoder is trained jointly with the scoring transformer so
    that latent embeddings learn features directly useful for ore discovery,
    not reconstruction.

    The transformer reasons globally across all observed borehole tokens and
    one candidate token via full self-attention.  The candidate token output
    is passed through a prediction head to produce predicted ore at that
    location.

    forward() API (training / single-candidate)
    --------------------------------------------
    boreholes     : (B, K, V, D)  — padded standardised boreholes
    ore_vals      : (B, K)        — observed ore at drilled cells
    positions     : (B, K, 2)     — [0,1] normalised (x,y) of drilled cells
    candidate_pos : (B, 2)        — [0,1] normalised (cx,cy) candidate
    padding_mask  : (B, K) bool   — True at zero-padded positions
    Returns: (B, 1)

    score_candidates() API (inference / all candidates)
    ----------------------------------------------------
    boreholes           : (K, V, D)  — no batch dim, no padding needed
    ore_vals            : (K,)
    positions           : (K, 2)
    candidate_positions : (C, 2)     — all unvisited cells to score
    Returns: (C,)
    """

    def __init__(self, cfg: E2EConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Borehole encoder — only present when latent_dim > 0 (variant B/D)
        if cfg.use_encoder:
            self.bh_encoder: nn.Module | None = BoreholeTransformerEncoder(cfg)
            self.bh_token_proj: nn.Module | None = nn.Linear(cfg.latent_dim, cfg.d_model)
        else:
            self.bh_encoder = None
            self.bh_token_proj = None

        # Scalar ore value → d_model
        self.ore_proj = nn.Linear(1, cfg.d_model)

        # Layer norm applied after assembling observed tokens
        self.obs_token_norm = nn.LayerNorm(cfg.d_model)

        # Candidate role projection: differentiates the candidate token from
        # observed tokens without a separate learnable query embedding.
        # Initialised to identity so training starts from a neutral point.
        self.candidate_proj = nn.Linear(cfg.d_model, cfg.d_model)
        nn.init.eye_(self.candidate_proj.weight)
        nn.init.zeros_(self.candidate_proj.bias)

        # Candidate scoring transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability (same as MapBeliefTransformer)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, cfg.n_layers)

        # Per-candidate prediction head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].bias)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_boreholes(
        self,
        boreholes: torch.Tensor,
        padding_mask: torch.Tensor | None,
        batch_mode: bool,
        B: int,
        K: int,
    ) -> torch.Tensor:
        """Encode raw boreholes to latent vectors, skipping padded positions.

        Passing only non-padded boreholes to the CNN encoder avoids polluting
        BatchNorm statistics with zero-filled padding rows.

        Returns
        -------
        latents : (B, K, latent_dim) if batch_mode else (K, latent_dim)
        """
        assert self.bh_encoder is not None
        assert self.bh_token_proj is not None

        if batch_mode:
            if padding_mask is not None:
                not_padded = ~padding_mask                    # (B, K)
                bh_valid = boreholes[not_padded]              # (N_valid, V, D)
                lat_valid = self.bh_encoder(bh_valid)         # (N_valid, latent_dim)
                lat = torch.zeros(
                    B, K, self.cfg.latent_dim,
                    device=boreholes.device, dtype=lat_valid.dtype
                )
                lat[not_padded] = lat_valid
            else:
                bh_flat = boreholes.reshape(B * K, *boreholes.shape[2:])
                lat = self.bh_encoder(bh_flat).reshape(B, K, self.cfg.latent_dim)
        else:
            lat = self.bh_encoder(boreholes)                  # (K, latent_dim)

        return lat

    def _build_obs_tokens(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Assemble observed borehole tokens.

        token_i = LayerNorm(bh_token_proj(latent_i) + ore_proj(ore_i) + pos_emb(xi, yi))

        Works for both batched (B, K, ...) and unbatched (K, ...) inputs,
        determined by the number of dimensions in ``boreholes``.

        Returns
        -------
        tokens : (B, K, d_model)  or  (K, d_model)
        """
        batch_mode = boreholes.dim() == 4
        if batch_mode:
            B, K = boreholes.shape[:2]
        else:
            K = boreholes.shape[0]
            B = 0  # unused

        # Borehole embedding component
        if self.cfg.use_encoder:
            lat = self._encode_boreholes(boreholes, padding_mask, batch_mode, B, K)
            bh_tok = self.bh_token_proj(lat)                  # (..., K, d_model)
        else:
            shape = (B, K, self.cfg.d_model) if batch_mode else (K, self.cfg.d_model)
            bh_tok = torch.zeros(*shape, device=boreholes.device, dtype=boreholes.dtype)

        # Ore value component
        ore_tok = self.ore_proj(ore_vals.unsqueeze(-1))       # (..., K, d_model)

        # Positional embedding component
        if batch_mode:
            pos_flat = positions.reshape(B * K, 2)
        else:
            pos_flat = positions                              # (K, 2)
        pe_flat = _sinusoidal_pe_2d(pos_flat, self.cfg.d_model, self.cfg.pe_max_freq)
        if batch_mode:
            pos_emb = pe_flat.reshape(B, K, self.cfg.d_model)
        else:
            pos_emb = pe_flat                                 # (K, d_model)

        return self.obs_token_norm(bh_tok + ore_tok + pos_emb)  # (..., K, d_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        candidate_pos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ore at a single candidate location per batch element.

        Parameters
        ----------
        boreholes     : (B, K, V, D) — padded standardised raw boreholes
        ore_vals      : (B, K)       — observed ore values (0 at padding)
        positions     : (B, K, 2)    — [0,1] normalised (x,y) of observations
        candidate_pos : (B, 2)       — [0,1] normalised (cx,cy) candidate
        padding_mask  : (B, K) bool  — True at zero-padded positions

        Returns
        -------
        pred : (B, 1) — predicted ore in normalised space
        """
        B = boreholes.shape[0]
        K = boreholes.shape[1]

        obs_tokens = self._build_obs_tokens(boreholes, ore_vals, positions, padding_mask)
        # (B, K, d_model)

        # Candidate token: candidate_proj applied to 2D positional encoding
        cand_pe = _sinusoidal_pe_2d(
            candidate_pos, self.cfg.d_model, self.cfg.pe_max_freq
        )  # (B, d_model)
        cand_tok = self.candidate_proj(cand_pe).unsqueeze(1)  # (B, 1, d_model)

        # Full sequence: [observed tokens ..., candidate token]
        seq = torch.cat([obs_tokens, cand_tok], dim=1)        # (B, K+1, d_model)

        # Transformer key-padding mask (True = ignore)
        if padding_mask is not None:
            cand_no_pad = torch.zeros(B, 1, dtype=torch.bool, device=boreholes.device)
            full_mask = torch.cat([padding_mask, cand_no_pad], dim=1)  # (B, K+1)
        else:
            full_mask = None

        out = self.transformer(seq, src_key_padding_mask=full_mask)  # (B, K+1, d_model)
        cand_out = out[:, -1, :]                                     # (B, d_model)
        return self.head(cand_out)                                   # (B, 1)

    def score_candidates(
        self,
        boreholes: torch.Tensor,
        ore_vals: torch.Tensor,
        positions: torch.Tensor,
        candidate_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Score all candidate locations given the current set of observations.

        Encodes the K observed boreholes once, then scores all C candidate
        positions in a single batched transformer forward pass by repeating
        the observed token sequence for each candidate.

        Call this method with ``model.eval()`` and inside ``torch.no_grad()``.

        Parameters
        ----------
        boreholes           : (K, V, D)  — no batch dim, no padding
        ore_vals            : (K,)       — observed ore values
        positions           : (K, 2)     — [0,1] normalised positions
        candidate_positions : (C, 2)     — positions of all candidates to score

        Returns
        -------
        scores : (C,) — predicted ore values for each candidate
        """
        C = candidate_positions.shape[0]

        # Build observed tokens once — unbatched helper branch
        obs_tokens = self._build_obs_tokens(
            boreholes, ore_vals, positions, padding_mask=None
        )  # (K, d_model)

        # Broadcast to (C, K, d_model) for parallel candidate scoring
        obs_rep = obs_tokens.unsqueeze(0).expand(C, -1, -1).contiguous()

        # Candidate tokens: (C, 1, d_model)
        cand_pe = _sinusoidal_pe_2d(
            candidate_positions, self.cfg.d_model, self.cfg.pe_max_freq
        )
        cand_tok = self.candidate_proj(cand_pe).unsqueeze(1)  # (C, 1, d_model)

        # Full sequence: (C, K+1, d_model)
        seq = torch.cat([obs_rep, cand_tok], dim=1)

        out = self.transformer(seq)          # (C, K+1, d_model)
        cand_out = out[:, -1, :]            # (C, d_model)
        return self.head(cand_out).squeeze(-1)  # (C,)
