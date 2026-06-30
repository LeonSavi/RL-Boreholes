"""
Autoencoder encoder — stage 1c (the ML side).

Takes a single synthetic borehole (a (n_depth, n_variables) array) and
produces a latent embedding of fixed size.  Trained via reconstruction
loss on simulator-generated boreholes.

Architecture: 1D CNN over depth axis.
  Input:  (batch, n_variables, n_depth)
  Encoder: 1D conv stack + global average pool -> latent_dim
  Decoder: latent -> transpose-conv stack -> (batch, n_variables, n_depth)

Training hook is exposed in the `train()` function.  Checkpoint saves the
encoder weights, the config, and the variable ordering so inference in
Stage 2 (the POMDP) uses exactly the same assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AEConfig:
    n_variables: int
    n_depth: int
    latent_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 256)
    # for partial-observation training: mask some input variables at random
    # so the encoder learns to produce consistent embeddings even when only
    # a subset is observed.  This is the bridge to Task 2's POMDP setting.
    mask_prob: float = 0.3
    # when True the input carries a 6th normalised-depth channel (matching the
    # JEPA encoder), so n_variables already includes it.  Eval code appends the
    # depth row by reading this flag off the saved config.
    include_depth: bool = False


class BoreholeEncoder(nn.Module):
    """1D CNN encoder: (B, V, D) -> (B, latent_dim)."""

    def __init__(self, cfg: AEConfig):
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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(cfg.channels[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, D)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class BoreholeDecoder(nn.Module):
    """Mirror decoder: (B, latent_dim) -> (B, V, D)."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        # compute the spatial size after the encoder's 4 MaxPool1d halvings
        n_pools = len(cfg.channels)
        self.bottleneck_len = max(1, cfg.n_depth // (2 ** n_pools))
        self.expand = nn.Linear(
            cfg.latent_dim,
            cfg.channels[-1] * self.bottleneck_len,
        )
        layers: list[nn.Module] = []
        in_ch = cfg.channels[-1]
        reversed_ch = list(cfg.channels[::-1][1:]) + [cfg.n_variables]
        for i, out_ch in enumerate(reversed_ch):
            is_last = (i == len(reversed_ch) - 1)
            layers += [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            ]
            if not is_last:
                layers += [nn.BatchNorm1d(out_ch), nn.GELU()]
            in_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        h = self.expand(z).view(
            z.size(0), self.cfg.channels[-1], self.bottleneck_len,
        )
        recon = self.deconv(h)
        # trim / pad to exactly n_depth (the successive upsamples may
        # overshoot slightly when n_depth isn't a clean power of 2)
        if recon.size(-1) != self.cfg.n_depth:
            recon = nn.functional.interpolate(
                recon, size=self.cfg.n_depth, mode="linear", align_corners=False
            )
        return recon


class BoreholeAutoencoder(nn.Module):
    """Full autoencoder.  If cfg.mask_prob > 0, randomly zeros out whole
    variable channels during training so the encoder learns to infer the
    full-variable representation from partial inputs."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = BoreholeEncoder(cfg)
        self.decoder = BoreholeDecoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        if self.training and self.cfg.mask_prob > 0:
            x_masked = self._random_mask(x)
        else:
            x_masked = x
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, z

    def _random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out variable channels (per-sample) with
        probability cfg.mask_prob, mimicking partial observations."""
        B, V, D = x.shape
        keep = (torch.rand(B, V, 1, device=x.device) > self.cfg.mask_prob).float()
        # guarantee at least one channel survives
        ensure = torch.zeros(B, V, 1, device=x.device)
        ensure.scatter_(
            1,
            torch.randint(0, V, (B, 1, 1), device=x.device),
            1.0,
        )
        mask = torch.maximum(keep, ensure)
        return x * mask


def standardise(x: np.ndarray, stats: dict[str, tuple[float, float]],
                variables: list[str], clip: float = 4.0) -> np.ndarray:
    """Per-variable z-score using training stats, then winsorise to ±clip σ.

    x has shape (..., V, D).  stats[var] = (mean, std).

    Clipping is important for training stability: some variables
    (res_deep_log, pef, sp_mv) have heavy tails in the real data.
    Without clipping, an occasional batch hits a ±5σ value, produces
    a huge MSE, and the gradient (even if clipped to norm=1.0) still
    points in a direction that destroys weights for other variables.
    ±4σ preserves 99.99% of legitimate data and clips the pathological
    tail.
    """
    out = x.copy()
    for i, v in enumerate(variables):
        m, s = stats.get(v, (0.0, 1.0))
        if s < 1e-8:
            s = 1.0
        z = (out[..., i, :] - m) / s
        out[..., i, :] = np.clip(z, -clip, clip)
    return out


def unstandardise(x: np.ndarray, stats, variables: list[str]) -> np.ndarray:
    out = x.copy()
    for i, v in enumerate(variables):
        m, s = stats.get(v, (0.0, 1.0))
        out[..., i, :] = out[..., i, :] * s + m
    return out


def save_checkpoint(
    model: BoreholeAutoencoder,
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


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[
        BoreholeAutoencoder, dict, list[str]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = AEConfig(**ck["cfg"])
    model = BoreholeAutoencoder(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["stats"], ck["variables"]