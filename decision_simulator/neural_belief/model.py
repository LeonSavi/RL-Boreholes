from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    """Two (Conv2d → BN → ReLU) layers with the same output channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBelief(nn.Module):
    """Lightweight 2-level U-Net for geological belief prediction.

    Maps a sparse observation tensor to a dense predicted ore map.

    Input  : (B, in_channels, n_x, n_y)   default in_channels = 130
    Output : (B, 1,           n_x, n_y)

    Architecture (default base_channels=64)
    ----------------------------------------
    Encoder
      enc1 : DoubleConv(in_ch → 64)   @ 32*32
      enc2 : DoubleConv(64   → 128)   @ 16*16  (after MaxPool)
    Bottleneck
      bot  : DoubleConv(128  → 256)   @  8*8   (after MaxPool)
    Decoder
      dec1 : DoubleConv(384  → 128)   @ 16*16  (bilinear up + skip from enc2)
      dec2 : DoubleConv(192  → 64)    @ 32*32  (bilinear up + skip from enc1)
    Head
      conv : Conv2d(64 → 1, k=1)      @ 32*32
    """

    def __init__(self, in_channels: int = 130, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = _DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(c * 2, c * 4)

        self.dec1 = _DoubleConv(c * 4 + c * 2, c * 2)
        self.dec2 = _DoubleConv(c * 2 + c, c)

        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def _up(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s1 = self.enc1(x)                   # (B, c,   H,   W)
        s2 = self.enc2(self.pool1(s1))       # (B, 2c, H/2, W/2)
        b  = self.bottleneck(self.pool2(s2)) # (B, 4c, H/4, W/4)

        # Decoder
        d1 = self.dec1(self._up(b,  s2))    # (B, 2c, H/2, W/2)
        d2 = self.dec2(self._up(d1, s1))    # (B, c,   H,   W)

        return self.head(d2)                 # (B, 1,   H,   W)
