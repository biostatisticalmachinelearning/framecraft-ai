from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
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


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.downs.append(DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = DoubleConv(ch, ch * 2)
        ch = ch * 2

        self.up_transpose = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = base_channels * (2**i)
            self.up_transpose.append(nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(out_ch * 2, out_ch))
            ch = out_ch

        self.final = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up_t, up_conv, skip in zip(self.up_transpose, self.up_convs, reversed(skips)):
            x = up_t(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)

        return self.final(x)
