from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos embedding")
    omega = torch.arange(embed_dim // 2, device=positions.device, dtype=positions.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = positions[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 2D sin/cos embedding")

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    grid_y = grid_y.reshape(-1).float()
    grid_x = grid_x.reshape(-1).float()

    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, grid_y)
    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, grid_x)
    emb = torch.cat([emb_y, emb_x], dim=1)
    return emb.unsqueeze(0)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)  # (B, E, H', W')
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x, (h, w)


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = x.shape

        tokens, (gh, gw) = self.patch_embed(x)
        pos = get_2d_sincos_pos_embed(self.embed_dim, gh, gw, tokens.device)
        tokens = tokens + pos

        tokens = self.encoder(tokens)
        patches = self.proj(tokens)  # (B, N, p*p*out_ch)

        patches = patches.view(b, gh, gw, self.out_channels, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = patches.view(b, self.out_channels, gh * p, gw * p)

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out
