from __future__ import annotations

from typing import Any

from .unet import UNet
from .vit import ViT


def _get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg: Any):
    name = _get(cfg, "name", "unet")
    if name == "unet":
        return UNet(
            in_channels=_get(cfg, "in_channels", 6),
            out_channels=_get(cfg, "out_channels", 3),
            base_channels=_get(cfg, "base_channels", 64),
            depth=_get(cfg, "depth", 4),
        )

    if name == "vit":
        vit_cfg = _get(cfg, "vit", {})
        return ViT(
            in_channels=_get(cfg, "in_channels", 6),
            out_channels=_get(cfg, "out_channels", 3),
            patch_size=_get(vit_cfg, "patch_size", 16),
            embed_dim=_get(vit_cfg, "embed_dim", 256),
            depth=_get(vit_cfg, "depth", 6),
            num_heads=_get(vit_cfg, "num_heads", 8),
            mlp_ratio=_get(vit_cfg, "mlp_ratio", 4.0),
            dropout=_get(vit_cfg, "dropout", 0.0),
        )

    raise ValueError(f"Unknown model name: {name}")
