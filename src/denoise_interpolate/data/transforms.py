from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def random_crop_params(h: int, w: int, crop_h: int, crop_w: int) -> Tuple[int, int]:
    if h == crop_h:
        top = 0
    else:
        top = random.randint(0, h - crop_h)
    if w == crop_w:
        left = 0
    else:
        left = random.randint(0, w - crop_w)
    return top, left


def apply_crop(img: torch.Tensor, top: int, left: int, crop_h: int, crop_w: int) -> torch.Tensor:
    return img[:, top : top + crop_h, left : left + crop_w]


def center_crop(img: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    _, h, w = img.shape
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return apply_crop(img, top, left, crop_h, crop_w)


def ensure_min_size(img: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    _, h, w = img.shape
    if h >= min_h and w >= min_w:
        return img
    scale = max(min_h / h, min_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return img.squeeze(0)


def joint_random_crop(
    prev: torch.Tensor,
    nxt: torch.Tensor,
    target: torch.Tensor,
    crop_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not crop_size:
        return prev, nxt, target

    prev = ensure_min_size(prev, crop_size, crop_size)
    nxt = ensure_min_size(nxt, crop_size, crop_size)
    target = ensure_min_size(target, crop_size, crop_size)

    _, h, w = prev.shape
    top, left = random_crop_params(h, w, crop_size, crop_size)

    prev = apply_crop(prev, top, left, crop_size, crop_size)
    nxt = apply_crop(nxt, top, left, crop_size, crop_size)
    target = apply_crop(target, top, left, crop_size, crop_size)
    return prev, nxt, target
