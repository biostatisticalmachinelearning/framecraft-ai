from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import cv2


def _to_bgr_uint8(img: torch.Tensor) -> np.ndarray:
    # img: (C, H, W), float [0,1]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.clamp(0, 1)
    np_img = (img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def _to_rgb_float(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def _warp(img: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0] * scale).astype(np.float32)
    map_y = (grid_y + flow[..., 1] * scale).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def baseline_hold_prev(prev: torch.Tensor, nxt: torch.Tensor, t: float) -> torch.Tensor:
    return prev


def baseline_hold_next(prev: torch.Tensor, nxt: torch.Tensor, t: float) -> torch.Tensor:
    return nxt


def baseline_blend(prev: torch.Tensor, nxt: torch.Tensor, t: float) -> torch.Tensor:
    return prev * (1.0 - t) + nxt * t


def baseline_flow(prev: torch.Tensor, nxt: torch.Tensor, t: float) -> torch.Tensor:
    prev_bgr = _to_bgr_uint8(prev)
    next_bgr = _to_bgr_uint8(nxt)

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)

    flow_fwd = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_bwd = cv2.calcOpticalFlowFarneback(
        next_gray,
        prev_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    pred_fwd = _warp(prev_bgr, flow_fwd, t)
    pred_bwd = _warp(next_bgr, flow_bwd, 1.0 - t)
    pred = ((pred_fwd.astype(np.float32) + pred_bwd.astype(np.float32)) * 0.5).astype(np.uint8)
    return _to_rgb_float(pred)


def get_baselines() -> Dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]]:
    return {
        "hold_prev": baseline_hold_prev,
        "hold_next": baseline_hold_next,
        "blend": baseline_blend,
        "flow": baseline_flow,
    }
