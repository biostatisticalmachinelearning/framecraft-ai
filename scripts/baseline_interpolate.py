from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from denoise_interpolate.baselines import get_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline interpolation between two frames.")
    parser.add_argument("--frame-a", required=True)
    parser.add_argument("--frame-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="blend", choices=list(get_baselines().keys()))
    parser.add_argument("--t", type=float, default=0.5)
    return parser.parse_args()


def load_image(path: str) -> torch.Tensor:
    img = read_image(path)
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img.float() / 255.0


def main() -> None:
    args = parse_args()
    prev = load_image(args.frame_a)
    nxt = load_image(args.frame_b)

    method = get_baselines()[args.method]
    pred = method(prev, nxt, args.t).clamp(0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
