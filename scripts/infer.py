from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from denoise_interpolate.models import build_model
from denoise_interpolate.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate a frame between two images.")
    parser.add_argument("--frame-a", required=True, help="Path to previous frame.")
    parser.add_argument("--frame-b", required=True, help="Path to next frame.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--force-rgb", action="store_true")
    return parser.parse_args()


def load_image(path: str, force_rgb: bool) -> torch.Tensor:
    img = read_image(path)
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1 and force_rgb:
        img = img.repeat(3, 1, 1)
    return img.float() / 255.0


def main() -> None:
    args = parse_args()
    device = get_device(prefer_mps=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    frame_a = load_image(args.frame_a, args.force_rgb)
    frame_b = load_image(args.frame_b, args.force_rgb)
    inp = torch.cat([frame_a, frame_b], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).squeeze(0).clamp(0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred.cpu(), str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
