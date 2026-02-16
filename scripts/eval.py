from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from denoise_interpolate.data import FrameInterpolationDataset
from denoise_interpolate.models import build_model
from denoise_interpolate.utils import get_device, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frame interpolation model.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(prefer_mps=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = FrameInterpolationDataset(args.manifest, force_rgb=args.force_rgb)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn = torch.nn.L1Loss()

    loss_total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            batch = to_device(batch, device)
            pred = model(batch["input"])
            loss = loss_fn(pred, batch["target"])
            loss_total += loss.item()
            psnr.update(pred, batch["target"])
            ssim.update(pred, batch["target"])

    avg_loss = loss_total / max(1, len(loader))
    print(f"L1: {avg_loss:.6f}")
    print(f"PSNR: {psnr.compute().item():.4f}")
    print(f"SSIM: {ssim.compute().item():.4f}")


if __name__ == "__main__":
    main()
