from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from denoise_interpolate.data import FrameInterpolationDataset
from denoise_interpolate.models import build_model
from denoise_interpolate.registry import append_entry, build_entry
from denoise_interpolate.utils import get_device, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and report metrics + visuals.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--out-dir", default="outputs/report")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-rgb", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--num-visuals", type=int, default=16)
    parser.add_argument("--lpips-max-samples", type=int, default=256)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--tag", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def load_model(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = get_device(prefer_mps=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})

    dataset = FrameInterpolationDataset(args.manifest, force_rgb=args.force_rgb)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(ckpt, device)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device)

    l1_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    sample_count = 0

    visuals: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            batch = to_device(batch, device)
            pred = model(batch["input"]).clamp(0, 1)
            target = batch["target"]

            batch_size = pred.shape[0]
            l1_sum += F.l1_loss(pred, target, reduction="mean").item() * batch_size
            psnr.update(pred, target)
            ssim.update(pred, target)

            if lpips_count < args.lpips_max_samples:
                pred_lp = pred * 2 - 1
                targ_lp = target * 2 - 1
                lp = lpips_model(pred_lp, targ_lp)
                lpips_sum += lp.mean().item() * batch_size
                lpips_count += batch_size

            if len(visuals) < args.num_visuals:
                inp = batch["input"]
                c = inp.shape[1] // 2
                prev = inp[:, :c]
                nxt = inp[:, c:]
                for i in range(batch_size):
                    if len(visuals) >= args.num_visuals:
                        break
                    visuals.extend([prev[i].cpu(), pred[i].cpu(), target[i].cpu(), nxt[i].cpu()])

            sample_count += batch_size
            if args.max_samples and sample_count >= args.max_samples:
                break

    metrics: Dict[str, float] = {
        "l1": l1_sum / max(1, sample_count),
        "psnr": psnr.compute().item(),
        "ssim": ssim.compute().item(),
    }
    if lpips_count > 0:
        metrics["lpips"] = lpips_sum / max(1, lpips_count)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"Manifest: `{args.manifest}`\n\n")
        f.write(f"Checkpoint: `{args.checkpoint}`\n\n")
        f.write("## Metrics\n\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v:.6f}\n")
        f.write("\n## Visuals\n\n")
        f.write("`visuals.png` contains rows of: prev | pred | target | next.\n")

    if visuals:
        grid = make_grid(visuals, nrow=4, padding=2)
        save_image(grid, str(out_dir / "visuals.png"))

    if args.register:
        entry = build_entry(
            metrics=metrics,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            report=str(report_path),
            tag=args.tag or None,
            notes=args.notes or None,
            cfg=cfg if isinstance(cfg, dict) else None,
        )
        append_entry(args.registry, entry)
        print(f"Registered run in {args.registry}")

    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
