from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lpips
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from denoise_interpolate.baselines import get_baselines
from denoise_interpolate.data import FrameInterpolationDataset
from denoise_interpolate.registry import append_entry, build_entry
from denoise_interpolate.utils import get_device, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline interpolation methods.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--methods", default="blend", help="Comma-separated baseline methods.")
    parser.add_argument("--out-dir", default="outputs/baselines")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force-rgb", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-visuals", type=int, default=16)
    parser.add_argument("--lpips-max-samples", type=int, default=256)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def split_prev_next(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = inp.shape[1] // 2
    return inp[:, :c], inp[:, c:]


def main() -> None:
    args = parse_args()
    device = get_device(prefer_mps=True)

    dataset = FrameInterpolationDataset(args.manifest, force_rgb=args.force_rgb)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_methods = get_baselines()
    for m in methods:
        if m not in all_methods:
            raise SystemExit(f"Unknown method: {m}. Options: {list(all_methods.keys())}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for method_name in methods:
        method = all_methods[method_name]
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_model = lpips.LPIPS(net="alex").to(device)

        l1_sum = 0.0
        lpips_sum = 0.0
        lpips_count = 0
        sample_count = 0
        visuals: List[torch.Tensor] = []

        for batch in tqdm(loader, desc=f"Eval {method_name}"):
            batch = to_device(batch, device)
            inp = batch["input"]
            target = batch["target"]
            t = batch.get("t", None)

            prev, nxt = split_prev_next(inp)

            preds = []
            # Baselines operate per-sample on CPU for optical flow.
            for i in range(prev.shape[0]):
                if t is None:
                    t_val = 0.5
                else:
                    t_val = float(t[i].item())
                pred = method(prev[i].cpu(), nxt[i].cpu(), t_val)
                preds.append(pred)
            pred = torch.stack(preds, dim=0).to(device)

            l1_sum += F.l1_loss(pred, target, reduction="mean").item() * pred.shape[0]
            psnr.update(pred, target)
            ssim.update(pred, target)

            if lpips_count < args.lpips_max_samples:
                pred_lp = pred * 2 - 1
                targ_lp = target * 2 - 1
                lp = lpips_model(pred_lp, targ_lp)
                lpips_sum += lp.mean().item() * pred.shape[0]
                lpips_count += pred.shape[0]

            if len(visuals) < args.num_visuals:
                for i in range(pred.shape[0]):
                    if len(visuals) >= args.num_visuals:
                        break
                    visuals.extend([
                        prev[i].cpu(),
                        pred[i].cpu(),
                        target[i].cpu(),
                        nxt[i].cpu(),
                    ])

            sample_count += pred.shape[0]
            if args.max_samples and sample_count >= args.max_samples:
                break

        metrics: Dict[str, float] = {
            "l1": l1_sum / max(1, sample_count),
            "psnr": psnr.compute().item(),
            "ssim": ssim.compute().item(),
        }
        if lpips_count > 0:
            metrics["lpips"] = lpips_sum / max(1, lpips_count)

        out_dir = out_root / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        report_path = out_dir / "report.md"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"# Baseline Report: {method_name}\n\n")
            f.write(f"Manifest: `{args.manifest}`\n\n")
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
                checkpoint="baseline",
                manifest=args.manifest,
                report=str(report_path),
                tag=args.tag or None,
                notes=f"baseline:{method_name}",
                cfg={"model": {"name": f"baseline_{method_name}"}},
                extra={"model_name": f"baseline_{method_name}"},
            )
            append_entry(args.registry, entry)

        print(f"Wrote baseline report to {report_path}")


if __name__ == "__main__":
    main()
