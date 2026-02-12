from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from denoise_interpolate.models import build_model
from denoise_interpolate.utils import get_device

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise original frames and interpolate mid-frames to double FPS."
    )
    parser.add_argument("--frames-dir", help="Directory of extracted frames.")
    parser.add_argument("--output-dir", default="outputs/interpolated")
    parser.add_argument("--output-video", default="", help="Optional MP4 output path.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ext", default="png", help="Frame extension for outputs.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--no-denoise-original",
        action="store_true",
        help="Skip denoising originals; only predict interleaved frames.",
    )
    parser.add_argument("--crf", type=int, default=16, help="CRF for x264 output.")
    parser.add_argument("--preset", default="slow", help="Encoder preset for x264.")
    parser.add_argument("--device", default="", help="Force device: cpu/cuda/mps.")
    return parser.parse_args()


def load_frame(path: Path) -> torch.Tensor:
    img = read_image(str(path))
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img.float() / 255.0


def save_frame(img: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img.clamp(0, 1), str(path))


def batch_predict(
    model: torch.nn.Module, pairs: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> List[torch.Tensor]:
    if not pairs:
        return []
    inputs = []
    for prev, nxt in pairs:
        inputs.append(torch.cat([prev, nxt], dim=0))
    batch = torch.stack(inputs, dim=0).to(device)
    with torch.no_grad():
        pred = model(batch).clamp(0, 1)
    return [p.detach().cpu() for p in pred]


def build_output_sequence(
    originals: List[torch.Tensor],
    denoised: List[torch.Tensor] | None,
    mids: List[torch.Tensor],
) -> List[torch.Tensor]:
    output = []
    n = len(originals)
    for i in range(n - 1):
        prev = denoised[i] if denoised else originals[i]
        nxt = denoised[i + 1] if denoised else originals[i + 1]
        if i == 0:
            output.append(prev)
        output.append(mids[i])
        output.append(nxt)
    return output


def encode_video(frames_dir: Path, output_video: Path, fps: float, crf: int, preset: str, ext: str) -> None:
    import subprocess

    pattern = frames_dir / f"%06d.{ext}"
    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir is None:
        raise SystemExit("--frames-dir is required")

    if not frames_dir.exists():
        raise SystemExit(f"Frames dir not found: {frames_dir}")

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    device = (
        torch.device(args.device)
        if args.device
        else get_device(prefer_mps=True)
    )
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("cfg", {}).get("model", {})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    originals = [load_frame(p) for p in tqdm(frame_paths, desc="Load frames")]

    denoise = not args.no_denoise_original
    denoised = None
    if denoise:
        denoised = []
        denoised.append(originals[0])
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        indices: List[int] = []
        for i in range(1, len(originals) - 1):
            pairs.append((originals[i - 1], originals[i + 1]))
            indices.append(i)
            if len(pairs) >= args.batch_size:
                preds = batch_predict(model, pairs, device)
                denoised.extend(preds)
                pairs.clear()
                indices.clear()
        if pairs:
            preds = batch_predict(model, pairs, device)
            denoised.extend(preds)
        denoised.append(originals[-1])

    mids: List[torch.Tensor] = []
    pairs = []
    for i in range(len(originals) - 1):
        prev = denoised[i] if denoised else originals[i]
        nxt = denoised[i + 1] if denoised else originals[i + 1]
        pairs.append((prev, nxt))
        if len(pairs) >= args.batch_size:
            mids.extend(batch_predict(model, pairs, device))
            pairs.clear()
    if pairs:
        mids.extend(batch_predict(model, pairs, device))

    output_frames = build_output_sequence(originals, denoised, mids)

    out_dir = Path(args.output_dir)
    frames_out = out_dir / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(tqdm(output_frames, desc="Write output frames")):
        out_path = frames_out / f"{i:06d}.{args.ext}"
        save_frame(frame, out_path)

    if args.output_video:
        output_video = Path(args.output_video)
        encode_video(frames_out, output_video, fps=args.fps * 2.0, crf=args.crf, preset=args.preset, ext=args.ext)

    print(f"Wrote {len(output_frames)} frames to {frames_out}")


if __name__ == "__main__":
    main()
