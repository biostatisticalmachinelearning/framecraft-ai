from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames using ffmpeg.")
    parser.add_argument("--input", help="Path to a video file.")
    parser.add_argument("--input-dir", help="Directory of video files.")
    parser.add_argument("--output-dir", default="data/frames")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--allow-non-24",
        action="store_true",
        help="Allow extracting at a frame rate other than 24 fps.",
    )
    parser.add_argument("--scale", default="", help="Scale filter, e.g. 'scale=-2:720'")
    parser.add_argument("--ext", default="png", help="Frame image extension.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_videos(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def build_filter(fps: int, scale: str) -> str:
    parts = [f"fps={fps}"]
    if scale:
        parts.append(scale)
    return ",".join(parts)


def run_ffmpeg(input_path: Path, output_dir: Path, fps: int, scale: str, ext: str, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"%06d.{ext}"
    vf = build_filter(fps, scale)
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vf",
        vf,
        str(pattern),
    ]
    if overwrite:
        cmd.insert(1, "-y")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not args.input and not args.input_dir:
        raise SystemExit("Provide --input or --input-dir")
    if args.fps != 24 and not args.allow_non_24:
        raise SystemExit(
            "This project expects 24 fps source frames. "
            "Re-run with --fps 24 or pass --allow-non-24 to override."
        )

    output_root = Path(args.output_dir)
    if args.input:
        input_path = Path(args.input)
        out_dir = output_root / input_path.stem
        run_ffmpeg(input_path, out_dir, args.fps, args.scale, args.ext, args.overwrite)
        return

    input_dir = Path(args.input_dir)
    for video in list_videos(input_dir):
        out_dir = output_root / video.stem
        run_ffmpeg(video, out_dir, args.fps, args.scale, args.ext, args.overwrite)


if __name__ == "__main__":
    main()
