from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

from tqdm import tqdm

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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input-dir recursively for videos (useful if videos are in subfolders).",
    )
    parser.add_argument("--scale", default="", help="Scale filter, e.g. 'scale=-2:720'")
    parser.add_argument("--ext", default="png", help="Frame image extension (lossless only).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0=full).")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel extraction across videos (directory mode only).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (per-movie only when --workers 1).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_videos(path: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS])
    return sorted([p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def build_filter(fps: int, scale: str) -> str:
    parts = [f"fps={fps}"]
    if scale:
        parts.append(scale)
    return ",".join(parts)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def parse_time_str(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(s) + 60 * float(m) + 3600 * float(h)
    try:
        return float(value)
    except ValueError:
        return 0.0


def run_ffmpeg(
    input_path: Path,
    output_dir: Path,
    fps: int,
    scale: str,
    ext: str,
    overwrite: bool,
    start: float,
    duration: float,
    progress: bool,
    total_seconds: Optional[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"%06d.{ext}"
    vf = build_filter(fps, scale)
    cmd = ["ffmpeg"]
    if overwrite:
        cmd.append("-y")
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_path)]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", vf]
    if progress and total_seconds and total_seconds > 0:
        cmd += ["-progress", "pipe:1", "-nostats", "-v", "error"]
    cmd += [str(pattern)]
    print(" ".join(cmd))
    if not (progress and total_seconds and total_seconds > 0):
        subprocess.run(cmd, check=True)
        return

    pbar = tqdm(total=total_seconds, desc=input_path.name, unit="s", leave=False)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    if proc.stdout is None:
        raise RuntimeError("Failed to read ffmpeg progress output.")
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time="):
            sec = parse_time_str(line.split("=", 1)[1])
            if sec > pbar.n:
                pbar.n = min(sec, total_seconds)
                pbar.refresh()
        elif line.startswith("progress=") and line.endswith("end"):
            break
    stdout, stderr = proc.communicate()
    pbar.n = total_seconds
    pbar.refresh()
    pbar.close()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)


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
    if args.ext.lower() not in {"png"}:
        raise SystemExit(
            "Only lossless extraction is allowed. Use --ext png."
        )
    if args.input:
        input_path = Path(args.input)
        out_dir = output_root / input_path.stem
        total_seconds = (
            args.duration
            if args.duration > 0
            else max(0.0, ffprobe_duration(input_path) - args.start)
        )
        run_ffmpeg(
            input_path,
            out_dir,
            args.fps,
            args.scale,
            args.ext,
            args.overwrite,
            args.start,
            args.duration,
            args.progress,
            total_seconds,
        )
        return

    input_dir = Path(args.input_dir)
    videos = list_videos(input_dir, args.recursive)
    if not videos:
        raise SystemExit(
            f"No videos found under {input_dir}. "
            "If your videos are in subfolders, pass --recursive or use --input."
        )
    def job(video: Path) -> None:
        if args.recursive and video.parent != input_dir:
            out_dir = output_root / video.parent.name
        else:
            out_dir = output_root / video.stem
        total_seconds = None
        if args.progress and args.workers <= 1:
            total_seconds = (
                args.duration
                if args.duration > 0
                else max(0.0, ffprobe_duration(video) - args.start)
            )
        run_ffmpeg(
            video,
            out_dir,
            args.fps,
            args.scale,
            args.ext,
            args.overwrite,
            args.start,
            args.duration,
            args.progress and args.workers <= 1,
            total_seconds,
        )

    if args.workers <= 1:
        iterable = tqdm(videos, desc="Movies") if args.progress else videos
        for video in iterable:
            job(video)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(job, v) for v in videos]
            if args.progress:
                outer = tqdm(total=len(videos), desc="Movies")
                for future in as_completed(futures):
                    future.result()
                    outer.update(1)
                outer.close()
            else:
                for future in as_completed(futures):
                    future.result()


if __name__ == "__main__":
    main()
