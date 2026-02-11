from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torchvision.io import read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA checks for downloaded datasets.")
    parser.add_argument("--registry", default="data/registry.jsonl")
    parser.add_argument("--out-dir", default="outputs/qa")
    parser.add_argument("--check-hash", action="store_true")
    parser.add_argument("--frames-root", default="")
    parser.add_argument("--frames-sample", type=int, default=0)
    parser.add_argument("--max-movies", type=int, default=0)
    parser.add_argument("--force-rgb", action="store_true")
    parser.add_argument("--gray-tol", type=float, default=1e-3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def iter_movie_dirs(frames_root: Path) -> List[Path]:
    return sorted([p for p in frames_root.iterdir() if p.is_dir()])


def sample_frames(movie_dir: Path, k: int) -> List[Path]:
    frames = sorted(movie_dir.glob("*.png"))
    if not frames:
        frames = sorted(movie_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(movie_dir.glob("*.jpeg"))
    if k <= 0 or len(frames) <= k:
        return frames
    step = max(1, len(frames) // k)
    return frames[::step][:k]


def is_grayscale(img: torch.Tensor, tol: float) -> bool:
    if img.shape[0] == 1:
        return True
    if img.shape[0] < 3:
        return False
    diff01 = (img[0] - img[1]).abs().mean().item()
    diff02 = (img[0] - img[2]).abs().mean().item()
    return max(diff01, diff02) < tol


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_registry(registry_path)
    missing_files = []
    size_mismatch = []
    hash_mismatch = []

    seen_hashes: Dict[str, List[str]] = defaultdict(list)
    total_bytes = 0

    for entry in entries:
        path = Path(entry.get("file_path", ""))
        if not path.exists():
            missing_files.append(str(path))
            continue
        size = path.stat().st_size
        total_bytes += size
        if entry.get("size_bytes") and int(entry["size_bytes"]) != size:
            size_mismatch.append(str(path))
        if args.check_hash:
            digest = sha256_file(path)
            if entry.get("sha256") and entry["sha256"] != digest:
                hash_mismatch.append(str(path))
            seen_hashes[digest].append(str(path))
        else:
            digest = entry.get("sha256")
            if digest:
                seen_hashes[digest].append(str(path))

    duplicates = {h: paths for h, paths in seen_hashes.items() if len(paths) > 1}

    license_missing = [e.get("id") for e in entries if not e.get("license")]
    license_url_missing = [e.get("id") for e in entries if not e.get("license_url")]

    frame_stats = {}
    if args.frames_root:
        frames_root = Path(args.frames_root)
        movies = iter_movie_dirs(frames_root)
        if args.max_movies > 0:
            movies = movies[: args.max_movies]

        for movie in movies:
            frames = sample_frames(movie, args.frames_sample)
            if not frames:
                continue

            sum_c = None
            sumsq_c = None
            count_px = 0
            gray_count = 0
            resolutions = Counter()

            for frame in frames:
                img = read_image(str(frame)).float() / 255.0
                if img.shape[0] == 4:
                    img = img[:3]
                if img.shape[0] == 1 and args.force_rgb:
                    img = img.repeat(3, 1, 1)
                if img.shape[0] >= 3 and args.force_rgb:
                    img = img[:3]

                if is_grayscale(img, args.gray_tol):
                    gray_count += 1
                resolutions[(int(img.shape[1]), int(img.shape[2]))] += 1

                c = img.shape[0]
                if sum_c is None:
                    sum_c = torch.zeros(c)
                    sumsq_c = torch.zeros(c)
                elif sum_c.numel() != c:
                    if c == 1 and sum_c.numel() == 3:
                        img = img.repeat(3, 1, 1)
                        c = 3
                    elif c == 3 and sum_c.numel() == 1:
                        img = img.mean(dim=0, keepdim=True)
                        c = 1
                    else:
                        continue
                flat = img.view(c, -1)
                sum_c += flat.sum(dim=1)
                sumsq_c += (flat ** 2).sum(dim=1)
                count_px += flat.shape[1]

            if sum_c is not None:
                mean_t = sum_c / max(1, count_px)
                var_t = sumsq_c / max(1, count_px) - mean_t ** 2
                mean = mean_t.tolist()
                std = [float(v) ** 0.5 for v in var_t.tolist()]
            else:
                mean = []
                std = []

            frame_stats[movie.name] = {
                "num_frames": len(list(movie.iterdir())),
                "sampled": len(frames),
                "mean": mean,
                "std": std,
                "grayscale_ratio": gray_count / max(1, len(frames)),
                "resolutions": {f"{k[0]}x{k[1]}": v for k, v in resolutions.items()},
            }

    summary = {
        "registry_entries": len(entries),
        "missing_files": missing_files,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "duplicate_hashes": duplicates,
        "license_missing": license_missing,
        "license_url_missing": license_url_missing,
        "total_bytes": total_bytes,
        "frame_stats": frame_stats,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Dataset QA Report\n\n")
        f.write(f"Registry: `{registry_path}`\n\n")
        f.write("## File Checks\n\n")
        f.write(f"- Registry entries: {len(entries)}\n")
        f.write(f"- Missing files: {len(missing_files)}\n")
        f.write(f"- Size mismatches: {len(size_mismatch)}\n")
        f.write(f"- Hash mismatches: {len(hash_mismatch)}\n")
        f.write(f"- Duplicate hashes: {len(duplicates)}\n")
        f.write(f"- Missing license: {len(license_missing)}\n")
        f.write(f"- Missing license URL: {len(license_url_missing)}\n")
        f.write(f"- Total bytes (on disk): {total_bytes}\n\n")

        if frame_stats:
            f.write("## Frame Stats (Sampled)\n\n")
            for movie, stats in frame_stats.items():
                f.write(f"### {movie}\n")
                f.write(f"- Sampled: {stats['sampled']}\n")
                f.write(f"- Mean: {stats['mean']}\n")
                f.write(f"- Std: {stats['std']}\n")
                f.write(f"- Grayscale ratio: {stats['grayscale_ratio']:.3f}\n")
                f.write(f"- Resolutions: {stats['resolutions']}\n\n")

    print(f"Wrote QA report to {report_path}")


if __name__ == "__main__":
    main()
