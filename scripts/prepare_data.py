from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSONL manifests from extracted frames.")
    parser.add_argument("--frames-root", required=True, help="Root dir with movie subfolders.")
    parser.add_argument("--train-out", default="data/manifests/train.jsonl")
    parser.add_argument("--val-out", default="data/manifests/val.jsonl")
    parser.add_argument("--test-out", default="data/manifests/test.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit-per-movie", type=int, default=0)
    parser.add_argument("--t", type=float, default=0.5, help="Interpolation time between frames.")
    parser.add_argument(
        "--extensions",
        default="png,jpg,jpeg",
        help="Comma-separated list of frame extensions.",
    )
    return parser.parse_args()


def list_movies(frames_root: Path) -> List[Path]:
    return sorted([p for p in frames_root.iterdir() if p.is_dir()])


def list_frames(movie_dir: Path, extensions: Iterable[str]) -> List[Path]:
    frames: List[Path] = []
    for ext in extensions:
        frames.extend(movie_dir.glob(f"*.{ext}"))
    return sorted(frames)


def write_manifest(path: Path, entries: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def build_entries(
    movie_dir: Path, frames: List[Path], stride: int, limit: int, t: float
) -> List[dict]:
    entries = []
    count = 0
    for i in range(1, len(frames) - 1, stride):
        entry = {
            "prev": str(frames[i - 1]),
            "next": str(frames[i + 1]),
            "target": str(frames[i]),
            "movie_id": movie_dir.name,
            "t": t,
        }
        entries.append(entry)
        count += 1
        if limit > 0 and count >= limit:
            break
    return entries


def main() -> None:
    args = parse_args()
    frames_root = Path(args.frames_root)
    extensions = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]

    movies = list_movies(frames_root)
    if not movies:
        raise SystemExit(f"No movie subfolders found in: {frames_root}")

    random.seed(args.seed)
    random.shuffle(movies)

    n_total = len(movies)
    n_val = int(n_total * args.val_ratio)
    n_test = int(n_total * args.test_ratio)
    if n_val + n_test >= n_total:
        n_val = max(0, min(n_val, n_total - 1))
        n_test = max(0, min(n_test, n_total - 1 - n_val))
    n_train = n_total - n_val - n_test
    if n_train == 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    train_movies = movies[:n_train]
    val_movies = movies[n_train : n_train + n_val]
    test_movies = movies[n_train + n_val : n_train + n_val + n_test]

    def collect(movies_subset: List[Path]) -> List[dict]:
        all_entries: List[dict] = []
        for movie in movies_subset:
            frames = list_frames(movie, extensions)
            if len(frames) < 3:
                continue
            all_entries.extend(
                build_entries(
                    movie, frames, stride=args.stride, limit=args.limit_per_movie, t=args.t
                )
            )
        return all_entries

    train_entries = collect(train_movies)
    val_entries = collect(val_movies)
    test_entries = collect(test_movies)

    write_manifest(Path(args.train_out), train_entries)
    write_manifest(Path(args.val_out), val_entries)
    write_manifest(Path(args.test_out), test_entries)

    print(f"Movies: train={len(train_movies)}, val={len(val_movies)}, test={len(test_movies)}")
    print(f"Entries: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")


if __name__ == "__main__":
    main()
