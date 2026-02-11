from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import yaml
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".mpg", ".mpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download curated sources and register hashes.")
    parser.add_argument("--sources", default="configs/sources.yaml")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--registry", default="data/registry.jsonl")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--max-total-gb", type=float, default=20.0)
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--loc-format", choices=["mp4", "mov"], default="mp4")
    parser.add_argument("--overwrite", action="store_true", help="Re-download if file exists.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sources(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def collect_video_urls(obj: Any, urls: List[str]) -> None:
    if isinstance(obj, dict):
        for _, v in obj.items():
            collect_video_urls(v, urls)
    elif isinstance(obj, list):
        for v in obj:
            collect_video_urls(v, urls)
    elif isinstance(obj, str):
        if obj.startswith("http") and obj.lower().endswith(VIDEO_EXTS):
            urls.append(obj)


def loc_resolve_download(item_url: str, prefer_ext: str) -> str:
    json_url = item_url.rstrip("/") + "/?fo=json"
    resp = requests.get(json_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    urls: List[str] = []
    collect_video_urls(data, urls)
    if not urls:
        raise RuntimeError(f"No video URLs found in LOC JSON: {json_url}")

    prefer_ext = "." + prefer_ext.lower().lstrip(".")
    # Prefer smaller MP4 for local development.
    def score(url: str) -> Tuple[int, int]:
        ext = os.path.splitext(url)[1].lower()
        return (0 if ext == prefer_ext else 1, len(url))

    urls = sorted(set(urls))
    return sorted(urls, key=score)[0]


def ia_resolve_download(identifier: str, quality: str) -> str:
    meta_url = f"https://archive.org/metadata/{identifier}"
    resp = requests.get(meta_url, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    prefer_formats = {
        "low": ["H.264", "MPEG4"],
        "medium": ["H.264", "MPEG4", "HIRES MPEG4"],
        "high": ["HIRES MPEG4", "MPEG4", "H.264"],
    }[quality]

    files = meta.get("files", [])
    candidates = []
    for f in files:
        name = f.get("name", "")
        fmt = f.get("format", "")
        if not name:
            continue
        if name.lower().endswith(VIDEO_EXTS):
            candidates.append((fmt, name, f))

    if not candidates:
        raise RuntimeError(f"No video files found in IA metadata: {meta_url}")

    def fmt_rank(fmt: str) -> int:
        return prefer_formats.index(fmt) if fmt in prefer_formats else len(prefer_formats)

    candidates.sort(key=lambda x: (fmt_rank(x[0]), x[1]))
    chosen = candidates[0][1]
    return f"https://archive.org/download/{identifier}/{chosen}"


def stream_download(url: str, dest: Path) -> Tuple[int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()

    bytes_written = 0
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                sha256.update(chunk)
                bytes_written += len(chunk)
                pbar.update(len(chunk))
            pbar.close()
    if total == 0:
        total = bytes_written
    return total, sha256.hexdigest()


def main() -> None:
    args = parse_args()
    sources = load_sources(args.sources)

    out_dir = Path(args.out_dir)
    registry_path = Path(args.registry)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    downloaded = 0

    for src in sources:
        if downloaded >= args.max_items:
            break

        provider = src.get("provider")
        title = src.get("title")
        item_url = src.get("item_url")

        if provider == "loc":
            download_url = src.get("download_url") or loc_resolve_download(
                item_url, args.loc_format
            )
        elif provider == "ia":
            identifier = src.get("identifier")
            download_url = src.get("download_url") or ia_resolve_download(
                identifier, args.quality
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        filename = download_url.split("/")[-1]
        dest = out_dir / src["id"] / filename

        if args.dry_run:
            print(f"{src['id']}: {title} -> {download_url}")
            downloaded += 1
            continue

        if dest.exists() and not args.overwrite:
            print(f"Skipping {src['id']} (already exists): {dest}")
            downloaded += 1
            continue

        # Optional size check via HEAD
        size = 0
        try:
            head = requests.head(download_url, allow_redirects=True, timeout=30)
            size = int(head.headers.get("content-length", 0))
        except requests.RequestException:
            size = 0
        projected = total_bytes + size
        if size > 0 and projected > args.max_total_gb * (1024**3):
            print(f"Skipping {src['id']} (size exceeds max-total-gb).")
            continue

        print(f"Downloading {title}...")
        size_bytes, sha256 = stream_download(download_url, dest)
        total_bytes += size_bytes

        entry = {
            "id": src.get("id"),
            "title": title,
            "provider": provider,
            "item_url": item_url,
            "download_url": download_url,
            "license": src.get("license"),
            "license_url": src.get("license_url"),
            "file_path": str(dest),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "notes": src.get("notes"),
        }

        with registry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        downloaded += 1

    print(f"Downloaded {downloaded} items. Registry: {registry_path}")


if __name__ == "__main__":
    main()
