from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import requests
import yaml

DEFAULT_PRELINGER_URL = (
    "https://huggingface.co/datasets/davanstrien/"
    "prelinger-archives-open/resolve/main/data/metadata.jsonl"
)

ALLOWED_LICENSE_URLS = {
    "http://creativecommons.org/publicdomain/mark/1.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/publicdomain/",
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh configs/sources.yaml with Prelinger items to reach a target count."
    )
    parser.add_argument("--output", default="configs/sources.yaml")
    parser.add_argument("--total-target", type=int, default=100)
    parser.add_argument("--prelinger-url", default=DEFAULT_PRELINGER_URL)
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Ignore existing sources and start fresh.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_sources(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("sources", [])


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def iter_prelinger(
    url: str,
    max_items: int,
    existing_ids: Set[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            license_url = obj.get("licenseurl")
            if license_url and license_url not in ALLOWED_LICENSE_URLS:
                continue
            identifier = obj.get("identifier")
            if not identifier:
                continue
            src_id = f"ia_{slugify(identifier)}"
            if src_id in existing_ids:
                continue
            title = obj.get("title") or identifier
            entry = {
                "id": src_id,
                "title": title,
                "provider": "ia",
                "identifier": identifier,
                "item_url": obj.get("source_url")
                or f"https://archive.org/details/{identifier}",
                "license": obj.get("license_type") or "Prelinger open-license",
                "license_url": license_url or "",
                "notes": "Prelinger Archive (open-license dataset)",
            }
            results.append(entry)
            existing_ids.add(src_id)
            if len(results) >= max_items:
                break
    return results


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    sources = load_sources(output_path)
    if args.drop_existing:
        sources = []

    existing_ids = {s.get("id", "") for s in sources}
    existing_ids.discard("")

    remaining = max(0, args.total_target - len(sources))
    if remaining > 0:
        sources.extend(iter_prelinger(args.prelinger_url, remaining, existing_ids))

    payload = {"sources": sources}
    if args.dry_run:
        print(yaml.safe_dump(payload, sort_keys=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"Wrote {len(sources)} sources to {output_path}")


if __name__ == "__main__":
    main()
