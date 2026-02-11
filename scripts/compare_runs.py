from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare registered runs.")
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--metric", default="psnr")
    parser.add_argument("--model", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="")
    parser.add_argument("--ascending", action="store_true")
    return parser.parse_args()


def load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def default_ascending(metric: str) -> bool:
    return metric.lower() in {"lpips", "l1", "loss"}


def main() -> None:
    args = parse_args()
    entries = load_entries(Path(args.registry))

    if args.model:
        entries = [e for e in entries if e.get("model_name") == args.model]
    if args.tag:
        entries = [e for e in entries if e.get("tag") == args.tag]

    metric = args.metric
    entries = [e for e in entries if metric in (e.get("metrics") or {})]

    asc = args.ascending or default_ascending(metric)
    entries.sort(key=lambda e: e["metrics"][metric], reverse=not asc)

    rows = []
    for e in entries[: args.limit]:
        rows.append(
            {
                "timestamp": e.get("timestamp", ""),
                "model": e.get("model_name", ""),
                metric: e.get("metrics", {}).get(metric, None),
                "tag": e.get("tag", ""),
                "checkpoint": e.get("checkpoint", ""),
            }
        )

    if not rows:
        print("No entries match.")
        return

    header = ["timestamp", "model", metric, "tag", "checkpoint"]
    lines = ["| " + " | ".join(header) + " |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(h, "")) for h in header)
            + " |"
        )

    output = "\n".join(lines)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote comparison to {out_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
