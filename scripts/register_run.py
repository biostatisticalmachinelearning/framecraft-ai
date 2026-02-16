from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from denoise_interpolate.registry import append_entry, build_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an evaluation run.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to manifest used")
    parser.add_argument("--report", default="", help="Path to report.md")
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--tag", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-cfg", action="store_true", help="Skip loading cfg from checkpoint")
    return parser.parse_args()


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(Path(args.metrics))

    cfg = None
    if not args.no_cfg:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", None)

    entry = build_entry(
        metrics=metrics,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        report=args.report or None,
        tag=args.tag or None,
        notes=args.notes or None,
        cfg=cfg,
    )
    append_entry(args.registry, entry)
    print(f"Appended entry to {args.registry}")


if __name__ == "__main__":
    main()
