from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def compute_hash(obj: Any) -> str:
    payload = _json_dumps(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_entry(
    *,
    metrics: Dict[str, Any],
    checkpoint: str,
    manifest: str,
    report: Optional[str] = None,
    tag: Optional[str] = None,
    notes: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint": checkpoint,
        "manifest": manifest,
        "metrics": metrics,
    }
    if report:
        entry["report"] = report
    if tag:
        entry["tag"] = tag
    if notes:
        entry["notes"] = notes
    if cfg is not None:
        entry["cfg"] = cfg
        entry["cfg_hash"] = compute_hash(cfg)
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        entry["model_name"] = model_cfg.get("name", "unknown")
    if extra:
        entry.update(extra)
    return entry


def append_entry(registry_path: str | Path, entry: Dict[str, Any]) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_json_dumps(entry) + "\n")
