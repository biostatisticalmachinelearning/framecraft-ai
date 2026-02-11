from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class FrameInterpolationDataset(Dataset):
    """Loads triplets of (prev, next, target) frames from a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        force_rgb: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}. Run scripts/prepare_data.py."
            )
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        self.force_rgb = force_rgb
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: str | Path) -> torch.Tensor:
        img = read_image(str(path))  # (C, H, W), uint8
        if img.shape[0] == 4:  # drop alpha
            img = img[:3]
        if img.shape[0] == 1 and self.force_rgb:
            img = img.repeat(3, 1, 1)
        img = img.float() / 255.0
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        prev = self._load_image(item["prev"])
        nxt = self._load_image(item["next"])
        target = self._load_image(item["target"])

        if self.transform is not None:
            prev = self.transform(prev)
            nxt = self.transform(nxt)
            target = self.transform(target)

        # Concatenate prev/next along channel dimension -> (2C, H, W)
        inp = torch.cat([prev, nxt], dim=0)
        output = {
            "input": inp,
            "target": target,
        }
        if "t" in item:
            output["t"] = torch.tensor(float(item["t"]), dtype=torch.float32)
        return output
