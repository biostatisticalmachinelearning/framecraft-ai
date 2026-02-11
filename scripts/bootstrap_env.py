from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create venv and install dependencies.")
    parser.add_argument("--venv", default=".venv", help="Path to virtual environment.")
    parser.add_argument("--cuda", action="store_true", help="Install CUDA-enabled PyTorch.")
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only PyTorch.")
    parser.add_argument("--torch-index", default="", help="Override PyTorch index URL.")
    parser.add_argument("--dev", action="store_true", help="Install dev extras.")
    return parser.parse_args()


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    venv_dir = Path(args.venv)
    if not venv_dir.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])

    py = venv_python(venv_dir)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    torch_index = args.torch_index
    if args.cuda:
        if not torch_index:
            torch_index = "https://download.pytorch.org/whl/cu118"
        run([str(py), "-m", "pip", "install", "torch", "torchvision", "--index-url", torch_index])
    elif args.cpu:
        if not torch_index:
            torch_index = "https://download.pytorch.org/whl/cpu"
        run([str(py), "-m", "pip", "install", "torch", "torchvision", "--index-url", torch_index])
    else:
        # Default: let pip choose the platform-appropriate build (MPS on Apple Silicon).
        run([str(py), "-m", "pip", "install", "torch", "torchvision"])

    extras = "[dev]" if args.dev else ""
    run([str(py), "-m", "pip", "install", "-e", f".{extras}"])

    print("Environment ready.")


if __name__ == "__main__":
    main()
