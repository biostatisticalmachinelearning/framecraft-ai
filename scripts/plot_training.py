from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


METRICS = {
    "train_loss": "train",
    "val_loss": "val",
    "val_psnr": "val",
    "val_ssim": "val",
}

BASELINE_MAP = {
    "val_loss": "l1",
    "val_psnr": "psnr",
    "val_ssim": "ssim",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training/validation metrics across runs.")
    parser.add_argument("--runs", nargs="+", help="Run directories (Hydra outputs).")
    parser.add_argument(
        "--runs-root",
        default="",
        help="If set, glob for metrics.jsonl under this root when --runs is omitted.",
    )
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for runs.")
    parser.add_argument("--out-dir", default="outputs/plots")
    parser.add_argument(
        "--metrics",
        default="val_loss,val_psnr,val_ssim,train_loss",
        help="Comma-separated metrics to plot.",
    )
    parser.add_argument("--x-axis", choices=["epoch", "step"], default="epoch")
    parser.add_argument("--baseline-dir", default="", help="Path to outputs/baselines for reference lines.")
    return parser.parse_args()


def load_label(run_dir: Path) -> str:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            model = cfg.get("model", {}).get("name", "model")
            residual = cfg.get("model", {}).get("residual", False)
            label = model
            if residual and "res" not in model:
                label = f"{label}+res"
            return label
        except Exception:
            pass
    return run_dir.name


def load_metrics(run_dir: Path) -> List[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    records = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_x(record: dict, x_axis: str) -> float:
    if x_axis == "step":
        return float(record.get("step", 0))
    steps_per_epoch = max(1, int(record.get("steps_per_epoch", 1)))
    step_in_epoch = int(record.get("step_in_epoch", steps_per_epoch))
    return float(record.get("epoch", 0)) + (step_in_epoch / steps_per_epoch)


def load_baselines(baseline_dir: Path) -> Dict[str, dict]:
    baselines = {}
    if not baseline_dir.exists():
        return baselines
    for child in baseline_dir.iterdir():
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.json"
        if metrics_path.exists():
            try:
                baselines[child.name] = json.loads(metrics_path.read_text())
            except Exception:
                continue
    return baselines


def main() -> None:
    args = parse_args()

    if not args.runs and args.runs_root:
        pattern = str(Path(args.runs_root) / "**" / "metrics.jsonl")
        run_paths = [Path(p).parent for p in glob.glob(pattern, recursive=True)]
    else:
        run_paths = [Path(p) for p in (args.runs or [])]

    if not run_paths:
        raise SystemExit("No runs specified. Pass --runs or --runs-root.")

    labels = args.labels or [load_label(p) for p in run_paths]
    if len(labels) != len(run_paths):
        raise SystemExit("Number of --labels must match number of runs.")

    metrics_to_plot = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = load_baselines(Path(args.baseline_dir)) if args.baseline_dir else {}

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
        }
    )

    for metric in metrics_to_plot:
        split = METRICS.get(metric)
        if split is None:
            print(f"Skipping unknown metric: {metric}")
            continue

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        for run_dir, label in zip(run_paths, labels):
            records = load_metrics(run_dir)
            xs: List[float] = []
            ys: List[float] = []
            for rec in records:
                if rec.get("split") != split:
                    continue
                if metric not in rec:
                    continue
                xs.append(get_x(rec, args.x_axis))
                ys.append(rec[metric])
            if xs:
                ax.plot(xs, ys, label=label)

        if baselines and metric in BASELINE_MAP:
            base_key = BASELINE_MAP[metric]
            for name, vals in baselines.items():
                if base_key in vals:
                    ax.axhline(
                        vals[base_key],
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"baseline:{name}",
                    )

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch" if args.x_axis == "epoch" else "Step")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()

        out_path = out_dir / f"{metric}.png"
        fig.savefig(out_path, dpi=300)
        fig.savefig(out_dir / f"{metric}.pdf")
        plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
