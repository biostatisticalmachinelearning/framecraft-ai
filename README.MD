# Denoise-Interpolate

This project builds a machine learning model that takes old movies as input and interpolates video frames to enhance frame-rate. By dropping selected frames and using adjacent frames to predict the missing frame, we train a model that also denoises as a side effect (noise is unpredictable, similar to a denoising autoencoder).

The repo includes:
- A data pipeline to create frame triplet manifests.
- PyTorch model training (baseline U-Net) and evaluation.
- Simple inference scripting for clip interpolation.

## Repo Structure
- `src/denoise_interpolate/` Core library code (models, data, utils)
- `scripts/` Training, evaluation, inference, and data manifest creation
- `configs/` Hydra configuration files
- `data/` Local datasets (ignored by git)
- `outputs/` Hydra outputs (ignored by git)
- `docs/` Project notes

## Setup
1. Create a Python 3.9+ environment.
2. Install dependencies:

```bash
python -m pip install -e .[dev,torch]
```

### Cross-Platform Bootstrap (Recommended)

```bash
python scripts/bootstrap_env.py --dev
```

Examples:
- macOS (Apple Silicon / MPS): `python scripts/bootstrap_env.py --dev`
- Linux + CUDA (e.g., RTX 2070): `python scripts/bootstrap_env.py --dev --cuda`
- CPU-only: `python scripts/bootstrap_env.py --dev --cpu`

If you need a different CUDA version, set `--torch-index` to the appropriate PyTorch index URL.

### FFmpeg
FFmpeg is required for frame extraction.
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: install from https://ffmpeg.org/download.html and add to PATH

## Data Preparation
1. (Optional) Expand the curated list to ~100 items (adds Prelinger Archive entries):

```bash
python scripts/refresh_sources.py --total-target 100
```

2. Download curated sources (defaults to 5 items from `configs/sources.yaml`):

```bash
python scripts/download_sources.py --max-items 5 --max-total-gb 20 --workers 2
```

3. Extract frames per movie into subfolders (24 fps expected):

```bash
python scripts/extract_frames.py --input-dir data/raw --fps 24 --recursive
```

If you intentionally want a different frame rate for ablations, pass `--allow-non-24`.

By default, frames are extracted at the **original resolution**. To downscale, pass `--scale`, e.g.:

```bash
python scripts/extract_frames.py --input-dir data/raw --fps 24 --recursive --scale \"scale=-2:360\"
```

Extraction is lossless by default (PNG only).

Tips for faster extraction:
- Extract a quick subset: `--start 0 --duration 60`
- Parallelize across movies: `--workers 2`
- Show progress bars: `--progress`

4. Build train/val/test manifests (default: contiguous within-movie splits):

```bash
python scripts/prepare_data.py --frames-root data/frames
```

This produces JSONL manifests in `data/manifests/` with triplets: `prev`, `next`, `target`, `movie_id`.

If you want **strict movie-level splits** (recommended for final results), use:

```bash
python scripts/prepare_data.py --frames-root data/frames --movie-level
```

Movie-level splits require multiple movies to avoid empty val/test splits.

## Training
Default training uses the baseline U-Net. Crops are 256x256 by default.
Training uses random horizontal flips (p=0.5) and validation uses center crops.
By default, we use cosine LR schedule with warmup, AMP (when available), gradient clipping,
and save `checkpoints/best.pt` based on validation L1 loss inside each run directory.

You can enable a residual skip from the input blend (helps frame replication):

```bash
python scripts/train.py model.name=resunet
```

Or keep the base model and toggle residual mode explicitly:

```bash
python scripts/train.py model.name=unet model.residual=true
```

For the ViT backbone:

```bash
python scripts/train.py model.name=resvit
```

The ViT decoder is configurable via `model.vit.decoder_*` (channels, depth, upsample mode).

```bash
python scripts/train.py
```

To override crop size (e.g., 128 or 512):

```bash
python scripts/train.py data.crop_size=128
```

For long epochs, you can run validation mid-epoch every N training steps:

```bash
python scripts/train.py train.val_interval_steps=500
```

Hydra outputs (including TensorBoard logs) are written under `outputs/` by default.
Each run directory includes `metrics.jsonl` (train/val metrics per epoch/interval) and
`checkpoints/` (best/last).

## Plotting Training Curves
Training writes `metrics.jsonl` to each run directory. You can compare runs and add baseline
references with:

```bash
python scripts/plot_training.py \\
  --runs outputs/2025-01-01/12-00-00 outputs/2025-01-02/13-30-00 \\
  --out-dir outputs/plots \\
  --baseline-dir outputs/baselines
```

Plots are saved as PNG and PDF in `outputs/plots/`.

## Evaluation

```bash
python scripts/eval.py --manifest data/manifests/test.jsonl --checkpoint checkpoints/last.pt
```

## Evaluation Report

Creates a markdown report, metrics JSON, and a visualization grid.

```bash
python scripts/report_eval.py \\
  --manifest data/manifests/test.jsonl \\
  --checkpoint checkpoints/last.pt \\
  --out-dir outputs/report
```

Register the run for comparisons:

```bash
python scripts/report_eval.py \\
  --manifest data/manifests/test.jsonl \\
  --checkpoint checkpoints/last.pt \\
  --out-dir outputs/report \\
  --register --tag unet_baseline
```

## Dataset QA

Checks registry integrity and (optionally) samples frames for basic statistics.

```bash
python scripts/dataset_qa.py \\
  --registry data/registry.jsonl \\
  --frames-root data/frames \\
  --frames-sample 16
```

## ViT Baseline

Train a Vision Transformer baseline by switching the model name:

```bash
python scripts/train.py model.name=vit model.vit.patch_size=16 model.vit.embed_dim=256
```

Notes:
- Input sizes are padded to be divisible by `patch_size`, then cropped back to original size.
- ViT models are more memory intensive than U-Nets; start with smaller `embed_dim` or `depth` if needed.

## Interpolation Baselines (Sanity Checks)

Evaluate classic interpolation baselines against the same manifest:

```bash
python scripts/baseline_eval.py --manifest data/manifests/test.jsonl --methods hold_prev,hold_next,blend,flow
```

Note: `flow` uses CPU optical flow (OpenCV Farneback) and can be slow on large frames.

Generate a single interpolated frame with a baseline:

```bash
python scripts/baseline_interpolate.py --frame-a A.png --frame-b B.png --method blend --output outputs/blend.png
```

Register baseline runs to compare against learned models:

```bash
python scripts/baseline_eval.py \\
  --manifest data/manifests/test.jsonl \\
  --methods blend,flow \\
  --register --tag baseline
```

## Model Registry

Register an existing report:

```bash
python scripts/register_run.py \\
  --metrics outputs/report/metrics.json \\
  --checkpoint checkpoints/last.pt \\
  --manifest data/manifests/test.jsonl \\
  --report outputs/report/report.md \\
  --tag unet_baseline
```

Compare runs (top 10 by PSNR by default):

```bash
python scripts/compare_runs.py --metric psnr
```

## Inference

```bash
python scripts/infer.py \
  --frame-a path/to/frame_0001.png \
  --frame-b path/to/frame_0003.png \
  --output outputs/pred_0002.png \
  --checkpoint checkpoints/last.pt
```

## Video Interpolation + Denoising

Denoise originals (predict each frame from its neighbors), then insert mid-frames to double FPS:

```bash
python scripts/interpolate_video.py \
  --frames-dir data/frames/your_movie \
  --checkpoint checkpoints/last.pt \
  --output-dir outputs/interpolated \
  --output-video outputs/interpolated.mp4
```

Only insert interleaved frames (keep originals):

```bash
python scripts/interpolate_video.py \
  --frames-dir data/frames/your_movie \
  --checkpoint checkpoints/last.pt \
  --output-dir outputs/interpolated \
  --output-video outputs/interpolated.mp4 \
  --no-denoise-original
```

## Notes
- Use movie-level splits to avoid data leakage.
- For grayscale films, enable `force_rgb` in configs or pass `--force-rgb` to scripts.
- If you have limited local disk keep `--max-total-gb` low.

## Suggested Public-Domain Sources (Initial Set)
- LOC National Screening Room: Duck and Cover (1951) — https://www.loc.gov/item/2022604365/
- LOC National Screening Room: The Hitch-Hiker (1953) — https://www.loc.gov/item/2023602025/
- LOC Collection: The Great Train Robbery (1903) — https://www.loc.gov/item/00694220/
- Prelinger Archive (IA): What About Juvenile Delinquency (1955) — https://archive.org/details/WhatAbou1955
- Prelinger Archive (IA): Wartime Nutrition (1943) — https://archive.org/details/WartimeN1943
 
Always verify the rights/usage statement on each item page before use.
