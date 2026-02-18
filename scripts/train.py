from __future__ import annotations

import math
import json
from pathlib import Path
import contextlib

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from denoise_interpolate.data import FrameInterpolationDataset
from denoise_interpolate.models import build_model
from denoise_interpolate.utils import count_parameters, get_device, seed_everything, to_device


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    device = get_device(prefer_mps=cfg.device.prefer_mps)
    print(f"Using device: {device}")

    train_ds = FrameInterpolationDataset(
        cfg.data.train_manifest,
        force_rgb=cfg.data.force_rgb,
        crop_size=cfg.data.crop_size,
        crop_mode=cfg.data.crop_mode,
        hflip_prob=cfg.data.hflip_prob,
    )
    val_ds = FrameInterpolationDataset(
        cfg.data.val_manifest,
        force_rgb=cfg.data.force_rgb,
        crop_size=cfg.data.crop_size,
        crop_mode=cfg.data.val_crop_mode,
        hflip_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=bool(cfg.data.persistent_workers) if cfg.data.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        persistent_workers=bool(cfg.data.persistent_workers) if cfg.data.num_workers > 0 else False,
    )
    steps_per_epoch = max(1, len(train_loader))

    model = build_model(cfg.model).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    loss_fn = torch.nn.L1Loss()

    scheduler = None
    if getattr(cfg, "scheduler", None) and cfg.scheduler.name not in ("", "none", None):
        if cfg.scheduler.name == "cosine":
            base_lr = cfg.optimizer.lr
            min_lr = cfg.scheduler.min_lr
            warmup_epochs = int(cfg.scheduler.warmup_epochs)
            total_epochs = int(cfg.train.epochs)
            min_factor = min_lr / base_lr if base_lr > 0 else 0.0

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(max(1, warmup_epochs))
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")

    log_dir = cfg.logging.log_dir
    if log_dir is None:
        log_dir = Path(HydraConfig.get().runtime.output_dir) / "tb"
    writer = SummaryWriter(log_dir=str(log_dir))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics_path = run_dir / "metrics.jsonl"

    def append_metrics(record: dict) -> None:
        record = dict(record)
        record.setdefault("run_dir", str(run_dir))
        record.setdefault("model", str(cfg.model.name))
        record.setdefault("residual", bool(getattr(cfg.model, "residual", False)))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    use_amp = bool(cfg.train.amp) and device.type in ("cuda", "mps")
    amp_device = device.type
    amp_dtype = torch.float16 if amp_device in ("cuda", "mps") else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    amp_autocast_ok = True

    def autocast_context():
        nonlocal amp_autocast_ok
        if not use_amp or not amp_autocast_ok:
            return contextlib.nullcontext()
        try:
            return torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True)
        except Exception:
            print(f"Warning: AMP autocast not supported on {amp_device}. Disabling AMP.")
            amp_autocast_ok = False
            return contextlib.nullcontext()

    best_value = None
    best_step = None
    best_metric = str(cfg.train.best_metric)
    best_mode = "min" if best_metric in ("val_loss", "loss") else "max"

    def save_checkpoint(path: Path, epoch: int, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
        }
        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()
        torch.save(payload, path)

    global_step = 0

    def run_validation(
        step_label: int,
        desc: str,
        epoch: int,
        step_in_epoch: int,
        event: str,
    ) -> dict:
        if len(val_loader) == 0:
            print("Warning: validation loader is empty. Skipping validation metrics.")
            return {}
        model.eval()
        val_loss = 0.0
        val_samples = 0
        psnr.reset()
        ssim.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=desc):
                batch = to_device(batch, device)
                with autocast_context():
                    pred = model(batch["input"])
                    if cfg.train.clamp_output:
                        pred = pred.clamp(0, 1)
                    loss = loss_fn(pred, batch["target"])
                batch_size = pred.shape[0]
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                psnr.update(pred, batch["target"])
                ssim.update(pred, batch["target"])

        avg_val_loss = val_loss / max(1, val_samples)
        metrics = {
            "val_loss": avg_val_loss,
            "val_psnr": psnr.compute().item(),
            "val_ssim": ssim.compute().item(),
        }
        writer.add_scalar("val/loss", avg_val_loss, step_label)
        writer.add_scalar("val/psnr", metrics["val_psnr"], step_label)
        writer.add_scalar("val/ssim", metrics["val_ssim"], step_label)
        print(
            "Val "
            f"epoch={epoch+1} step={step_label} "
            f"loss={metrics['val_loss']:.6f} "
            f"psnr={metrics['val_psnr']:.4f} "
            f"ssim={metrics['val_ssim']:.4f}"
        )
        append_metrics(
            {
                "split": "val",
                "epoch": epoch,
                "step": step_label,
                "step_in_epoch": step_in_epoch,
                "steps_per_epoch": steps_per_epoch,
                "event": event,
                **metrics,
            }
        )
        model.train()
        return metrics

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.train.epochs}")):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                pred = model(batch["input"])
                if cfg.train.clamp_output:
                    pred = pred.clamp(0, 1)
                loss = loss_fn(pred, batch["target"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                optimizer.step()

            batch_size = pred.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if cfg.train.val_interval_steps and (step + 1) % cfg.train.val_interval_steps == 0:
                metrics = run_validation(
                    global_step,
                    desc=f"Val step {global_step}",
                    epoch=epoch,
                    step_in_epoch=step + 1,
                    event="interval",
                )
                if cfg.train.save_best and metrics:
                    value = metrics.get(best_metric)
                    if value is not None:
                        is_better = best_value is None or (
                            value < best_value if best_mode == "min" else value > best_value
                        )
                        if is_better:
                            best_value = value
                            best_step = global_step
                            save_checkpoint(Path("checkpoints") / "best.pt", epoch, global_step)
                            print(
                                f"New best {best_metric}: {best_value:.6f} at step {best_step}"
                            )

        avg_loss = epoch_loss / max(1, epoch_samples)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        append_metrics(
            {
                "split": "train",
                "epoch": epoch,
                "step": global_step,
                "step_in_epoch": steps_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "event": "epoch",
                "train_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if (epoch + 1) % cfg.train.val_every == 0:
            metrics = run_validation(
                global_step,
                desc=f"Val {epoch+1}/{cfg.train.epochs}",
                epoch=epoch,
                step_in_epoch=steps_per_epoch,
                event="epoch",
            )
            if cfg.train.save_best and metrics:
                value = metrics.get(best_metric)
                if value is not None:
                    is_better = best_value is None or (
                        value < best_value if best_mode == "min" else value > best_value
                    )
                    if is_better:
                        best_value = value
                        best_step = global_step
                        save_checkpoint(Path("checkpoints") / "best.pt", epoch, global_step)
                        print(f"New best {best_metric}: {best_value:.6f} at step {best_step}")

        if cfg.train.save_last:
            save_checkpoint(Path("checkpoints") / "last.pt", epoch, global_step)

        if scheduler is not None:
            scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
