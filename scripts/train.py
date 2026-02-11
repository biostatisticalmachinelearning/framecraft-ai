from __future__ import annotations

from pathlib import Path

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
        cfg.data.train_manifest, force_rgb=cfg.data.force_rgb
    )
    val_ds = FrameInterpolationDataset(
        cfg.data.val_manifest, force_rgb=cfg.data.force_rgb
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
    )

    model = build_model(cfg.model).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    loss_fn = torch.nn.L1Loss()

    log_dir = cfg.logging.log_dir
    if log_dir is None:
        log_dir = Path(HydraConfig.get().runtime.output_dir) / "tb"
    writer = SummaryWriter(log_dir=str(log_dir))

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    global_step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch+1}/{cfg.train.epochs}"):
            batch = to_device(batch, device)
            pred = model(batch["input"])
            loss = loss_fn(pred, batch["target"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / max(1, len(train_loader))
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        if (epoch + 1) % cfg.train.val_every == 0:
            model.eval()
            val_loss = 0.0
            psnr.reset()
            ssim.reset()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch+1}/{cfg.train.epochs}"):
                    batch = to_device(batch, device)
                    pred = model(batch["input"])
                    loss = loss_fn(pred, batch["target"])
                    val_loss += loss.item()
                    psnr.update(pred, batch["target"])
                    ssim.update(pred, batch["target"])

            avg_val_loss = val_loss / max(1, len(val_loader))
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("val/psnr", psnr.compute().item(), epoch)
            writer.add_scalar("val/ssim", ssim.compute().item(), epoch)

        if cfg.train.save_last:
            ckpt_path = Path("checkpoints") / "last.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )

    writer.close()


if __name__ == "__main__":
    main()
