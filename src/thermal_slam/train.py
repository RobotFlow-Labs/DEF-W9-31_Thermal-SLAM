"""Training loop for Thermal-SLAM depth estimation.

Usage:
    python -m thermal_slam.train --config configs/paper.toml
    python -m thermal_slam.train --config configs/paper.toml --resume /path/to/ckpt.pth
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from thermal_slam.dataset import ThermalDepthDataset, VIVIDPlusPlusDataset
from thermal_slam.losses import build_loss
from thermal_slam.model import build_model
from thermal_slam.utils import (
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    load_config,
    set_seed,
)


def _build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    h = model_cfg.get("input_height", 256)
    w = model_cfg.get("input_width", 320)
    max_d = model_cfg.get("max_depth", 10.0)
    min_d = model_cfg.get("min_depth", 0.1)

    dataset_format = data_cfg.get("dataset_format", "vivid_pp")

    if dataset_format == "vivid_pp":
        root = data_cfg.get("root", "/mnt/forge-data/datasets/vivid_plus_plus")
        train_ds = VIVIDPlusPlusDataset(
            root=root, split="train", height=h, width=w,
            augmentation=data_cfg.get("augmentation", True),
            max_depth=max_d, min_depth=min_d,
        )
        val_ds = VIVIDPlusPlusDataset(
            root=root, split="val", height=h, width=w,
            augmentation=False, max_depth=max_d, min_depth=min_d,
        )
    else:
        train_ds = ThermalDepthDataset(
            root=data_cfg.get("train_path", "/tmp/thermal_slam_data/train"),
            height=h, width=w,
            augmentation=data_cfg.get("augmentation", True),
            max_depth=max_d, min_depth=min_d,
        )
        val_ds = ThermalDepthDataset(
            root=data_cfg.get("val_path", "/tmp/thermal_slam_data/val"),
            height=h, width=w,
            augmentation=False, max_depth=max_d, min_depth=min_d,
        )

    bs = train_cfg.get("batch_size", 4)
    if isinstance(bs, str) and bs == "auto":
        bs = 4  # placeholder — replaced by batch finder at runtime

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader


def _build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {})
    opt_name = train_cfg.get("optimizer", "adamw")
    lr = train_cfg.get("learning_rate", 1e-4)
    wd = train_cfg.get("weight_decay", 0.01)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def train(cfg: dict, resume_path: str | None = None) -> None:
    """Main training function."""
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    es_cfg = cfg.get("early_stopping", {})
    sched_cfg = cfg.get("scheduler", {})
    log_cfg = cfg.get("logging", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {n_params / 1e6:.1f}M parameters")

    # Loss
    criterion = build_loss(cfg)

    # Data
    train_loader, val_loader = _build_dataloaders(cfg)
    print(f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # Optimizer
    optimizer = _build_optimizer(model, cfg)

    # Scheduler
    epochs = train_cfg.get("epochs", 100)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * sched_cfg.get("warmup_ratio", 0.05))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # Checkpoint manager
    ckpt_dir = ckpt_cfg.get("output_dir", "/mnt/artifacts-datai/checkpoints/DEF-thermal-slam")
    ckpt_mgr = CheckpointManager(
        save_dir=ckpt_dir,
        keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_loss"),
        mode=ckpt_cfg.get("mode", "min"),
    )

    # Early stopping
    early_stop = None
    if es_cfg.get("enabled", True):
        early_stop = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            min_delta=es_cfg.get("min_delta", 0.001),
        )

    # AMP
    precision = train_cfg.get("precision", "bf16")
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    # Logging
    log_dir = log_cfg.get("log_dir", "/mnt/artifacts-datai/logs/DEF-thermal-slam")
    os.makedirs(log_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    global_step = 0
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        print(f"[RESUME] epoch={start_epoch} step={global_step}")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    save_every = ckpt_cfg.get("save_every_n_steps", 500)

    print(f"[TRAIN] {epochs} epochs, lr={train_cfg.get('learning_rate', 1e-4)}, "
          f"precision={precision}")

    for epoch in range(start_epoch, epochs):
        model.train()
        model.reset_state()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            thermal = batch["thermal"].to(device)
            depth_gt = batch["depth"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(thermal, return_refined=True)
                loss_dict = criterion(out["depth"], depth_gt, image=out.get("normalized"))

            loss = loss_dict["total"]

            # NaN check
            if torch.isnan(loss):
                print("[FATAL] Loss is NaN — stopping training")
                print("[FIX] Reduce lr, check data for corrupt samples")
                return

            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            # Step-based checkpoint
            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": cfg,
                }
                ckpt_mgr.save(state, epoch_loss / (batch_idx + 1), global_step)

        # Epoch stats
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        model.reset_state()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                thermal = batch["thermal"].to(device)
                depth_gt = batch["depth"].to(device)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(thermal)
                    loss_dict = criterion(out["depth"], depth_gt)
                val_loss += loss_dict["total"].item()
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"lr={scheduler.get_lr():.2e} "
            f"time={elapsed:.1f}s"
        )

        # Epoch checkpoint
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "config": cfg,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        ckpt_mgr.save(state, avg_val_loss, global_step)

        # Early stopping
        if early_stop and early_stop.step(avg_val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs")
            break

    print("[DONE] Training complete")
    print(f"[BEST] val_loss={ckpt_mgr.best_metric:.4f}")
    print(f"[CKPT] {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal-SLAM training")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
