"""CUDA-accelerated training loop for Thermal-SLAM depth estimation.

Features:
  - Structured logging with Python logging module (auto-flushed)
  - VRAM monitoring every N steps
  - Per-step and per-epoch metrics with ETA
  - TensorBoard integration
  - Shared CUDA depth ops for inference (cuda_ops.py)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m thermal_slam.train_cu \
        --config configs/paper.toml
    CUDA_VISIBLE_DEVICES=2 python -m thermal_slam.train_cu \
        --config configs/paper.toml --resume ckpt.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from thermal_slam.cuda_ops import is_cuda_available
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

# ---------------------------------------------------------------------------
# Logging setup — flushed, structured, file + console
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str) -> logging.Logger:
    """Configure structured logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("thermal_slam")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (unbuffered)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (auto-flushed)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _log_vram(logger: logging.Logger, tag: str = "") -> dict:
    """Log current GPU VRAM usage. Returns usage dict."""
    if not torch.cuda.is_available():
        return {}
    used = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct = used / total * 100
    prefix = f"[{tag}] " if tag else ""
    logger.info(
        f"{prefix}VRAM: {used:.1f}GB used / {total:.1f}GB total "
        f"({pct:.0f}%) | reserved={reserved:.1f}GB"
    )
    return {"used_gb": used, "total_gb": total, "pct": pct}


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

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
            root=data_cfg.get("train_path", ""),
            height=h, width=w,
            augmentation=data_cfg.get("augmentation", True),
            max_depth=max_d, min_depth=min_d,
        )
        val_ds = ThermalDepthDataset(
            root=data_cfg.get("val_path", ""),
            height=h, width=w,
            augmentation=False, max_depth=max_d, min_depth=min_d,
        )

    bs = train_cfg.get("batch_size", 4)
    if isinstance(bs, str) and bs == "auto":
        bs = 4

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, resume_path: str | None = None) -> None:
    """CUDA-accelerated training with structured logging."""
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    es_cfg = cfg.get("early_stopping", {})
    sched_cfg = cfg.get("scheduler", {})
    log_cfg = cfg.get("logging", {})

    # Logging
    log_dir = log_cfg.get(
        "log_dir", "/mnt/artifacts-datai/logs/DEF-thermal-slam"
    )
    log = _setup_logging(log_dir)

    log.info("=" * 70)
    log.info("DEF-thermal-slam — CUDA Training")
    log.info("=" * 70)

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        _log_vram(log, "INIT")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    log.info(f"Model: {n_params / 1e6:.2f}M total, {n_train / 1e6:.2f}M trainable")

    # CUDA ops status
    cuda_avail = is_cuda_available() and device.type == "cuda"
    log.info(f"CUDA depth ops: {'available' if cuda_avail else 'not found'}")

    # Loss
    criterion = build_loss(cfg)
    log.info(
        f"Loss: SIlog={cfg.get('loss', {}).get('silog_weight', 0.9)} "
        f"SSIM={cfg.get('loss', {}).get('ssim_weight', 0.4)} "
        f"Ord={cfg.get('loss', {}).get('ordinal_weight', 0.1)} "
        f"Sm={cfg.get('loss', {}).get('smoothness_weight', 0.1)}"
    )

    # Data
    train_loader, val_loader = _build_dataloaders(cfg)
    log.info(
        f"Data: train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"batch_size={train_loader.batch_size}"
    )

    # Optimizer
    lr = train_cfg.get("learning_rate", 1e-4)
    wd = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    epochs = train_cfg.get("epochs", 100)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * sched_cfg.get("warmup_ratio", 0.05))
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )
    log.info(
        f"Scheduler: warmup={warmup_steps} steps, "
        f"total={total_steps} steps"
    )

    # Checkpoint
    ckpt_dir = ckpt_cfg.get(
        "output_dir", "/mnt/artifacts-datai/checkpoints/DEF-thermal-slam"
    )
    ckpt_mgr = CheckpointManager(
        save_dir=ckpt_dir, keep_top_k=ckpt_cfg.get("keep_top_k", 2),
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

    # TensorBoard
    tb_dir = log_cfg.get(
        "tensorboard_dir",
        "/mnt/artifacts-datai/tensorboard/DEF-thermal-slam",
    )
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # History
    history: list[dict] = []

    # Resume
    start_epoch = 0
    global_step = 0
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        log.info(f"Resumed from epoch={start_epoch} step={global_step}")

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    save_every = ckpt_cfg.get("save_every_n_steps", 500)
    log_every = 25  # log every N steps

    log.info(
        f"Training: epochs={epochs} lr={lr} precision={precision} "
        f"grad_clip={max_grad_norm}"
    )
    log.info(f"Checkpoints: {ckpt_dir}")
    log.info(f"TensorBoard: {tb_dir}")
    log.info("=" * 70)

    # --- Training loop ---
    for epoch in range(start_epoch, epochs):
        model.train()
        model.reset_state()
        epoch_loss = 0.0
        epoch_losses = {
            "silog": 0.0, "ssim": 0.0, "ordinal": 0.0, "smoothness": 0.0,
        }
        t_epoch = time.time()
        n_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            t_step = time.time()
            thermal = batch["thermal"].to(device, non_blocking=True)
            depth_gt = batch["depth"].to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=use_amp
            ):
                out = model(thermal, return_refined=True)
                loss_dict = criterion(
                    out["depth"], depth_gt, image=out.get("normalized")
                )

            loss = loss_dict["total"]

            if torch.isnan(loss) or torch.isinf(loss):
                kind = "NaN" if torch.isnan(loss) else "Inf"
                log.warning(f"Loss is {kind} at step {global_step} — skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

            # Step logging
            if global_step % log_every == 0:
                step_time = time.time() - t_step
                steps_done = batch_idx + 1
                steps_left = n_batches - steps_done
                eta_epoch = _format_eta(steps_left * step_time)
                log.info(
                    f"  step {steps_done}/{n_batches} | "
                    f"loss={loss.item():.4f} "
                    f"silog={loss_dict['silog'].item():.4f} "
                    f"ssim={loss_dict['ssim'].item():.4f} | "
                    f"lr={scheduler.get_lr():.2e} | "
                    f"ETA={eta_epoch}"
                )

            # VRAM check (first 3 steps + every 200)
            if global_step <= 3 or global_step % 200 == 0:
                _log_vram(log, f"step={global_step}")

            # TB logging
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar(
                    "train/lr", scheduler.get_lr(), global_step
                )
                for k, v in loss_dict.items():
                    if k != "total":
                        writer.add_scalar(
                            f"train/{k}", v.item(), global_step
                        )

            # Step checkpoint
            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "step": global_step, "config": cfg,
                }
                ckpt_mgr.save(
                    state, epoch_loss / (batch_idx + 1), global_step
                )
                log.info(f"  checkpoint saved at step {global_step}")

        # --- Epoch summary ---
        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_epoch

        # Validation
        model.eval()
        model.reset_state()
        val_loss = 0.0
        val_count = 0
        t_val = time.time()
        with torch.no_grad():
            for batch in val_loader:
                thermal = batch["thermal"].to(device, non_blocking=True)
                depth_gt = batch["depth"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    out = model(thermal)
                    vl = criterion(out["depth"], depth_gt)
                val_loss += vl["total"].item()
                val_count += 1
        avg_val_loss = val_loss / max(val_count, 1)
        val_time = time.time() - t_val

        # Epoch ETA
        elapsed_total = time.time() - t_epoch
        remaining_epochs = epochs - (epoch + 1)
        eta_total = _format_eta(remaining_epochs * elapsed_total)

        log.info("-" * 70)
        log.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train={avg_train_loss:.4f} val={avg_val_loss:.4f} | "
            f"lr={scheduler.get_lr():.2e} | "
            f"train={elapsed:.0f}s val={val_time:.0f}s | "
            f"ETA={eta_total}"
        )

        # Component losses
        log.info(
            f"  losses: "
            f"silog={epoch_losses['silog'] / max(n_batches, 1):.4f} "
            f"ssim={epoch_losses['ssim'] / max(n_batches, 1):.4f} "
            f"ord={epoch_losses['ordinal'] / max(n_batches, 1):.4f} "
            f"sm={epoch_losses['smoothness'] / max(n_batches, 1):.4f}"
        )

        _log_vram(log, f"epoch={epoch + 1}")
        log.info("-" * 70)

        # TB
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)

        # History
        entry = {
            "epoch": epoch + 1, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "lr": scheduler.get_lr(),
            "train_time_s": elapsed, "val_time_s": val_time,
        }
        for k in epoch_losses:
            entry[f"train_{k}"] = epoch_losses[k] / max(n_batches, 1)
        history.append(entry)

        hist_path = os.path.join(log_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # Epoch checkpoint
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "step": global_step, "config": cfg,
            "train_loss": avg_train_loss, "val_loss": avg_val_loss,
        }
        ckpt_mgr.save(state, avg_val_loss, global_step)

        # Early stopping
        if early_stop and early_stop.step(avg_val_loss):
            log.info(
                f"EARLY STOP — no improvement for "
                f"{early_stop.patience} epochs"
            )
            break

    # --- Done ---
    writer.close()
    log.info("=" * 70)
    log.info("Training complete")
    if ckpt_mgr.best_metric is not None:
        log.info(f"Best val_loss: {ckpt_mgr.best_metric:.4f}")
    log.info(f"Checkpoints: {ckpt_dir}")
    log.info(f"Logs: {log_dir}")
    log.info(f"TensorBoard: {tb_dir}")
    log.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thermal-SLAM CUDA training"
    )
    parser.add_argument(
        "--config", required=True, help="Path to TOML config"
    )
    parser.add_argument(
        "--resume", default=None, help="Checkpoint to resume from"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
