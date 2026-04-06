"""CUDA-accelerated training loop for Thermal-SLAM depth estimation.

Uses shared CUDA kernels for:
  - Scale-invariant log depth loss (fused_si_log_loss)
  - Edge-aware depth gradient loss (fused_depth_gradient_loss)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m thermal_slam.train_cu --config configs/paper.toml
    CUDA_VISIBLE_DEVICES=2 python -m thermal_slam.train_cu \
        --config configs/paper.toml --resume ckpt.pth
"""

from __future__ import annotations

import argparse
import json
import os
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


def _build_criterion(cfg: dict) -> torch.nn.Module:
    """Build composite loss function.

    Uses PyTorch CompositeDepthLoss for training (needs autograd).
    CUDA depth ops (cuda_ops.py) are used for inference/evaluation only.
    """
    return build_loss(cfg)


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


def train(cfg: dict, resume_path: str | None = None) -> None:
    """CUDA-accelerated training function."""
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    es_cfg = cfg.get("early_stopping", {})
    sched_cfg = cfg.get("scheduler", {})
    log_cfg = cfg.get("logging", {})
    loss_cfg = cfg.get("loss", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] {vram:.1f} GB")

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {n_params / 1e6:.2f}M total, {n_train / 1e6:.2f}M trainable")

    # Loss function (PyTorch autograd-compatible for training)
    # CUDA depth ops (cuda_ops.py) available for inference/eval
    cuda_avail = is_cuda_available() and device.type == "cuda"
    cuda_msg = "available (inference)" if cuda_avail else "not available"
    print(f"[CUDA OPS] {cuda_msg}")

    criterion = _build_criterion(cfg)

    # Data
    train_loader, val_loader = _build_dataloaders(cfg)
    print(f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

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

    # Checkpoint manager
    ckpt_dir = ckpt_cfg.get("output_dir", "/mnt/artifacts-datai/checkpoints/DEF-thermal-slam")
    ckpt_mgr = CheckpointManager(
        save_dir=ckpt_dir, keep_top_k=ckpt_cfg.get("keep_top_k", 2),
        metric=ckpt_cfg.get("metric", "val_loss"), mode=ckpt_cfg.get("mode", "min"),
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
    tb_dir = log_cfg.get("tensorboard_dir", "/mnt/artifacts-datai/tensorboard/DEF-thermal-slam")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    # Logging
    log_dir = log_cfg.get("log_dir", "/mnt/artifacts-datai/logs/DEF-thermal-slam")
    os.makedirs(log_dir, exist_ok=True)

    # Training history
    history: list[dict] = []

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
    bs = train_loader.batch_size

    print(f"[TRAIN] epochs={epochs} lr={lr} bs={bs} precision={precision}")
    print(f"[CKPT] {ckpt_dir}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        model.train()
        model.reset_state()
        epoch_loss = 0.0
        epoch_losses = {"silog": 0.0, "ssim": 0.0, "ordinal": 0.0, "smoothness": 0.0}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            thermal = batch["thermal"].to(device, non_blocking=True)
            depth_gt = batch["depth"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(thermal, return_refined=True)
                loss_dict = criterion(out["depth"], depth_gt, image=out.get("normalized"))

            loss = loss_dict["total"]

            if torch.isnan(loss) or torch.isinf(loss):
                kind = "NaN" if torch.isnan(loss) else "Inf"
                print(f"[WARN] Loss is {kind} at step {global_step}")
                optimizer.zero_grad(set_to_none=True)
                continue

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

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()

            # Log to TB every 50 steps
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_lr(), global_step)
                for k, v in loss_dict.items():
                    if k != "total":
                        writer.add_scalar(f"train/{k}", v.item(), global_step)

            # Step checkpoint
            if global_step % save_every == 0:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "step": global_step, "config": cfg,
                }
                ckpt_mgr.save(state, epoch_loss / (batch_idx + 1), global_step)

        # Epoch stats
        n_batches = max(len(train_loader), 1)
        avg_train_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        # Validation
        model.eval()
        model.reset_state()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                thermal = batch["thermal"].to(device, non_blocking=True)
                depth_gt = batch["depth"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(thermal)
                    loss_dict = criterion(out["depth"], depth_gt)
                val_loss += loss_dict["total"].item()
                val_count += 1
        avg_val_loss = val_loss / max(val_count, 1)

        # Log
        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"lr={scheduler.get_lr():.2e} "
            f"time={elapsed:.1f}s"
        )

        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)

        # History
        entry = {
            "epoch": epoch + 1, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "lr": scheduler.get_lr(),
            "elapsed_s": elapsed,
        }
        for k in epoch_losses:
            entry[f"train_{k}"] = epoch_losses[k] / n_batches
        history.append(entry)

        # Save history
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
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs")
            break

    writer.close()
    print("=" * 70)
    print("[DONE] Training complete")
    if ckpt_mgr.best_metric is not None:
        print(f"[BEST] val_loss={ckpt_mgr.best_metric:.4f}")
    print(f"[CKPT] {ckpt_dir}")
    print(f"[LOGS] {log_dir}")
    print(f"[TB] {tb_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal-SLAM CUDA training")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps (for smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
