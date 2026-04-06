"""Utilities: config loading, seeding, checkpoint management, export helpers."""

from __future__ import annotations

import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a TOML config file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Warmup + cosine decay LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save top-K checkpoints ranked by metric, plus a best.pth."""

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 2,
        metric: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict,
        metric_value: float,
        step: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort: best first
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Prune
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            if old_path.exists():
                old_path.unlink()

        # Save best separately
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")

        return path

    @property
    def best_metric(self) -> float | None:
        if not self.history:
            return None
        return self.history[0][0]


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping monitor."""

    def __init__(
        self, patience: int = 20, min_delta: float = 0.001, mode: str = "min"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Export Helpers
# ---------------------------------------------------------------------------

def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 17,
) -> str:
    """Export model to ONNX format."""
    model.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["thermal_input"],
        output_names=["depth"],
        dynamic_axes={
            "thermal_input": {0: "batch"},
            "depth": {0: "batch"},
        },
    )
    return output_path


def export_safetensors(model: torch.nn.Module, output_path: str) -> str:
    """Export model weights to safetensors format."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(model.state_dict(), output_path)
    return output_path
