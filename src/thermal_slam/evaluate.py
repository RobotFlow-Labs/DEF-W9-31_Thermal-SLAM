"""Evaluation metrics and script for thermal depth estimation.

Metrics (standard monocular depth):
  - AbsRel, SqRel, RMSE, RMSE_log
  - delta < 1.25, 1.25^2, 1.25^3

Usage:
    python -m thermal_slam.evaluate --config configs/paper.toml --checkpoint best.pth
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from thermal_slam.dataset import ThermalDepthDataset
from thermal_slam.model import build_model
from thermal_slam.utils import load_config

# ---------------------------------------------------------------------------
# Depth Metrics
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: np.ndarray, target: np.ndarray, min_depth: float = 0.1, max_depth: float = 10.0
) -> dict[str, float]:
    """Compute standard monocular depth estimation metrics.

    Args:
        pred: Predicted depth (N,) or (H, W)
        target: Ground truth depth (N,) or (H, W)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth

    Returns:
        Dictionary with AbsRel, SqRel, RMSE, RMSE_log, a1, a2, a3
    """
    pred = pred.flatten()
    target = target.flatten()

    # Valid mask
    mask = (target > min_depth) & (target < max_depth) & (pred > min_depth) & (pred < max_depth)
    if mask.sum() == 0:
        return {
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "rmse": float("nan"),
            "rmse_log": float("nan"),
            "a1": float("nan"),
            "a2": float("nan"),
            "a3": float("nan"),
        }

    p = pred[mask]
    t = target[mask]

    # Error metrics
    abs_rel = float(np.mean(np.abs(p - t) / t))
    sq_rel = float(np.mean(((p - t) ** 2) / t))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(t)) ** 2)))

    # Threshold accuracy
    ratio = np.maximum(p / t, t / p)
    a1 = float(np.mean(ratio < 1.25))
    a2 = float(np.mean(ratio < 1.25**2))
    a3 = float(np.mean(ratio < 1.25**3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
    }


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    cfg: dict,
    checkpoint_path: str,
    split: str = "test",
) -> dict[str, float]:
    """Run evaluation on a dataset split.

    Args:
        cfg: Config dictionary
        checkpoint_path: Path to model checkpoint
        split: "test" or "val"

    Returns:
        Aggregated metrics dictionary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Build model and load weights
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Dataset
    h = model_cfg.get("input_height", 512)
    w = model_cfg.get("input_width", 640)
    max_d = model_cfg.get("max_depth", 10.0)
    min_d = model_cfg.get("min_depth", 0.1)

    data_path = data_cfg.get(f"{split}_path", data_cfg.get("test_path", ""))
    dataset = ThermalDepthDataset(
        root=data_path, height=h, width=w, augmentation=False,
        max_depth=max_d, min_depth=min_d,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_metrics: list[dict[str, float]] = []
    model.reset_state()

    for batch in loader:
        thermal = batch["thermal"].to(device)
        depth_gt = batch["depth"]

        out = model(thermal)
        pred_depth = out["depth"].cpu().numpy()
        gt_depth = depth_gt.numpy()

        metrics = compute_depth_metrics(
            pred_depth, gt_depth, min_depth=min_d, max_depth=max_d
        )
        all_metrics.append(metrics)

    # Aggregate
    if not all_metrics:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0,
                "a1": 0.0, "a2": 0.0, "a3": 0.0}

    agg: dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
        agg[key] = float(np.mean(vals)) if vals else float("nan")

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal-SLAM evaluation")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate(cfg, args.checkpoint, split=args.split)

    # Print
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>10s}: {v:.4f}")
    print("=" * 50)

    # Save
    output = args.output
    if output is None:
        report_dir = "/mnt/artifacts-datai/reports/DEF-thermal-slam"
        os.makedirs(report_dir, exist_ok=True)
        output = os.path.join(report_dir, f"eval_{args.split}.json")

    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
