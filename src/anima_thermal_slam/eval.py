from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class DepthMetrics:
    abs_rel: float
    rmse: float
    mae: float
    a1: float
    a2: float
    a3: float


def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> DepthMetrics:
    gt = np.asarray(gt).squeeze()
    pred = np.asarray(pred).squeeze()

    valid = (gt > 0) & (pred > 0)
    if mask is not None:
        valid &= np.asarray(mask).squeeze().astype(bool)

    if valid.sum() == 0:
        return DepthMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    g = gt[valid]
    p = pred[valid]
    thresh = np.maximum(g / (p + eps), p / (g + eps))

    return DepthMetrics(
        abs_rel=float(np.mean(np.abs(g - p) / (g + eps))),
        rmse=float(np.sqrt(np.mean((g - p) ** 2))),
        mae=float(np.mean(np.abs(g - p))),
        a1=float(np.mean(thresh < 1.25)),
        a2=float(np.mean(thresh < 1.25**2)),
        a3=float(np.mean(thresh < 1.25**3)),
    )


def mean_trajectory_error(gt_xyz: np.ndarray, est_xyz: np.ndarray) -> float:
    gt = np.asarray(gt_xyz)
    est = np.asarray(est_xyz)
    if gt.shape != est.shape:
        raise ValueError("gt and est trajectories must have same shape")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError("expected trajectories with shape (N,3)")
    return float(np.linalg.norm(gt - est, axis=1).mean())


def render_eval_report(module_name: str, depth_metrics: DepthMetrics, traj_error: float | None = None) -> str:
    lines = [
        f"# {module_name} Evaluation Report",
        "",
        "## Depth Metrics",
        f"- AbsRel: {depth_metrics.abs_rel:.4f}",
        f"- RMSE: {depth_metrics.rmse:.4f}",
        f"- MAE: {depth_metrics.mae:.4f}",
        f"- a1: {depth_metrics.a1:.4f}",
        f"- a2: {depth_metrics.a2:.4f}",
        f"- a3: {depth_metrics.a3:.4f}",
    ]
    if traj_error is not None:
        lines += ["", "## Trajectory", f"- Mean Euclidean Error: {traj_error:.4f} m"]
    return "\n".join(lines) + "\n"
