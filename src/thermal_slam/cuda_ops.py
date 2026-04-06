"""CUDA-accelerated operations for Thermal-SLAM.

Uses shared infrastructure kernels from /mnt/forge-data/shared_infra/cuda_extensions/
JIT-compiled with torch.utils.cpp_extension.load()
"""

from __future__ import annotations

import os

import torch

_depth_ops = None


def _load_depth_ops():
    """Lazy-load depth estimation CUDA ops."""
    global _depth_ops
    if _depth_ops is not None:
        return _depth_ops

    from torch.utils.cpp_extension import load

    cu_path = "/mnt/forge-data/shared_infra/cuda_extensions/depth_estimation_ops/depth_ops.cu"
    if not os.path.exists(cu_path):
        raise FileNotFoundError(f"Shared CUDA kernel not found: {cu_path}")

    _depth_ops = load(
        name="depth_estimation_ops",
        sources=[cu_path],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    )
    return _depth_ops


def cuda_si_log_loss(
    pred: torch.Tensor, gt: torch.Tensor, lambda_si: float = 0.5
) -> torch.Tensor:
    """CUDA-accelerated scale-invariant log depth loss.

    Uses fused kernel — 1 kernel launch vs 5+ PyTorch ops.
    """
    ops = _load_depth_ops()
    # Flatten + create mask for valid pixels
    pred_flat = pred.contiguous().view(-1).float()
    gt_flat = gt.contiguous().view(-1).float()
    mask = (gt_flat > 0).float()
    return ops.fused_si_log_loss(pred_flat, gt_flat, mask, lambda_si)


def cuda_depth_gradient_loss(
    depth: torch.Tensor, image: torch.Tensor
) -> torch.Tensor:
    """CUDA-accelerated edge-aware depth gradient loss.

    Uses fused kernel — ~5x faster than PyTorch equivalent.
    """
    ops = _load_depth_ops()
    # Expects (H, W) depth and (C, H, W) image — process per-batch
    b = depth.shape[0]
    total = torch.tensor(0.0, device=depth.device)
    for i in range(b):
        d = depth[i, 0].contiguous().float()  # (H, W)
        img = image[i].contiguous().float()  # (C, H, W)
        loss_map = ops.fused_depth_gradient_loss(d, img)
        total = total + loss_map.mean()
    return total / b


def cuda_depth_to_pointcloud(
    depth: torch.Tensor, fx: float, fy: float, cx: float, cy: float
) -> torch.Tensor:
    """Convert depth map to 3D point cloud using CUDA kernel.

    Args:
        depth: (H, W) depth map
        fx, fy, cx, cy: camera intrinsics

    Returns:
        points: (H*W, 3) xyz point cloud
    """
    ops = _load_depth_ops()
    return ops.fused_depth_to_pointcloud(depth.contiguous().float(), fx, fy, cx, cy)


def is_cuda_available() -> bool:
    """Check if CUDA depth ops are available."""
    try:
        _load_depth_ops()
        return True
    except Exception:
        return False
