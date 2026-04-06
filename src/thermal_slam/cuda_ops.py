"""CUDA-accelerated operations for Thermal-SLAM.

Two sets of shared CUDA kernels:

1. **thermal_depth_ops** (DIFFERENTIABLE — for training):
   - silog_loss: Scale-invariant log depth loss with autograd
   - depth_edge_smooth: Edge-aware smoothness with autograd
   - thermal_normalize: Per-sample min-max normalization
   Saved at: /mnt/forge-data/shared_infra/cuda_extensions/thermal_depth_ops/

2. **depth_estimation_ops** (NON-DIFFERENTIABLE — for inference):
   - fused_depth_to_pointcloud: Depth map → 3D point cloud
   - fused_disparity_to_depth: Disparity → depth
   Saved at: /mnt/forge-data/shared_infra/cuda_extensions/depth_estimation_ops/
"""

from __future__ import annotations

import os

import torch

_thermal_ops = None
_depth_ops = None

THERMAL_OPS_PATH = (
    "/mnt/forge-data/shared_infra/cuda_extensions/"
    "thermal_depth_ops/thermal_depth_ops.cu"
)
DEPTH_OPS_PATH = (
    "/mnt/forge-data/shared_infra/cuda_extensions/"
    "depth_estimation_ops/depth_ops.cu"
)
CUDA_FLAGS = [
    "-O3", "--use_fast_math",
    "-gencode=arch=compute_89,code=sm_89",
]


def _load_thermal_ops():
    """Load differentiable thermal depth CUDA ops (for training)."""
    global _thermal_ops
    if _thermal_ops is not None:
        return _thermal_ops

    from torch.utils.cpp_extension import load

    if not os.path.exists(THERMAL_OPS_PATH):
        raise FileNotFoundError(
            f"Shared CUDA kernel not found: {THERMAL_OPS_PATH}"
        )

    _thermal_ops = load(
        name="thermal_depth_ops",
        sources=[THERMAL_OPS_PATH],
        verbose=False,
        extra_cuda_cflags=CUDA_FLAGS,
    )
    return _thermal_ops


def _load_depth_ops():
    """Load non-differentiable depth estimation ops (for inference)."""
    global _depth_ops
    if _depth_ops is not None:
        return _depth_ops

    from torch.utils.cpp_extension import load

    if not os.path.exists(DEPTH_OPS_PATH):
        raise FileNotFoundError(
            f"Shared CUDA kernel not found: {DEPTH_OPS_PATH}"
        )

    _depth_ops = load(
        name="depth_estimation_ops",
        sources=[DEPTH_OPS_PATH],
        verbose=False,
        extra_cuda_cflags=CUDA_FLAGS,
    )
    return _depth_ops


# ---------------------------------------------------------------------------
# Differentiable ops (training)
# ---------------------------------------------------------------------------

def cuda_silog_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_si: float = 0.5,
) -> torch.Tensor:
    """CUDA-accelerated SILog loss WITH autograd support.

    Uses custom torch::autograd::Function with analytic backward.
    """
    ops = _load_thermal_ops()
    mask = (gt > 0).float()
    return ops.silog_loss(pred, gt, mask, lambda_si)


def cuda_depth_edge_smooth(
    depth: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """CUDA-accelerated edge-aware smoothness WITH autograd support.

    Uses custom torch::autograd::Function with analytic backward.
    """
    ops = _load_thermal_ops()
    return ops.depth_edge_smooth(depth, image)


def cuda_thermal_normalize(
    raw_thermal: torch.Tensor,
) -> torch.Tensor:
    """CUDA-accelerated per-sample min-max normalization.

    Args:
        raw_thermal: (B, 1, H, W) raw 16-bit thermal values.

    Returns:
        Normalized tensor in [0, 1].
    """
    ops = _load_thermal_ops()
    return ops.thermal_normalize(raw_thermal)


# ---------------------------------------------------------------------------
# Non-differentiable ops (inference)
# ---------------------------------------------------------------------------

def cuda_depth_to_pointcloud(
    depth: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
) -> torch.Tensor:
    """Convert depth map to 3D point cloud using CUDA kernel.

    Args:
        depth: (H, W) depth map
        fx, fy, cx, cy: camera intrinsics

    Returns:
        points: (H*W, 3) xyz point cloud
    """
    ops = _load_depth_ops()
    return ops.fused_depth_to_pointcloud(
        depth.contiguous().float(), fx, fy, cx, cy
    )


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Check if differentiable thermal depth CUDA ops are available."""
    try:
        _load_thermal_ops()
        return True
    except Exception:
        return False


def is_inference_ops_available() -> bool:
    """Check if inference depth ops are available."""
    try:
        _load_depth_ops()
        return True
    except Exception:
        return False
