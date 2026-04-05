from __future__ import annotations

import cv2
import numpy as np
import torch


class ORBSLAM3ThermalAdapter:
    """Utilities for converting model outputs to ORB-SLAM3-friendly inputs."""

    def __init__(self, percent_low: float = 5.0, percent_high: float = 95.0, invert: bool = True) -> None:
        self.percent_low = percent_low
        self.percent_high = percent_high
        self.invert = invert

    def to_orb_thermal_u8(self, refined_01: np.ndarray | torch.Tensor) -> np.ndarray:
        arr = refined_01.detach().cpu().numpy() if isinstance(refined_01, torch.Tensor) else np.asarray(refined_01)
        arr = arr.squeeze().astype(np.float32)
        p_lo, p_hi = np.percentile(arr, [self.percent_low, self.percent_high])
        arr = np.clip((arr - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
        if self.invert:
            arr = 1.0 - arr
        return (arr * 255.0 + 0.5).astype(np.uint8)

    def depth_to_float32(self, depth: np.ndarray | torch.Tensor) -> np.ndarray:
        arr = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
        return arr.squeeze().astype(np.float32)

    def debug_colormap(self, thermal_u8: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(thermal_u8, cv2.COLORMAP_INFERNO)
