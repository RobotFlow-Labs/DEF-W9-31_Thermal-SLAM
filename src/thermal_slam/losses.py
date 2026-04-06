"""Loss functions for thermal depth estimation.

Composite loss: L = 0.9*L_SIlog + 0.4*L_SSIM + 0.1*L_ord + 0.1*L_sm

References:
  - SIlog: Eigen et al. "Depth Map Prediction from a Single Image", NeurIPS 2014
  - SSIM: Godard et al. "Digging into Self-Supervised Monocular Depth Estimation", ICCV 2019
  - Ordinal: Xian et al. "Structure-Guided Ranking Loss", CVPR 2020
  - Smoothness: Xu et al. "Multi-Scale Continuous CRFs as Sequential Deep Networks for
    Monocular Depth Estimation", CVPR 2022 (edge-aware variant)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ScaleInvariantLogLoss(nn.Module):
    """Scale-invariant logarithmic loss (Eigen et al. 2014).

    L_SIlog = (1/n) * sum(d_i^2) - (lambda/n^2) * (sum(d_i))^2
    where d_i = log(y_hat_i) - log(y_i)
    """

    def __init__(self, variance_focus: float = 0.5) -> None:
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_log = torch.log(pred[mask].clamp(min=1e-6))
        target_log = torch.log(target[mask].clamp(min=1e-6))
        d = pred_log - target_log

        loss = (d**2).mean() - self.variance_focus * (d.mean() ** 2)
        return loss


class SSIMLoss(nn.Module):
    """Structural similarity loss (Godard et al. 2019).

    L_SSIM = (1 - SSIM(y_hat, y)) / 2
    """

    def __init__(self, window_size: int = 7) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = 0.01**2
        self.c2 = 0.03**2

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=pad)

        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x**2, self.window_size, stride=1, padding=pad) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y**2, self.window_size, stride=1, padding=pad) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=pad) - mu_xy

        ssim_map = ((2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x_sq + mu_y_sq + self.c1) * (sigma_x_sq + sigma_y_sq + self.c2)
        )
        return ssim_map.clamp(0, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        ssim_map = self._ssim(pred, target)
        # Apply mask
        loss = ((1 - ssim_map) / 2)
        if mask.any():
            loss = (loss * mask.float()).sum() / mask.float().sum()
        else:
            loss = loss.mean()
        return loss


class OrdinalDepthLoss(nn.Module):
    """Ordinal (ranking) depth loss (Xian et al. 2020).

    Enforces correct relative depth ordering between sampled pixel pairs.
    """

    def __init__(self, num_pairs: int = 1000, margin: float = 0.1) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        b, c, h, w = pred.shape

        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0

        for i in range(b):
            valid = mask[i, 0]
            valid_idx = valid.nonzero(as_tuple=False)
            if valid_idx.shape[0] < 2:
                continue

            n_valid = valid_idx.shape[0]
            n_pairs = min(self.num_pairs, n_valid * (n_valid - 1) // 2)

            # Random pair sampling
            idx = torch.randint(0, n_valid, (n_pairs, 2), device=pred.device)
            p1 = valid_idx[idx[:, 0]]
            p2 = valid_idx[idx[:, 1]]

            d_pred_1 = pred[i, 0, p1[:, 0], p1[:, 1]]
            d_pred_2 = pred[i, 0, p2[:, 0], p2[:, 1]]
            d_gt_1 = target[i, 0, p1[:, 0], p1[:, 1]]
            d_gt_2 = target[i, 0, p2[:, 0], p2[:, 1]]

            # Ordinal relation
            gt_order = torch.sign(d_gt_1 - d_gt_2)
            pred_diff = d_pred_1 - d_pred_2

            # Hinge loss: if gt says A > B, then pred(A) - pred(B) should be > margin
            loss_pairs = F.relu(self.margin - gt_order * pred_diff)
            total_loss = total_loss + loss_pairs.mean()
            count += 1

        if count > 0:
            total_loss = total_loss / count
        return total_loss


class EdgeAwareSmoothnessLoss(nn.Module):
    """Edge-aware depth smoothness loss.

    L_sm = |dx(depth)| * exp(-|dx(image)|) + |dy(depth)| * exp(-|dy(image)|)

    The image gradients attenuate smoothness at edges, allowing depth
    discontinuities where the image has strong edges.
    """

    def forward(
        self, pred_depth: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        assert pred_depth.shape[1] == 1, f"Expected 1-channel depth, got {pred_depth.shape[1]}"
        # Normalize depth to mean=1 for scale invariance
        mean_depth = pred_depth.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        norm_depth = pred_depth / mean_depth

        # Depth gradients
        dx_depth = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
        dy_depth = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])

        # Image gradients (mean across channels)
        dx_image = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
        dy_image = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)

        # Edge-aware weighting
        loss_x = dx_depth * torch.exp(-dx_image)
        loss_y = dy_depth * torch.exp(-dy_image)

        return loss_x.mean() + loss_y.mean()


class CompositeDepthLoss(nn.Module):
    """Composite loss as defined in the paper.

    L_total = w1*L_SIlog + w2*L_SSIM + w3*L_ord + w4*L_sm
    Default weights: [0.9, 0.4, 0.1, 0.1]
    """

    def __init__(
        self,
        silog_weight: float = 0.9,
        ssim_weight: float = 0.4,
        ordinal_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        silog_variance_focus: float = 0.5,
    ) -> None:
        super().__init__()
        self.silog_weight = silog_weight
        self.ssim_weight = ssim_weight
        self.ordinal_weight = ordinal_weight
        self.smoothness_weight = smoothness_weight

        self.silog = ScaleInvariantLogLoss(variance_focus=silog_variance_focus)
        self.ssim = SSIMLoss()
        self.ordinal = OrdinalDepthLoss()
        self.smoothness = EdgeAwareSmoothnessLoss()

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            pred_depth: (B, 1, H, W) predicted depth
            target_depth: (B, 1, H, W) ground truth depth
            image: (B, C, H, W) input image for smoothness (uses pred if None)

        Returns:
            Dictionary with 'total', 'silog', 'ssim', 'ordinal', 'smoothness'
        """
        l_silog = self.silog(pred_depth, target_depth)
        l_ssim = self.ssim(pred_depth, target_depth)
        l_ord = self.ordinal(pred_depth, target_depth)

        if image is None:
            image = pred_depth
        l_sm = self.smoothness(pred_depth, image)

        total = (
            self.silog_weight * l_silog
            + self.ssim_weight * l_ssim
            + self.ordinal_weight * l_ord
            + self.smoothness_weight * l_sm
        )

        return {
            "total": total,
            "silog": l_silog.detach(),
            "ssim": l_ssim.detach(),
            "ordinal": l_ord.detach(),
            "smoothness": l_sm.detach(),
        }


def build_loss(cfg: dict) -> CompositeDepthLoss:
    """Build composite loss from config dictionary."""
    loss_cfg = cfg.get("loss", {})
    return CompositeDepthLoss(
        silog_weight=loss_cfg.get("silog_weight", 0.9),
        ssim_weight=loss_cfg.get("ssim_weight", 0.4),
        ordinal_weight=loss_cfg.get("ordinal_weight", 0.1),
        smoothness_weight=loss_cfg.get("smoothness_weight", 0.1),
        silog_variance_focus=loss_cfg.get("silog_variance_focus", 0.5),
    )
