from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def silog_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, lam: float = 0.85) -> torch.Tensor:
    pred = pred.clamp_min(1e-6)
    gt = gt.clamp_min(1e-6)
    if mask is not None:
        pred = pred[mask > 0]
        gt = gt[mask > 0]
    d = pred.log() - gt.log()
    return torch.sqrt((d.pow(2).mean() - lam * d.mean().pow(2)).clamp_min(0.0)) * 10.0


def _ssim_map(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = n / (d + 1e-6)
    return torch.clamp((1.0 - ssim) / 2.0, 0.0, 1.0)


def ssim_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    m = _ssim_map(pred, gt)
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return _masked_mean(m, mask)


def smoothness_loss(depth: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]

    if image is not None:
        if image.dim() == 3:
            image = image.unsqueeze(1)
        gx = image[:, :, :, 1:] - image[:, :, :, :-1]
        gy = image[:, :, 1:, :] - image[:, :, :-1, :]
        wx = torch.exp(-torch.mean(torch.abs(gx), dim=1, keepdim=True))
        wy = torch.exp(-torch.mean(torch.abs(gy), dim=1, keepdim=True))
        dx = dx * wx
        dy = dy * wy

    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def ordinal_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 0.15) -> torch.Tensor:
    diff = pred - gt
    if mask is not None:
        diff = diff[mask > 0]
    if diff.numel() == 0:
        return pred.new_tensor(0.0)
    return ((diff.abs() > tau).float() * (diff < 0).float()).mean()


def temporal_consistency_loss(pred_seq: torch.Tensor, mask_seq: torch.Tensor | None = None, mode: str = "l1") -> torch.Tensor:
    """pred_seq: (B,T,1,H,W) or (B,T,H,W)."""
    if pred_seq.dim() == 5:
        pred_seq = pred_seq.squeeze(2)

    if pred_seq.shape[1] <= 1:
        return pred_seq.new_tensor(0.0)

    terms = []
    for t in range(1, pred_seq.shape[1]):
        a = pred_seq[:, t]
        b = pred_seq[:, t - 1]
        if mode == "si":
            d = (a.clamp_min(1e-6).log() - b.clamp_min(1e-6).log()).abs()
        else:
            d = (a - b).abs()

        if mask_seq is not None:
            m = (mask_seq[:, t] > 0).float() * (mask_seq[:, t - 1] > 0).float()
            terms.append(_masked_mean(d, m))
        else:
            terms.append(d.mean())

    return torch.stack(terms).mean()


def combined_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    image: torch.Tensor | None = None,
    w_silog: float = 0.9,
    w_ssim: float = 0.4,
    w_order: float = 0.1,
    w_smooth: float = 0.1,
    use_silog: bool = True,
) -> torch.Tensor:
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if gt.dim() == 4:
        gt = gt.squeeze(1)
    if mask is not None and mask.dim() == 4:
        mask = mask.squeeze(1)

    l_core = silog_loss(pred, gt, mask=mask) if use_silog else _masked_mean((pred - gt).abs(), mask)
    l_ssim = ssim_loss(pred, gt, mask=mask)
    l_ord = ordinal_loss(pred, gt, mask=mask)
    l_smooth = smoothness_loss(pred, image=image)
    return w_silog * l_core + w_ssim * l_ssim + w_order * l_ord + w_smooth * l_smooth
