from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidBackbone(Protocol):
    num_ch_enc: list[int]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]: ...


class SimpleCNNBackbone(nn.Module):
    """Dependency-free 5-scale pyramid backbone for local testing.

    This is not paper-SOTA, but keeps interfaces stable until timm backbones
    are enabled on training servers.
    """

    def __init__(self, in_chans: int = 1) -> None:
        super().__init__()
        chs = [16, 32, 64, 96, 128]
        self.stages = nn.ModuleList()
        c_in = in_chans
        for c_out in chs:
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
            )
            c_in = c_out
        self.num_ch_enc = chs

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        y = x
        for stage in self.stages:
            y = stage(y)
            feats.append(y)
        return feats


class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, in_chans: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("timm is required for non-simple backbones") from exc

        self.model = timm.create_model(model_name, features_only=True, in_chans=in_chans, pretrained=pretrained)
        self.num_ch_enc = [fi["num_chs"] for fi in self.model.feature_info]
        if len(self.num_ch_enc) < 5:
            self.num_ch_enc = (self.num_ch_enc + [self.num_ch_enc[-1]] * 5)[:5]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = list(self.model(x))
        if len(feats) < 5:
            feats = (feats + [feats[-1]] * 5)[:5]
        return feats


def build_backbone(name: str, in_chans: int = 1, pretrained: bool = False) -> nn.Module:
    n = (name or "").lower()
    if n in {"simple", "simple_cnn", "baseline"}:
        return SimpleCNNBackbone(in_chans=in_chans)

    try:
        return TimmBackbone(model_name=n, in_chans=in_chans, pretrained=pretrained)
    except Exception:
        # Safe fallback so local tests still run without timm.
        return SimpleCNNBackbone(in_chans=in_chans)


def ensure_five_scales(feats: list[torch.Tensor]) -> list[torch.Tensor]:
    if len(feats) == 5:
        return feats
    if len(feats) > 5:
        return feats[:5]
    out = feats[:]
    while len(out) < 5:
        out.append(F.interpolate(out[-1], scale_factor=0.5, mode="bilinear", align_corners=False))
    return out
