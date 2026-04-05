from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthDecoder(nn.Module):
    """U-Net style decoder matching reference structure."""

    def __init__(self, num_ch_enc: list[int], scales: range = range(4), use_skips: bool = True) -> None:
        super().__init__()
        self.alpha = 10.0
        self.beta = 0.01
        self.scales = list(scales)
        self.use_skips = use_skips
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [8, 16, 32, 64, 128]

        convs: OrderedDict[tuple[str, int, int] | tuple[str, int], nn.Module] = OrderedDict()
        for i in range(4, -1, -1):
            ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            convs[("upconv", i, 0)] = ConvBlock(ch_in, self.num_ch_dec[i])
            ch_in = self.num_ch_dec[i]
            if use_skips and i > 0:
                ch_in += num_ch_enc[i - 1]
            convs[("upconv", i, 1)] = ConvBlock(ch_in, self.num_ch_dec[i])

        for s in self.scales:
            convs[("dispconv", s)] = nn.Conv2d(self.num_ch_dec[s], 1, kernel_size=3, padding=1)

        self.convs = nn.ModuleDict({str(k): v for k, v in convs.items()})
        self._keymap = convs
        self.sigmoid = nn.Sigmoid()

    def _get(self, key: tuple[str, int, int] | tuple[str, int]) -> nn.Module:
        return self.convs[str(key)]

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        x = feats[-1]
        outs = []
        for i in range(4, -1, -1):
            x = self._get(("upconv", i, 0))(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.use_skips and i > 0:
                skip = feats[i - 1]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="nearest")
                x = torch.cat([x, skip], dim=1)
            x = self._get(("upconv", i, 1))(x)
            if i in self.scales:
                disp = self.alpha * self.sigmoid(self._get(("dispconv", i))(x)) + self.beta
                outs.append(disp)
        return outs[::-1]
