from __future__ import annotations

import torch
import torch.nn as nn


class TRefNet(nn.Module):
    """Thermal refinement network from paper Figure 1.

    Layout: Conv(1->16)-ReLU-Conv(16->16)-ReLU-Conv(16->8)-ReLU-Conv(8->1)-Sigmoid
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
