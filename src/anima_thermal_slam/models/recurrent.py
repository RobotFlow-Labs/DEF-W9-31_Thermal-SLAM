from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        p = kernel_size // 2
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size, padding=p)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size, padding=p)
        self.hid_ch = hid_ch

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        if h_prev is None:
            b, _, h, w = x.shape
            h_prev = torch.zeros((b, self.hid_ch, h, w), device=x.device, dtype=x.dtype)
        joint = torch.cat([x, h_prev], dim=1)
        zr = self.conv_zr(joint)
        z, r = torch.chunk(zr, 2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h_prev], dim=1)))
        return (1.0 - z) * h_prev + z * h_tilde


class ConvGRU(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, layers: int = 1) -> None:
        super().__init__()
        cells = []
        c_in = in_ch
        for _ in range(layers):
            cells.append(ConvGRUCell(c_in, hid_ch))
            c_in = hid_ch
        self.cells = nn.ModuleList(cells)

    def forward(self, x_seq: torch.Tensor, return_sequence: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x_seq: (B,T,C,H,W) -> last: (B,C,H,W), optional full seq."""
        hs: list[Optional[torch.Tensor]] = [None for _ in self.cells]
        seq_out: list[torch.Tensor] = []
        for t in range(x_seq.shape[1]):
            z = x_seq[:, t]
            for i, cell in enumerate(self.cells):
                hs[i] = cell(z, hs[i])
                z = hs[i]
            if return_sequence:
                seq_out.append(z)
        last = z
        if return_sequence:
            return last, torch.stack(seq_out, dim=1)
        return last, None


class ReservoirTemporal(nn.Module):
    """Lightweight reservoir-inspired temporal block on per-pixel feature vectors.

    Input  : (B,T,C,H,W)
    Output : last (B,C,H,W), optional full sequence (B,T,C,H,W)
    """

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden

        # Trainable projections. Can be frozen later to mimic strict RC behavior.
        self.in_proj = nn.Linear(channels, hidden)
        self.state_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, channels)
        self.act = nn.Tanh()

    def forward(self, x_seq: torch.Tensor, return_sequence: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, c, h, w = x_seq.shape
        # (B,H,W,T,C) -> (BHW,T,C)
        xs = x_seq.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)

        state = xs.new_zeros((b * h * w, self.hidden))
        seq_out = []
        for i in range(t):
            state = self.act(self.in_proj(xs[:, i]) + self.state_proj(state))
            y = self.out_proj(state)
            if return_sequence:
                seq_out.append(y)

        last = y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if not return_sequence:
            return last, None

        ys = torch.stack(seq_out, dim=1)  # (BHW,T,C)
        full = ys.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
        return last, full
