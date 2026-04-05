from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import build_backbone, ensure_five_scales
from .decoder import DepthDecoder
from .recurrent import ConvGRU, ReservoirTemporal
from .trefnet import TRefNet


@dataclass
class ThermalDepthOutput:
    depth: torch.Tensor  # (B,1,H,W)
    depth_pyramid: list[torch.Tensor]
    refined_last: torch.Tensor  # (B,1,H,W)
    depth_sequence: torch.Tensor | None = None  # (B,T,1,H,W)


class ThermalDepthNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        backbone: str = "simple_cnn",
        recurrent: str = "convgru",
        hidden_channels: int = 128,
        recurrent_layers: int = 1,
        use_trefnet: bool = True,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.recurrent_mode = recurrent.lower()
        self.refine = TRefNet(in_channels=in_channels) if use_trefnet else nn.Identity()
        self.backbone = build_backbone(name=backbone, in_chans=in_channels, pretrained=pretrained_backbone)

        ch_enc = list(getattr(self.backbone, "num_ch_enc", [16, 32, 64, 96, 128]))
        bottleneck_ch = ch_enc[-1]
        self.proj_in = nn.Conv2d(bottleneck_ch, hidden_channels, 1) if hidden_channels != bottleneck_ch else nn.Identity()
        self.proj_out = nn.Conv2d(hidden_channels, bottleneck_ch, 1) if hidden_channels != bottleneck_ch else nn.Identity()

        if self.recurrent_mode == "convgru":
            self.recurrent = ConvGRU(in_ch=hidden_channels, hid_ch=hidden_channels, layers=recurrent_layers)
        elif self.recurrent_mode == "reservoir":
            self.recurrent = ReservoirTemporal(channels=hidden_channels, hidden=max(16, hidden_channels // 2))
        else:
            self.recurrent = None

        self.decoder = DepthDecoder(num_ch_enc=ch_enc)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = ensure_five_scales(list(self.backbone(x)))
        return feats

    def forward(self, x: torch.Tensor, decode_all: bool = False) -> ThermalDepthOutput:
        if x.dim() == 4:
            # Single frame -> fake sequence of length 1 for uniform processing.
            x = x.unsqueeze(1)

        if x.dim() != 5:
            raise ValueError("Expected input shape (B,C,H,W) or (B,T,C,H,W)")

        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        refined = self.refine(x_flat)

        feats_bt = self._encode(refined)
        bottleneck = feats_bt[-1]
        _, cb, hb, wb = bottleneck.shape
        bottleneck_seq = bottleneck.view(b, t, cb, hb, wb)
        bottleneck_proj = self.proj_in(bottleneck_seq.view(b * t, cb, hb, wb)).view(b, t, -1, hb, wb)

        seq_out: torch.Tensor | None = None
        if self.recurrent is None:
            last_hidden = bottleneck_proj[:, -1]
            if decode_all:
                seq_out = bottleneck_proj
        else:
            last_hidden, seq_out = self.recurrent(bottleneck_proj, return_sequence=decode_all)

        bottleneck_last = self.proj_out(last_hidden)

        # last frame features from pyramid
        feats_last = []
        for f in feats_bt:
            ck, hk, wk = f.shape[1:]
            fv = f.view(b, t, ck, hk, wk)
            feats_last.append(fv[:, -1])
        feats_last[-1] = bottleneck_last

        depth_pyramid = self.decoder(feats_last)
        depth = depth_pyramid[0]

        depth_sequence: torch.Tensor | None = None
        if decode_all and seq_out is not None:
            seq_preds = []
            for i in range(seq_out.shape[1]):
                bneck_i = self.proj_out(seq_out[:, i])
                feats_i = []
                for f in feats_bt:
                    ck, hk, wk = f.shape[1:]
                    fv = f.view(b, t, ck, hk, wk)
                    feats_i.append(fv[:, i])
                feats_i[-1] = bneck_i
                seq_preds.append(self.decoder(feats_i)[0])
            depth_sequence = torch.stack(seq_preds, dim=1)

        refined_last = refined.view(b, t, c, h, w)[:, -1]
        return ThermalDepthOutput(
            depth=depth,
            depth_pyramid=depth_pyramid,
            refined_last=refined_last,
            depth_sequence=depth_sequence,
        )
