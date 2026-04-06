"""Thermal-SLAM model: T-RefNet + Encoder + Recurrent Block + Depth Decoder.

Paper: "Thermal Image Refinement with Depth Estimation using Recurrent Networks
        for Monocular ORB-SLAM3" (Sahin et al., ICRA 2026)
"""

from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# T-RefNet — Thermal Refinement Network
# ---------------------------------------------------------------------------

class TRefNet(nn.Module):
    """Thermal Refinement Network.

    Takes raw 16-bit thermal input (1-channel) and produces:
      - normalized thermal (1-channel, 0-1 range) for depth estimation
      - 8-bit colormap (3-channel) for ORB feature extraction in SLAM
    """

    def __init__(self) -> None:
        super().__init__()
        # Learnable normalization branch
        self.norm_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        # Colormap branch: produces 3-channel pseudo-color
        self.color_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Raw thermal input (B, 1, H, W), float32 in arbitrary range.

        Returns:
            normalized: (B, 1, H, W) in [0, 1]
            colormap: (B, 3, H, W) in [0, 1]
        """
        # Coarse normalization: shift to 0-1 per-sample
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x_coarse = (x - x_min) / (x_max - x_min + 1e-6)

        normalized = self.norm_branch(x_coarse)
        colormap = self.color_branch(x_coarse)
        return normalized, colormap


# ---------------------------------------------------------------------------
# Encoder Backbones (via timm)
# ---------------------------------------------------------------------------

class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder producing multi-scale features."""

    def __init__(self, pretrained: bool = True, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.feature_dims = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features [1/4, 1/8, 1/16, 1/32]."""
        return self.backbone(x)


class MobileNetEncoder(nn.Module):
    """MobileNetV2 encoder producing multi-scale features."""

    def __init__(self, pretrained: bool = True, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.feature_dims = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)


def build_encoder(
    name: str, pretrained: bool = True, in_channels: int = 1
) -> EfficientNetEncoder | MobileNetEncoder:
    """Factory for encoder backbones."""
    if name == "efficientnet_b0":
        return EfficientNetEncoder(pretrained=pretrained, in_channels=in_channels)
    if name in ("mobilenetv2", "mobilenetv2_100"):
        return MobileNetEncoder(pretrained=pretrained, in_channels=in_channels)
    raise ValueError(f"Unknown encoder: {name}")


# ---------------------------------------------------------------------------
# Recurrent Blocks
# ---------------------------------------------------------------------------

class ConvGRU(nn.Module):
    """Convolutional GRU for temporal feature propagation (~800K params).

    Operates on the deepest encoder feature map.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2

        # Gates
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)

        # Project hidden back to input dim for skip addition
        self.proj = nn.Conv2d(hidden_dim, input_dim, 1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward one timestep.

        Args:
            x: Input features (B, C, H, W)
            h: Previous hidden state (B, hidden_dim, H, W) or None

        Returns:
            out: Processed features (B, C, H, W) — same shape as input
            h_new: Updated hidden state
        """
        b, _, fh, fw = x.shape
        if h is None:
            h = torch.zeros(b, self.hidden_dim, fh, fw, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        h_new = (1 - z) * h + z * h_tilde

        out = x + self.proj(h_new)
        return out, h_new


class ReservoirComputing(nn.Module):
    """Reservoir Computing block with LIF neurons (~50K params).

    Uses a fixed random reservoir weight matrix with trainable readout.
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 32,
        tau_m: float = 20.0,
        r_m: float = 1.0,
        leak_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.reservoir_size = reservoir_size
        self.tau_m = tau_m
        self.r_m = r_m
        self.leak_rate = leak_rate

        # Input projection (trainable)
        self.input_proj = nn.Conv2d(input_dim, reservoir_size, 1)

        # Reservoir weights (fixed random, not trained)
        # Sparse random matrix scaled for echo state property
        w_res = torch.randn(reservoir_size, reservoir_size) * 0.1
        # Spectral radius scaling
        spectral_radius = torch.linalg.eigvals(w_res.float()).abs().max()
        w_res = w_res * (0.9 / (spectral_radius + 1e-6))
        self.register_buffer("w_reservoir", w_res)

        # Readout (trainable)
        self.readout = nn.Sequential(
            nn.Conv2d(reservoir_size, input_dim, 1),
            nn.ReLU(inplace=True),
        )

    def _lif_step(
        self, current: torch.Tensor, voltage: torch.Tensor
    ) -> torch.Tensor:
        """Leaky-integrate-and-fire neuron step.

        tau_m * dV/dt = -V + R_m * I
        Discretized: V(t+1) = (1 - dt/tau_m) * V(t) + (R_m * dt / tau_m) * I(t)
        """
        dt = 1.0
        decay = 1.0 - dt / self.tau_m
        voltage_new = decay * voltage + (self.r_m * dt / self.tau_m) * current
        # Soft threshold (differentiable approximation of spike)
        spike = torch.sigmoid(10.0 * (voltage_new - 1.0))
        # Reset after spike
        voltage_new = voltage_new * (1.0 - spike * self.leak_rate)
        return voltage_new

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward one timestep.

        Args:
            x: (B, C, H, W)
            state: (B, reservoir_size, H, W) or None

        Returns:
            out: (B, C, H, W)
            state_new: (B, reservoir_size, H, W)
        """
        b, _, fh, fw = x.shape
        if state is None:
            state = torch.zeros(
                b, self.reservoir_size, fh, fw, device=x.device, dtype=x.dtype
            )

        # Input drive
        u = self.input_proj(x)  # (B, R, H, W)

        # Reservoir recurrence: x(t+1) = f(W_in * u + W * x(t))
        # Reshape for matmul: (B, R, H*W) -> reservoir -> (B, R, H*W)
        state_flat = state.reshape(b, self.reservoir_size, -1)
        recurrent = torch.einsum("ij,bjn->bin", self.w_reservoir, state_flat)
        recurrent = recurrent.reshape(b, self.reservoir_size, fh, fw)

        current = u + recurrent
        state_new = self._lif_step(current, state)

        out = x + self.readout(state_new)
        return out, state_new


# ---------------------------------------------------------------------------
# Depth Decoder
# ---------------------------------------------------------------------------

class UpProjectBlock(nn.Module):
    """Up-projection block for depth decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Align spatial dims if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = x + skip
        x = self.refine(x)
        return x


class DepthDecoder(nn.Module):
    """Multi-scale depth decoder with skip connections from encoder."""

    def __init__(
        self,
        encoder_dims: list[int],
        max_depth: float = 10.0,
        min_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth

        # Decoder blocks: from deepest to shallowest
        dims = list(reversed(encoder_dims))
        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.blocks.append(UpProjectBlock(dims[i], dims[i + 1]))

        # Final upsampling to full resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: list[torch.Tensor], target_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Decode multi-scale features to depth map.

        Args:
            features: [feat_1/4, feat_1/8, feat_1/16, feat_1/32] from encoder
            target_size: (H, W) for final output

        Returns:
            depth: (B, 1, H, W) in [min_depth, max_depth]
        """
        feats = list(reversed(features))  # deepest first
        x = feats[0]
        for i, block in enumerate(self.blocks):
            skip = feats[i + 1] if (i + 1) < len(feats) else None
            x = block(x, skip)

        x = self.final_up(x)
        if target_size is not None and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        depth = self.min_depth + (self.max_depth - self.min_depth) * x
        return depth


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class ThermalDepthNet(nn.Module):
    """Complete Thermal-SLAM depth estimation network.

    Pipeline: T-RefNet -> Encoder -> Recurrent Block -> Depth Decoder
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        recurrent_type: Literal["convgru", "rc_lif"] = "convgru",
        pretrained_encoder: bool = True,
        in_channels: int = 1,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
        convgru_hidden: int = 128,
        convgru_kernel: int = 3,
        rc_reservoir_size: int = 32,
        rc_tau_m: float = 20.0,
        rc_r_m: float = 1.0,
        rc_leak_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.t_refnet = TRefNet()
        self.encoder = build_encoder(encoder_name, pretrained_encoder, in_channels=in_channels)

        encoder_dims = self.encoder.feature_dims
        deepest_dim = encoder_dims[-1]

        if recurrent_type == "convgru":
            self.recurrent = ConvGRU(
                input_dim=deepest_dim,
                hidden_dim=convgru_hidden,
                kernel_size=convgru_kernel,
            )
        elif recurrent_type == "rc_lif":
            self.recurrent = ReservoirComputing(
                input_dim=deepest_dim,
                reservoir_size=rc_reservoir_size,
                tau_m=rc_tau_m,
                r_m=rc_r_m,
                leak_rate=rc_leak_rate,
            )
        else:
            raise ValueError(f"Unknown recurrent type: {recurrent_type}")

        self.decoder = DepthDecoder(
            encoder_dims=encoder_dims,
            max_depth=max_depth,
            min_depth=min_depth,
        )
        self._recurrent_state: torch.Tensor | None = None

    def reset_state(self) -> None:
        """Reset recurrent hidden state (call at start of each new sequence)."""
        self._recurrent_state = None

    def forward(
        self, x: torch.Tensor, return_refined: bool = False
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Raw thermal input (B, 1, H, W)
            return_refined: If True, also return T-RefNet outputs

        Returns:
            Dictionary with keys:
              - depth: (B, 1, H, W) predicted depth
              - normalized: (B, 1, H, W) refined thermal (if return_refined)
              - colormap: (B, 3, H, W) colormap for SLAM (if return_refined)
        """
        target_size = (x.shape[2], x.shape[3])

        # T-RefNet: normalize raw thermal
        normalized, colormap = self.t_refnet(x)

        # Encoder: multi-scale features from normalized thermal
        features = self.encoder(normalized)

        # Recurrent block: temporal propagation on deepest features
        features[-1], self._recurrent_state = self.recurrent(
            features[-1], self._recurrent_state
        )

        # Decoder: depth prediction
        depth = self.decoder(features, target_size=target_size)

        out: dict[str, torch.Tensor] = {"depth": depth}
        if return_refined:
            out["normalized"] = normalized
            out["colormap"] = colormap
        return out


def build_model(cfg: dict) -> ThermalDepthNet:
    """Build model from config dictionary."""
    model_cfg = cfg.get("model", {})
    gru_cfg = model_cfg.get("convgru", {})
    rc_cfg = model_cfg.get("rc_lif", {})

    return ThermalDepthNet(
        encoder_name=model_cfg.get("encoder", "efficientnet_b0"),
        recurrent_type=model_cfg.get("recurrent_block", "convgru"),
        pretrained_encoder=model_cfg.get("pretrained_encoder", True),
        in_channels=model_cfg.get("input_channels", 1),
        max_depth=model_cfg.get("max_depth", 10.0),
        min_depth=model_cfg.get("min_depth", 0.1),
        convgru_hidden=gru_cfg.get("hidden_dim", 128),
        convgru_kernel=gru_cfg.get("kernel_size", 3),
        rc_reservoir_size=rc_cfg.get("reservoir_size", 32),
        rc_tau_m=rc_cfg.get("tau_m", 20.0),
        rc_r_m=rc_cfg.get("r_m", 1.0),
        rc_leak_rate=rc_cfg.get("leak_rate", 0.3),
    )
