"""Tests for Thermal-SLAM model components."""

from __future__ import annotations

import pytest
import torch

from thermal_slam.model import (
    ConvGRU,
    EfficientNetEncoder,
    ReservoirComputing,
    TRefNet,
    build_model,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTRefNet:
    def test_output_shapes(self, device: torch.device) -> None:
        model = TRefNet().to(device)
        x = torch.randn(2, 1, 128, 160, device=device)
        normalized, colormap = model(x)

        assert normalized.shape == (2, 1, 128, 160)
        assert colormap.shape == (2, 3, 128, 160)

    def test_output_range(self, device: torch.device) -> None:
        model = TRefNet().to(device)
        x = torch.randn(1, 1, 64, 80, device=device) * 1000 + 5000
        normalized, colormap = model(x)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert colormap.min() >= 0.0
        assert colormap.max() <= 1.0


class TestEncoder:
    def test_efficientnet_shapes(self, device: torch.device) -> None:
        enc = EfficientNetEncoder(pretrained=False, in_channels=1).to(device)
        x = torch.randn(1, 1, 128, 160, device=device)
        feats = enc(x)

        assert len(feats) == 4
        # Each feature should be progressively smaller spatially
        for i in range(len(feats) - 1):
            assert feats[i].shape[2] >= feats[i + 1].shape[2]

    def test_feature_dims(self, device: torch.device) -> None:
        enc = EfficientNetEncoder(pretrained=False, in_channels=1).to(device)
        assert len(enc.feature_dims) == 4
        assert all(d > 0 for d in enc.feature_dims)


class TestConvGRU:
    def test_forward(self, device: torch.device) -> None:
        gru = ConvGRU(input_dim=64, hidden_dim=32, kernel_size=3).to(device)
        x = torch.randn(2, 64, 8, 10, device=device)
        out, h = gru(x, None)

        assert out.shape == x.shape
        assert h.shape == (2, 32, 8, 10)

    def test_stateful(self, device: torch.device) -> None:
        gru = ConvGRU(input_dim=64, hidden_dim=32, kernel_size=3).to(device)
        x = torch.randn(2, 64, 8, 10, device=device)
        out1, h1 = gru(x, None)
        out2, h2 = gru(x, h1)
        # Second call with state should produce different output
        assert not torch.allclose(out1, out2)


class TestReservoirComputing:
    def test_forward(self, device: torch.device) -> None:
        rc = ReservoirComputing(input_dim=64, reservoir_size=16).to(device)
        x = torch.randn(2, 64, 8, 10, device=device)
        out, state = rc(x, None)

        assert out.shape == x.shape
        assert state.shape == (2, 16, 8, 10)

    def test_param_count(self, device: torch.device) -> None:
        rc = ReservoirComputing(input_dim=320, reservoir_size=32).to(device)
        trainable = sum(p.numel() for p in rc.parameters() if p.requires_grad)
        # RC should be much lighter than ConvGRU
        assert trainable < 100_000


class TestThermalDepthNet:
    def test_forward_convgru(self, device: torch.device) -> None:
        cfg = {
            "model": {
                "encoder": "efficientnet_b0",
                "recurrent_block": "convgru",
                "pretrained_encoder": False,
                "input_channels": 1,
                "max_depth": 10.0,
                "min_depth": 0.1,
                "convgru": {"hidden_dim": 64, "kernel_size": 3},
            }
        }
        model = build_model(cfg).to(device)
        x = torch.randn(1, 1, 128, 160, device=device)
        out = model(x, return_refined=True)

        assert "depth" in out
        assert "normalized" in out
        assert "colormap" in out
        assert out["depth"].shape == (1, 1, 128, 160)
        assert out["depth"].min() >= 0.1
        assert out["depth"].max() <= 10.0

    def test_forward_rc(self, device: torch.device) -> None:
        cfg = {
            "model": {
                "encoder": "efficientnet_b0",
                "recurrent_block": "rc_lif",
                "pretrained_encoder": False,
                "input_channels": 1,
                "max_depth": 10.0,
                "min_depth": 0.1,
                "rc_lif": {"reservoir_size": 16, "tau_m": 20.0},
            }
        }
        model = build_model(cfg).to(device)
        x = torch.randn(1, 1, 128, 160, device=device)
        out = model(x)

        assert out["depth"].shape == (1, 1, 128, 160)

    def test_reset_state(self, device: torch.device) -> None:
        cfg = {
            "model": {
                "encoder": "efficientnet_b0",
                "recurrent_block": "convgru",
                "pretrained_encoder": False,
                "input_channels": 1,
                "convgru": {"hidden_dim": 32, "kernel_size": 3},
            }
        }
        model = build_model(cfg).to(device)
        x = torch.randn(1, 1, 64, 80, device=device)

        model(x)
        assert model._recurrent_state is not None
        model.reset_state()
        assert model._recurrent_state is None

    def test_gradient_flow(self, device: torch.device) -> None:
        cfg = {
            "model": {
                "encoder": "efficientnet_b0",
                "recurrent_block": "convgru",
                "pretrained_encoder": False,
                "input_channels": 1,
                "convgru": {"hidden_dim": 32, "kernel_size": 3},
            }
        }
        model = build_model(cfg).to(device)
        x = torch.randn(1, 1, 64, 80, device=device)
        out = model(x)
        loss = out["depth"].mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad
