"""Tests for Thermal-SLAM dataset and losses."""

from __future__ import annotations

import pytest
import torch

from thermal_slam.dataset import (
    ThermalDepthDataset,
    VIVIDPlusPlusDataset,
    create_split_indices,
)
from thermal_slam.losses import (
    CompositeDepthLoss,
    EdgeAwareSmoothnessLoss,
    OrdinalDepthLoss,
    ScaleInvariantLogLoss,
    SSIMLoss,
    build_loss,
)


class TestThermalDepthDataset:
    def test_empty_dataset_warns(self) -> None:
        """Empty dataset should have len 0 and warn."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = ThermalDepthDataset(
                root="/tmp/nonexistent_thermal_data",
                height=64,
                width=80,
            )
            _ = len(ds)
            assert len(ds) == 0
            assert any("no samples" in str(warning.message).lower() for warning in w)

    def test_split_indices(self) -> None:
        split = create_split_indices(100, train_ratio=0.9, val_ratio=0.05, seed=42)
        assert len(split["train"]) == 90
        assert len(split["val"]) == 5
        assert len(split["test"]) == 5

        # No overlap
        all_idx = set(split["train"]) | set(split["val"]) | set(split["test"])
        assert len(all_idx) == 100

    def test_deterministic_split(self) -> None:
        s1 = create_split_indices(50, seed=42)
        s2 = create_split_indices(50, seed=42)
        assert s1["train"] == s2["train"]


class TestVIVIDPlusPlusDataset:
    def test_empty_dataset_warns(self) -> None:
        """Empty dataset should have len 0 and warn."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = VIVIDPlusPlusDataset(
                root="/tmp/nonexistent_vivid",
                split="train",
                height=64,
                width=80,
            )
            _ = len(ds)
            assert len(ds) == 0
            assert any("no samples" in str(warning.message).lower() for warning in w)

    @pytest.mark.skipif(
        not __import__("pathlib").Path(
            "/mnt/forge-data/datasets/vivid_plus_plus/dataset/train/train.txt"
        ).exists(),
        reason="VIVID++ not available",
    )
    def test_real_data_loading(self) -> None:
        """Test loading real VIVID++ data."""
        ds = VIVIDPlusPlusDataset(
            root="/mnt/forge-data/datasets/vivid_plus_plus",
            split="train",
            height=256,
            width=320,
        )
        assert len(ds) > 100, f"Expected >100 samples, got {len(ds)}"
        sample = ds[0]
        assert sample["thermal"].shape == (1, 256, 320)
        assert sample["depth"].shape == (1, 256, 320)
        assert sample["thermal"].min() >= 0.0
        assert sample["thermal"].max() <= 1.0
        assert sample["depth"].min() >= 0.0


class TestScaleInvariantLogLoss:
    def test_zero_loss(self) -> None:
        loss_fn = ScaleInvariantLogLoss()
        pred = torch.ones(2, 1, 4, 4) * 5.0
        target = torch.ones(2, 1, 4, 4) * 5.0
        loss = loss_fn(pred, target)
        assert loss.item() < 1e-6

    def test_gradient(self) -> None:
        loss_fn = ScaleInvariantLogLoss()
        pred = torch.randn(1, 1, 4, 4).abs() + 0.1
        pred.requires_grad = True
        target = torch.randn(1, 1, 4, 4).abs() + 0.1
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestSSIMLoss:
    def test_identical_zero(self) -> None:
        loss_fn = SSIMLoss()
        x = torch.rand(1, 1, 16, 16)
        loss = loss_fn(x, x.clone())
        assert loss.item() < 0.05

    def test_gradient(self) -> None:
        loss_fn = SSIMLoss()
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = torch.rand(1, 1, 16, 16)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestOrdinalDepthLoss:
    def test_gradient(self) -> None:
        loss_fn = OrdinalDepthLoss(num_pairs=100)
        pred = torch.rand(1, 1, 8, 8, requires_grad=True)
        target = torch.rand(1, 1, 8, 8) + 0.1
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestEdgeAwareSmoothnessLoss:
    def test_gradient(self) -> None:
        loss_fn = EdgeAwareSmoothnessLoss()
        pred = torch.rand(1, 1, 8, 8, requires_grad=True)
        image = torch.rand(1, 1, 8, 8)
        loss = loss_fn(pred, image)
        loss.backward()
        assert pred.grad is not None


class TestCompositeLoss:
    def test_forward(self) -> None:
        loss_fn = CompositeDepthLoss()
        pred = torch.rand(1, 1, 16, 16) * 9.0 + 1.0
        pred.requires_grad = True
        target = torch.rand(1, 1, 16, 16) * 9.0 + 1.0
        image = torch.rand(1, 1, 16, 16)

        result = loss_fn(pred, target, image)
        assert "total" in result
        assert "silog" in result
        assert "ssim" in result
        assert "ordinal" in result
        assert "smoothness" in result

        result["total"].backward()
        assert pred.grad is not None

    def test_build_from_config(self) -> None:
        cfg = {
            "loss": {
                "silog_weight": 0.9,
                "ssim_weight": 0.4,
                "ordinal_weight": 0.1,
                "smoothness_weight": 0.1,
            }
        }
        loss_fn = build_loss(cfg)
        assert loss_fn.silog_weight == 0.9
        assert loss_fn.ssim_weight == 0.4
