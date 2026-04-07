"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from thermal_slam.evaluate import compute_depth_metrics


class TestDepthMetrics:
    def test_perfect_prediction(self) -> None:
        pred = np.ones((64, 80)) * 5.0
        target = np.ones((64, 80)) * 5.0
        m = compute_depth_metrics(pred, target)
        assert m["abs_rel"] < 1e-6
        assert m["rmse"] < 1e-6
        assert m["a1"] == 1.0

    def test_scaled_prediction(self) -> None:
        target = np.random.uniform(1.0, 8.0, (32, 32))
        pred = target * 1.1  # 10% overestimate
        m = compute_depth_metrics(pred, target)
        assert 0.05 < m["abs_rel"] < 0.15
        assert m["a1"] > 0.9

    def test_all_invalid(self) -> None:
        pred = np.zeros((16, 16))
        target = np.zeros((16, 16))
        m = compute_depth_metrics(pred, target)
        assert np.isnan(m["abs_rel"])

    def test_metric_keys(self) -> None:
        pred = np.random.uniform(0.5, 5.0, (8, 8))
        target = np.random.uniform(0.5, 5.0, (8, 8))
        m = compute_depth_metrics(pred, target)
        expected = {"abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"}
        assert set(m.keys()) == expected

    def test_delta_ordering(self) -> None:
        pred = np.random.uniform(1.0, 5.0, (32, 32))
        target = pred + np.random.normal(0, 0.3, pred.shape)
        target = np.clip(target, 0.5, 8.0)
        m = compute_depth_metrics(pred, target)
        assert m["a1"] <= m["a2"] <= m["a3"]
