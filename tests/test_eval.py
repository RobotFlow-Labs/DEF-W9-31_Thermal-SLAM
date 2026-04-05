from __future__ import annotations

import numpy as np

from anima_thermal_slam.eval import compute_depth_metrics, mean_trajectory_error, render_eval_report


def test_compute_depth_metrics_perfect_prediction() -> None:
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = gt.copy()

    m = compute_depth_metrics(gt, pred)

    assert m.abs_rel == 0.0
    assert m.rmse == 0.0
    assert m.mae == 0.0
    assert m.a1 == 1.0
    assert m.a2 == 1.0
    assert m.a3 == 1.0


def test_mean_trajectory_error() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    est = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    err = mean_trajectory_error(gt, est)
    assert np.isclose(err, 0.5, atol=1e-6)


def test_render_eval_report_contains_sections() -> None:
    gt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pred = np.array([1.0, 2.5, 3.0], dtype=np.float32)
    metrics = compute_depth_metrics(gt, pred)
    report = render_eval_report("Thermal-SLAM", metrics, traj_error=0.1234)
    assert "# Thermal-SLAM Evaluation Report" in report
    assert "## Depth Metrics" in report
    assert "## Trajectory" in report

