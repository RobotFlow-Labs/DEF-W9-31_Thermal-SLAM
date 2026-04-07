"""Tests for utility functions."""

from __future__ import annotations

import os
import tempfile

import torch

from thermal_slam.utils import (
    CheckpointManager,
    EarlyStopping,
    WarmupCosineScheduler,
    set_seed,
)


class TestWarmupCosineScheduler:
    def test_warmup_phase(self) -> None:
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)

        # LR should increase during warmup
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(sched.get_lr())
        assert lrs[-1] > lrs[0]
        assert abs(lrs[-1] - 1e-3) < 1e-6  # should reach base LR

    def test_cosine_decay(self) -> None:
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=100)

        for _ in range(5):
            sched.step()
        lr_peak = sched.get_lr()

        for _ in range(50):
            sched.step()
        lr_mid = sched.get_lr()
        assert lr_mid < lr_peak

    def test_state_dict_resume(self) -> None:
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=100)

        for _ in range(20):
            sched.step()
        state = sched.state_dict()
        lr_before = sched.get_lr()

        # New scheduler, load state
        opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched2 = WarmupCosineScheduler(opt2, warmup_steps=5, total_steps=100)
        sched2.load_state_dict(state)
        assert abs(sched2.get_lr() - lr_before) < 1e-8


class TestEarlyStopping:
    def test_no_stop_when_improving(self) -> None:
        es = EarlyStopping(patience=5)
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert not es.step(v)

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(0.5)  # best
        es.step(0.6)  # worse
        es.step(0.7)  # worse
        assert es.step(0.8)  # patience exhausted

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(0.5)
        es.step(0.6)
        es.step(0.7)
        es.step(0.4)  # improvement resets
        assert not es.step(0.5)
        assert not es.step(0.6)


class TestCheckpointManager:
    def test_keeps_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir, keep_top_k=2)
            state = {"model": {}}
            mgr.save(state, 0.5, 100)
            mgr.save(state, 0.3, 200)
            mgr.save(state, 0.4, 300)

            # Should keep 2 best (0.3, 0.4) and delete 0.5
            pth_files = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint")]
            assert len(pth_files) == 2

    def test_best_pth_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir, keep_top_k=2)
            mgr.save({"model": {}}, 0.5, 100)
            assert os.path.exists(os.path.join(tmpdir, "best.pth"))

    def test_best_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir, keep_top_k=2)
            mgr.save({"model": {}}, 0.5, 100)
            mgr.save({"model": {}}, 0.3, 200)
            assert mgr.best_metric == 0.3


class TestSetSeed:
    def test_reproducibility(self) -> None:
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)
