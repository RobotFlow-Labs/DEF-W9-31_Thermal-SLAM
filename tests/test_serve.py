"""Tests for FastAPI serving endpoint."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient

from thermal_slam.serve import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health(self) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["module"] == "DEF-thermal-slam"

    def test_ready_without_model(self) -> None:
        r = client.get("/ready")
        assert r.status_code == 503

    def test_info(self) -> None:
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert data["arxiv"] == "2603.14998"
        assert data["encoder"] == "efficientnet_b0"


class TestPredictEndpoint:
    def test_predict_without_model(self) -> None:
        buf = io.BytesIO()
        np.save(buf, np.random.randn(64, 80).astype(np.float32))
        buf.seek(0)
        r = client.post("/predict", files={"file": ("test.npy", buf)})
        assert r.status_code == 503

    def test_oversized_upload_constant(self) -> None:
        from thermal_slam.serve import MAX_UPLOAD_BYTES
        assert MAX_UPLOAD_BYTES == 50 * 1024 * 1024
