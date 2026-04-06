"""FastAPI serving endpoint for Thermal-SLAM depth estimation.

Endpoints:
  GET  /health  — status + uptime
  GET  /ready   — weights loaded check
  GET  /info    — module metadata
  POST /predict — thermal frame -> depth map
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from thermal_slam.model import build_model
from thermal_slam.utils import load_config

app = FastAPI(title="DEF-thermal-slam", version="0.1.0")

# Global state
_state = {
    "model": None,
    "device": None,
    "config": None,
    "start_time": time.time(),
    "ready": False,
}


def _load_model(config_path: str, checkpoint_path: str | None = None) -> None:
    """Load model into global state."""
    import os

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.eval()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

    _state["model"] = model
    _state["device"] = device
    _state["config"] = cfg
    _state["ready"] = True


@app.get("/health")
def health() -> dict:
    uptime = time.time() - _state["start_time"]
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["gpu_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    return {
        "status": "ok",
        "module": "DEF-thermal-slam",
        "uptime_s": round(uptime, 1),
        **gpu_info,
    }


@app.get("/ready")
def ready() -> JSONResponse:
    if _state["ready"]:
        return JSONResponse(
            {"ready": True, "module": "DEF-thermal-slam", "version": "0.1.0",
             "weights_loaded": True}
        )
    return JSONResponse(
        {"ready": False, "module": "DEF-thermal-slam"}, status_code=503
    )


@app.get("/info")
def info() -> dict:
    return {
        "module": "DEF-thermal-slam",
        "version": "0.1.0",
        "task": "thermal_depth_estimation_slam",
        "paper": "Thermal Image Refinement with Depth Estimation using Recurrent Networks "
                 "for Monocular ORB-SLAM3",
        "arxiv": "2603.14998",
        "encoder": "efficientnet_b0",
        "recurrent": "convgru",
    }


MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_RESOLUTION = 2048


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:  # noqa: B008
    """Run depth estimation on uploaded thermal image.

    Accepts .npy (16-bit float32 array) or raw bytes.
    Returns depth map as nested list.
    """
    if not _state["ready"]:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File too large"}, status_code=413)

    buf = io.BytesIO(contents)

    # Load as numpy array
    try:
        thermal_np = np.load(buf, allow_pickle=False).astype(np.float32)
    except Exception:
        # Try loading as raw image
        import cv2

        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return JSONResponse({"error": "Cannot decode image"}, status_code=400)
        thermal_np = img.astype(np.float32)

    # Validate resolution
    h, w = thermal_np.shape[:2]
    if h > MAX_RESOLUTION or w > MAX_RESOLUTION:
        return JSONResponse(
            {"error": f"Resolution {w}x{h} exceeds limit {MAX_RESOLUTION}"},
            status_code=400,
        )

    if thermal_np.ndim == 2:
        thermal_np = thermal_np[np.newaxis, np.newaxis, :, :]
    elif thermal_np.ndim == 3:
        thermal_np = thermal_np[np.newaxis, :1, :, :]

    # Normalize
    t_min, t_max = thermal_np.min(), thermal_np.max()
    if t_max - t_min > 1e-6:
        thermal_np = (thermal_np - t_min) / (t_max - t_min)

    device = _state["device"]
    model = _state["model"]
    tensor = torch.from_numpy(thermal_np).float().to(device)

    with torch.no_grad():
        model.reset_state()
        out = model(tensor, return_refined=True)

    depth = out["depth"].cpu().numpy()[0, 0].tolist()
    return {"depth": depth, "shape": list(out["depth"].shape)}


def main() -> None:
    """Entry point for serving."""
    import os

    config_path = os.environ.get("ANIMA_CONFIG", "configs/paper.toml")
    checkpoint_path = os.environ.get("ANIMA_CHECKPOINT", None)
    port = int(os.environ.get("ANIMA_SERVE_PORT", "8080"))

    if Path(config_path).exists():
        _load_model(config_path, checkpoint_path)

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
