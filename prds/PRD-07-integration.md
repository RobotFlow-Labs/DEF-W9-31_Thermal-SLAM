# PRD-07: Integration

## Objective
Docker serving infrastructure, FastAPI endpoints, and ANIMA ecosystem integration.

## Components

### Dockerfile.serve
- Base: `ghcr.io/robotflow-labs/anima-serve:jazzy`
- Install module code + dependencies
- Runtime weight download from HuggingFace

### docker-compose.serve.yml
- Profiles: serve, ros2, api, test
- GPU passthrough via NVIDIA runtime
- Health endpoint at /health

### FastAPI Endpoints
- `GET /health` — status, uptime, GPU VRAM
- `GET /ready` — weights loaded check
- `GET /info` — module metadata
- `POST /predict` — accept 16-bit thermal, return depth map + refined thermal

### ROS2 Topics (via anima_module.yaml)
- Subscribe: `/camera/thermal/raw`
- Publish: `/defense/thermal_slam/depth`, `/defense/thermal_slam/refined`

## Deliverables
- [x] `Dockerfile.serve`
- [x] `docker-compose.serve.yml`
- [x] `src/thermal_slam/serve.py` — FastAPI app

## Acceptance Criteria
- `docker compose --profile api up` starts without error
- `/health` returns 200
- `/predict` accepts thermal image and returns depth
