# PRD-05: API & Docker

> Module: Thermal-SLAM | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose model inference as a service with health/readiness/predict endpoints and containerization artifacts.

## Context (from paper)
Real-time deployment focus for UAV and embedded workflows requires stable service interfaces.

Paper references:
- §I and §V: real-time operation and deployment constraints.

## Acceptance Criteria
- [ ] FastAPI app has `/health`, `/ready`, `/predict`.
- [ ] Predict endpoint accepts thermal frame payload and returns depth + refined thermal.
- [ ] Dockerfile and compose skeleton exist.

## Files to Create
| File | Purpose |
|------|---------|
| `src/anima_thermal_slam/api.py` | FastAPI service |
| `docker/Dockerfile.serve` | inference container |
| `docker/docker-compose.serve.yml` | local compose stack |

## Test Plan
```bash
uv run python -m anima_thermal_slam.api
```

## References
- arXiv:2603.14998 §V
