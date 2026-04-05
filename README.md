# ANIMA Thermal-SLAM (Module 31)

Paper-aligned baseline implementation for:

- Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3 (arXiv:2603.14998)

This repository contains:
- PRD suite (`prds/`) and implementation task graph (`tasks/`)
- essential baseline code under `src/anima_thermal_slam/`
- tests and deployment skeletons for API/ROS2 integration

## Quick start
```bash
uv run pytest
uv run python -m anima_thermal_slam.infer --help
```
