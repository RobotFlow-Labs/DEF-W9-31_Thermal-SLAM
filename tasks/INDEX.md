# Thermal-SLAM Task Index — 15 Tasks

## Build Order
| Task | Title | Depends | Status |
|------|-------|---------|--------|
| PRD-0101 | Project scaffold and package bootstrapping | None | ⬜ |
| PRD-0102 | Config models and parser | PRD-0101 | ⬜ |
| PRD-0103 | Sequence dataset loader | PRD-0102 | ⬜ |
| PRD-0201 | T-RefNet module | PRD-0102 | ⬜ |
| PRD-0202 | Encoder abstraction and feature pyramid | PRD-0201 | ⬜ |
| PRD-0203 | ConvGRU + RC recurrent blocks | PRD-0202 | ⬜ |
| PRD-0204 | Depth decoder and integrated network | PRD-0203 | ⬜ |
| PRD-0205 | Composite losses + temporal consistency | PRD-0204 | ⬜ |
| PRD-0301 | ORB-SLAM adapter utilities | PRD-0204 | ⬜ |
| PRD-0302 | Inference CLI pipeline | PRD-0301 | ⬜ |
| PRD-0401 | Depth and trajectory evaluation metrics | PRD-0205 | ⬜ |
| PRD-0501 | FastAPI service endpoints | PRD-0302 | ⬜ |
| PRD-0502 | Docker serving artifacts | PRD-0501 | ⬜ |
| PRD-0601 | ROS2 node skeleton | PRD-0302 | ⬜ |
| PRD-0701 | Profiling + production hardening docs | PRD-0401, PRD-0502, PRD-0601 | ⬜ |
