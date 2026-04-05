# PRD-06: ROS2 Integration

> Module: Thermal-SLAM | Priority: P1
> Depends on: PRD-05
> Status: ⬜ Not started

## Objective
Add ROS2-facing node wrapper for thermal frame subscription and depth/refined-output publication.

## Context (from paper)
UAV and handheld experiments rely on online sequence processing and SLAM ingestion.

Paper references:
- §IV-C: trajectory experiments with handheld/UAV sequences.

## Acceptance Criteria
- [ ] ROS2 node skeleton subscribes thermal topic and publishes depth topic.
- [ ] Node uses same preprocessing/model path as CLI inference.
- [ ] Clear TODOs for deployment-specific message contracts.

## Files to Create
| File | Purpose |
|------|---------|
| `src/anima_thermal_slam/ros2_node.py` | ROS2 wrapper node |
| `docs/ros2_topics.md` | topic contract doc |

## Test Plan
```bash
uv run python -m anima_thermal_slam.ros2_node --help
```

## References
- arXiv:2603.14998 §IV-C
