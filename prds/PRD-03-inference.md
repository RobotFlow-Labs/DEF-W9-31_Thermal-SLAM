# PRD-03: Inference Pipeline

> Module: Thermal-SLAM | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Provide an inference CLI that refines thermal frames, predicts depth, and exports ORB-SLAM-ready thermal/depth artifacts.

## Context (from paper)
Refined 8-bit thermal output is used for ORB feature extraction while depth map provides metric priors.

Paper references:
- Fig.1: enhanced thermal output + depth output path to ORB-SLAM3.
- §IV-C: localization behavior depends on stable feature extraction.

## Acceptance Criteria
- [ ] CLI consumes image folder or sequence list.
- [ ] Exports depth map and ORB-friendly thermal visualization.
- [ ] Supports checkpoint loading and model mode switches.
- [ ] Includes smoke test with synthetic input.

## Files to Create
| File | Purpose | Paper Ref |
|------|---------|-----------|
| `src/anima_thermal_slam/slam_adapter.py` | ORB-friendly preprocessing adapter | Fig.1, §IV-C |
| `src/anima_thermal_slam/infer.py` | CLI inference entrypoint | §IV |
| `scripts/run_infer.sh` | runnable wrapper | — |
| `tests/test_slam_adapter.py` | adapter output tests | — |

## Test Plan
```bash
uv run pytest tests/test_slam_adapter.py -v
uv run python -m anima_thermal_slam.infer --help
```

## References
- arXiv:2603.14998 Fig.1, §IV-C
