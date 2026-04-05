# PRD-02: Core Model

> Module: Thermal-SLAM | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement T-RefNet + encoder-decoder + recurrent bottleneck (ConvGRU and RC) for thermal-to-depth prediction.

## Context (from paper)
Paper pipeline: T-RefNet enhances thermal input, encoder extracts features, recurrent module enforces temporal consistency, decoder predicts depth.

Paper references:
- Fig.1: full pipeline and T-RefNet block layout.
- §III-B/C: training flow, loss composition, recurrent options.

## Acceptance Criteria
- [ ] T-RefNet matches 4-layer conv+ReLU+sigmoid structure from paper figure.
- [ ] Model supports both single frame and sequence inputs.
- [ ] ConvGRU and RC mode are both available via config switch.
- [ ] Forward pass returns depth tensor with expected shape.

## Files to Create
| File | Purpose | Paper Ref |
|------|---------|-----------|
| `src/anima_thermal_slam/models/trefnet.py` | thermal refinement network | Fig.1 |
| `src/anima_thermal_slam/models/backbone.py` | encoder abstraction | §III-B |
| `src/anima_thermal_slam/models/recurrent.py` | ConvGRU + RC | §III-C |
| `src/anima_thermal_slam/models/decoder.py` | depth decoder | Fig.1 |
| `src/anima_thermal_slam/models/network.py` | integrated model | Fig.1, Alg.1 |
| `tests/test_model_forward.py` | shape and mode tests | — |

## Test Plan
```bash
uv run pytest tests/test_model_forward.py -v
```

## References
- arXiv:2603.14998 Fig.1, Algorithm 1, §III
- `repositories/RBs-thermal2depth/gru/DispNet.py`
- `repositories/RBs-thermal2depth/liquid_network/temporal_liquid.py`
