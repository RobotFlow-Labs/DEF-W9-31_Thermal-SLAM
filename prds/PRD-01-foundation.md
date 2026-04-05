# PRD-01: Foundation

> Module: Thermal-SLAM | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Establish a reproducible project skeleton with config, dataset IO, shared types, and smoke tests.

## Context (from paper)
Paper requires sequence-based thermal-depth training and thermal-only ORB-SLAM3 integration.

Paper references:
- §III Methodology: sequence thermal input, preprocessing, depth estimation.
- §IV-A: dual dataset evaluation (VIVID++ radiometric + custom non-radiometric).

## Acceptance Criteria
- [ ] `pyproject.toml` and importable package exist.
- [ ] Config loader supports model/data/train/inference sections.
- [ ] Sequence dataset loader yields `thermal_seq`, `depth`, and `mask`.
- [ ] Unit tests pass for config and dataset indexing.

## Files to Create
| File | Purpose | Paper Ref |
|------|---------|-----------|
| `pyproject.toml` | dependency + scripts | §IV setup implied |
| `configs/default.toml` | baseline config | §III/§IV |
| `src/anima_thermal_slam/config.py` | typed config models | §III |
| `src/anima_thermal_slam/data/dataset.py` | sequence thermal-depth dataset | §IV-A |
| `tests/test_config.py` | config parser tests | — |
| `tests/test_dataset.py` | dataset construction tests | — |

## Test Plan
```bash
uv run pytest tests/test_config.py tests/test_dataset.py -v
```

## References
- arXiv:2603.14998 §III, §IV-A
- `repositories/RBs-thermal2depth/gru/dataset.py`
