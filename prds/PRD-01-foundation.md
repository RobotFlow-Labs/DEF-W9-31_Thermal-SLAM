# PRD-01: Foundation & Config

## Objective
Set up project structure, package scaffolding, TOML configs, and build infrastructure.

## Deliverables
- [x] `pyproject.toml` with hatchling backend, torch cu128
- [x] `src/thermal_slam/__init__.py` with version
- [x] `src/thermal_slam/utils.py` — config loading, seeding, checkpoint manager
- [x] `configs/paper.toml` — paper-faithful hyperparameters
- [x] `configs/debug.toml` — quick smoke test config
- [x] `anima_module.yaml` — module manifest

## Config Contract
All hyperparameters in TOML. Training script reads config, never hardcodes values.
Supports `--config` and `--resume` CLI flags.

## Acceptance Criteria
- `uv sync` succeeds
- `python -c "import thermal_slam"` works
- Config loads without error
