# PRD-04: Evaluation

> Module: Thermal-SLAM | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement reproducible depth and trajectory metric evaluation aligned with paper tables/scenarios.

## Context (from paper)
Paper reports depth metrics on VIVID++ and custom non-radiometric set (AbsRel, RMSE, a1/a2/a3), plus trajectory error in bright/dark/UAV scenarios.

Paper references:
- Table I, Table II: depth metrics.
- §IV-C and Figs.6-8: trajectory evaluation scenarios.

## Acceptance Criteria
- [ ] Depth metric evaluator outputs AbsRel, RMSE, MAE, a1, a2, a3.
- [ ] Trajectory evaluator supports mean Euclidean position error.
- [ ] Evaluation report template compares paper vs current run.

## Files to Create
| File | Purpose | Paper Ref |
|------|---------|-----------|
| `src/anima_thermal_slam/eval.py` | metric computation and reporting | Table I/II, §IV |
| `docs/eval_report_template.md` | standard run report | §IV |
| `tests/test_losses.py` | loss and metric sanity checks | §III-B |

## Test Plan
```bash
uv run pytest tests/test_losses.py -v
```

## References
- arXiv:2603.14998 Table I, Table II, §IV
