# PRD-07: Production Hardening

> Module: Thermal-SLAM | Priority: P2
> Depends on: PRD-04, PRD-05, PRD-06
> Status: ⬜ Not started

## Objective
Prepare module for CUDA-server optimization/export and production observability.

## Context (from paper)
Paper emphasizes real-time constraints, robustness under severe conditions, and future embedded optimization.

Paper references:
- §V Future Work: improve robustness against NUC artifacts and optimize pipeline for embedded platforms.

## Acceptance Criteria
- [ ] Export path documented (PyTorch -> ONNX -> TRT fp16/fp32).
- [ ] Runtime profiling hooks added for depth + adapter stages.
- [ ] Failure mode handling documented (low features, NUC artifacts).

## Files to Create
| File | Purpose |
|------|---------|
| `docs/production_plan.md` | deployment/hardening plan |
| `benchmarks/README.md` | benchmark instructions |
| `scripts/profile_infer.py` | stage timing utility |

## Test Plan
```bash
uv run python scripts/profile_infer.py --help
```

## References
- arXiv:2603.14998 §V
