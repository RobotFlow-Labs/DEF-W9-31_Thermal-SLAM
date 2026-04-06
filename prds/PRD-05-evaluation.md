# PRD-05: Evaluation

## Objective
Implement all evaluation metrics from the paper and provide an evaluation script.

## Metrics (from Tables I and II)

### Depth Estimation Metrics
| Metric | Formula | Lower/Higher Better |
|--------|---------|---------------------|
| AbsRel | mean(|y-y_hat| / y) | Lower |
| SqRel | mean((y-y_hat)^2 / y) | Lower |
| RMSE | sqrt(mean((y-y_hat)^2)) | Lower |
| RMSE_log | sqrt(mean((log(y)-log(y_hat))^2)) | Lower |
| delta < 1.25 (a1) | % of max(y/y_hat, y_hat/y) < 1.25 | Higher |
| delta < 1.25^2 (a2) | % of max(...) < 1.5625 | Higher |
| delta < 1.25^3 (a3) | % of max(...) < 1.953 | Higher |

### Paper Targets
- VIVID++ indoor-dark: AbsRel=0.063, RMSE=0.298, a1=0.940
- Non-radiometric: AbsRel=0.076, RMSE=0.439, a1=0.929

## Deliverables
- [x] `src/thermal_slam/evaluate.py` — metric computation + eval loop
- [x] `scripts/evaluate.py` — CLI entry point
- [x] JSON output of all metrics

## Acceptance Criteria
- All 7 metrics compute correctly on synthetic data
- Evaluation script loads checkpoint and runs inference
- Results saved to `/mnt/artifacts-datai/reports/DEF-thermal-slam/`
