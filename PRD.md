# DEF-thermal-slam — Build Plan

## Objective
Deliver a paper-faithful implementation of Thermal Image Refinement with Depth
Estimation using Recurrent Networks for Monocular ORB-SLAM3 (Sahin et al., ICRA 2026).
Build complete scaffolding with real architecture code, configs, tests, and Docker
serving infrastructure.

## PRD Execution Board
| PRD | Title | Priority | Status | Notes |
|-----|-------|----------|--------|-------|
| PRD-01 | Foundation & Config | P0 | COMPLETE | pyproject.toml, configs, package structure |
| PRD-02 | Core Model | P0 | COMPLETE | T-RefNet, encoder, ConvGRU, RC-LIF, decoder |
| PRD-03 | Loss Functions | P0 | COMPLETE | SIlog, SSIM, ordinal, smoothness, composite |
| PRD-04 | Training Pipeline | P1 | COMPLETE | Dataset, dataloader, train loop, checkpointing |
| PRD-05 | Evaluation | P1 | COMPLETE | AbsRel, RMSE, delta metrics, eval script |
| PRD-06 | Export Pipeline | P1 | COMPLETE | ONNX export, safetensors, TRT hooks |
| PRD-07 | Integration | P1 | COMPLETE | Docker, serve.py, docker-compose |

## Constraints
- Upstream repo exists (github.com/hurkansah/RBs-thermal2depth) but paper omits
  many hyperparameters (optimizer, LR, epochs, batch size).
- Datasets (VIVID++, Kaggle thermal) are NOT downloaded yet.
- No training runs — build only.

## Definition of Done (MVP)
- [x] Package installs and imports cleanly
- [x] Model forward pass executes (T-RefNet + encoder + recurrent + decoder)
- [x] All 4 loss functions implemented with correct weights
- [x] Training script with config-driven hyperparameters
- [x] Evaluation script with all paper metrics
- [x] Dataset loader for thermal-depth pairs
- [x] ONNX export helper
- [x] Docker serving infrastructure
- [x] Tests pass (model shapes, loss computation)
