# NEXT_STEPS.md
> Last updated: 2026-04-06
> MVP Readiness: 65%

## Done
- [x] Paper analyzed (arxiv 2603.14998)
- [x] CLAUDE.md — paper summary, architecture, hyperparameters
- [x] ASSETS.md — dataset/model inventory
- [x] PRD.md — master build plan with 7 PRDs (all COMPLETE)
- [x] prds/ — 7 PRD files
- [x] anima_module.yaml — module manifest
- [x] pyproject.toml — hatchling build config (cu128)
- [x] configs/ — TOML training configs (paper.toml, debug.toml)
- [x] src/thermal_slam/ — full Python package
  - model.py (T-RefNet, EfficientNet encoder, ConvGRU, RC-LIF, DepthDecoder)
  - dataset.py (VIVIDPlusPlusDataset + ThermalDepthDataset)
  - train.py (config-driven training loop)
  - train_cu.py (CUDA-accelerated training with shared kernels)
  - cuda_ops.py (JIT-compiled depth estimation CUDA ops)
  - evaluate.py (AbsRel, RMSE, delta metrics)
  - losses.py (SIlog, SSIM, Ordinal, Smoothness, CompositeDepthLoss)
  - utils.py (config, seeding, checkpoint manager, WarmupCosine, ONNX export)
  - serve.py (FastAPI endpoints)
- [x] scripts/train.py, scripts/evaluate.py
- [x] tests/test_model.py, tests/test_dataset.py — 25 tests PASS
- [x] Dockerfile.serve + docker-compose.serve.yml
- [x] tasks/INDEX.md
- [x] .venv created (Python 3.11, torch cu128)
- [x] VIVID++ dataset verified (59K train, 9.6K val, 8.9K test)
- [x] CUDA smoke test PASS (model forward + backward + CUDA ops)
- [x] Batch size profiling: BS=128 → 16.6GB (72% of L4 23GB)
- [x] Ruff lint: PASS

## In Progress
- [ ] Full training run on VIVID++ data (need GPU assignment)

## TODO
- [ ] Full training on VIVID++ (100 epochs, BS=128, bf16)
- [ ] Evaluate on VIVID++ test split
- [ ] ONNX export
- [ ] TensorRT FP16 + FP32 export (MANDATORY)
- [ ] Push checkpoint to HuggingFace (ilessio-aiflowlab/DEF-thermal-slam)
- [ ] Generate TRAINING_REPORT.md

## Batch Size Results (L4 23GB, bf16)
| Batch Size | VRAM | % Used |
|-----------|------|--------|
| 64 | 8.4GB | 36% |
| 96 | 12.5GB | 54% |
| **128** | **16.6GB** | **72%** |
| 160 | 20.8GB | 90% |
| 192 | OOM | — |

**Optimal: BS=128 (72% VRAM)**

## Data Summary
- VIVID++: 78,094 total (59,508 train / 9,679 val / 8,907 test)
- Resolution: 320×256 (native VIVID++)
- Thermal: uint16 PNG → float32 [0,1] normalized
- Depth: float32 NPY, range 0-8m (clamped to max_depth=10m)

## Model Summary
- 5.73M parameters (EfficientNet-B0 + ConvGRU + UpProjection decoder)
- CUDA ops: fused SIlog loss + edge-aware gradient loss
