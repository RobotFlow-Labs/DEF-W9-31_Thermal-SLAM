# NEXT_STEPS.md
> Last updated: 2026-04-06
> MVP Readiness: 75%

## Done
- [x] Paper analyzed (arxiv 2603.14998)
- [x] CLAUDE.md, ASSETS.md, PRD.md — all docs complete
- [x] All 7 PRDs complete (foundation → integration)
- [x] anima_module.yaml, pyproject.toml, configs/
- [x] src/thermal_slam/ — full package with CUDA-accelerated pipeline
  - model.py (T-RefNet + EfficientNet-B0 + ConvGRU/RC-LIF + UpProjection)
  - dataset.py (VIVIDPlusPlusDataset + ThermalDepthDataset)
  - train_cu.py (CUDA training + structured logging + VRAM monitoring)
  - cuda_ops.py (JIT-compiled shared depth estimation kernels)
  - losses.py (SIlog + SSIM + Ordinal + Smoothness composite)
  - evaluate.py (AbsRel, RMSE, delta metrics + VIVID++ support)
  - serve.py (FastAPI with input validation + security hardening)
  - utils.py (config, scheduler, checkpoint manager, ONNX export)
- [x] tests/ — 25 tests PASS, ruff lint clean
- [x] Code review — all 2C, 5H, 9M issues fixed
- [x] .venv (Python 3.11, torch 2.11.0+cu128)
- [x] VIVID++ verified: 78K thermal-depth pairs (59K/9.6K/8.9K)
- [x] Batch size profiling: BS=128 → 87% VRAM (19.9GB/23GB)
- [x] Training launched on GPU 6 with nohup+disown
- [12:45] Step 50/464 — loss=0.76, decreasing normally

## In Progress
- [ ] Training: 100 epochs on GPU 6, ETA ~12h
  - PID: see /mnt/artifacts-datai/logs/DEF-thermal-slam/train.pid
  - Log: /mnt/artifacts-datai/logs/DEF-thermal-slam/train_20260406_1243.log
  - Monitor: `tail -f /mnt/artifacts-datai/logs/DEF-thermal-slam/train_20260406_1243.log`
- [ ] Custom CUDA kernel building (thermal_depth_ops — differentiable)

## TODO (after training completes)
- [ ] Evaluate on VIVID++ test split
- [ ] Generate TRAINING_REPORT.md
- [ ] ONNX export
- [ ] TensorRT FP16 + FP32 export (MANDATORY)
- [ ] Push to HuggingFace: ilessio-aiflowlab/DEF-thermal-slam
- [ ] Add more tests (eval, serve, utils, training smoke)

## Code Review Summary (2026-04-06)
Score: 68/100 → all findings fixed in commit ae9c0ba
- 2 CRITICAL: dataset __len__ silent noise, weights_only=False → FIXED
- 5 HIGH: depth clipping, SSIM masking, ordinal init, serve validation → FIXED
- 9 MEDIUM: scheduler resume, persistent workers, eval format, val loss → FIXED

## Training Config
- BS=128, 100 epochs, bf16, AdamW lr=1e-4
- 464 steps/epoch, ~7 min/epoch, ETA ~12h
- VRAM: 19.9GB/23GB (87%)
- Loss: 0.9*SIlog + 0.4*SSIM + 0.1*Ord + 0.1*Sm
