# Task Index — DEF-thermal-slam

## PRD-01: Foundation & Config
- [x] T01.1: Create pyproject.toml with hatchling, torch cu128, all deps
- [x] T01.2: Create src/thermal_slam/__init__.py
- [x] T01.3: Implement utils.py (config loader, seed setter, checkpoint manager)
- [x] T01.4: Create configs/paper.toml with all hyperparameters
- [x] T01.5: Create configs/debug.toml for smoke testing
- [x] T01.6: Create anima_module.yaml

## PRD-02: Core Model
- [x] T02.1: Implement T-RefNet (16-bit thermal -> normalized + colormap)
- [x] T02.2: Implement EfficientNet-B0 encoder with multi-scale features
- [x] T02.3: Implement MobileNetV2 encoder variant
- [x] T02.4: Implement ConvGRU recurrent block (~800K params)
- [x] T02.5: Implement RC-LIF reservoir computing block (~50K params)
- [x] T02.6: Implement UpProjection depth decoder with skip connections
- [x] T02.7: Implement ThermalDepthNet (full pipeline: T-RefNet -> encoder -> RB -> decoder)
- [x] T02.8: Verify forward pass shape (B,1,512,640) -> (B,1,512,640)

## PRD-03: Loss Functions
- [x] T03.1: Implement SIlog loss (Eigen et al. 2014)
- [x] T03.2: Implement SSIM loss (Godard et al. 2019)
- [x] T03.3: Implement ordinal depth loss (Xian et al. 2020)
- [x] T03.4: Implement edge-aware smoothness loss (Xu et al. 2022)
- [x] T03.5: Implement CompositeDepthLoss with configurable weights
- [x] T03.6: Verify all losses produce valid gradients

## PRD-04: Training Pipeline
- [x] T04.1: Implement ThermalDepthDataset (16-bit thermal + depth loading)
- [x] T04.2: Implement data augmentation (flip, crop, brightness jitter)
- [x] T04.3: Implement train/val/test split with saved indices
- [x] T04.4: Implement training loop with AMP, gradient clipping
- [x] T04.5: Implement warmup + cosine LR scheduler
- [x] T04.6: Implement checkpoint manager (top-k, best.pth)
- [x] T04.7: Implement early stopping
- [x] T04.8: Implement NaN detection
- [x] T04.9: Create scripts/train.py CLI entry point

## PRD-05: Evaluation
- [x] T05.1: Implement AbsRel, SqRel, RMSE, RMSE_log metrics
- [x] T05.2: Implement delta accuracy (a1, a2, a3) metrics
- [x] T05.3: Implement eval loop over test set
- [x] T05.4: JSON metrics output
- [x] T05.5: Create scripts/evaluate.py CLI entry point

## PRD-06: Export Pipeline
- [x] T06.1: Implement ONNX export (opset 17, dynamic batch)
- [x] T06.2: Implement safetensors save/load
- [x] T06.3: Add TRT export hooks (via shared toolkit)

## PRD-07: Integration
- [x] T07.1: Create Dockerfile.serve (3-layer pattern)
- [x] T07.2: Create docker-compose.serve.yml (profiles: serve, api, test)
- [x] T07.3: Implement serve.py with FastAPI (health, ready, predict)
- [x] T07.4: Create .env.serve
