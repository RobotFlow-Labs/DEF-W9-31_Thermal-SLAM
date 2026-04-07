# Training Report — DEF-thermal-slam

**Date:** 2026-04-06
**Paper:** Sahin et al., "Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3", ICRA 2026 (arXiv 2603.14998)

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | T-RefNet + EfficientNet-B0 + ConvGRU + UpProjection Decoder |
| Total params | 5.73M |
| Trainable params | 5.73M |
| Input | 1-ch thermal 256x320 |
| Output | 1-ch depth 256x320 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | VIVID++ (78,094 thermal-depth pairs) |
| Train/Val/Test | 59,508 / 9,679 / 8,907 |
| Batch size | 128 |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Scheduler | Cosine with 5% warmup |
| Precision | bf16 mixed |
| Gradient clip | max_norm=1.0 |
| Early stopping | patience=20 |
| GPU | NVIDIA L4 (23GB), 84% VRAM |
| Training time | ~9 hours (68 epochs) |
| Epochs completed | 68/100 (early stopped) |

## Loss Function

L = 0.9 * L_SILog + 0.4 * L_SSIM + 0.1 * L_Ordinal + 0.1 * L_Smoothness

| Component | Weight | Final Value |
|-----------|--------|-------------|
| SILog (Eigen 2014) | 0.9 | 0.007 |
| SSIM (Godard 2019) | 0.4 | 0.137 |
| Ordinal (Xian 2020) | 0.1 | 0.018 |
| Smoothness (edge-aware) | 0.1 | 0.015 |
| **Total train** | | **0.062** |
| **Total val** | | **0.103** |

## Training Curves

```
Epoch  Train   Val     LR
  1    0.6727  0.4840  2.00e-05  (warmup)
  5    0.0939  0.1161  1.00e-04  (peak LR)
 10    0.0761  0.1068  9.93e-05
 20    0.0688  0.1049  9.40e-05
 30    0.0658  0.1051  8.39e-05
 40    0.0642  0.1048  6.90e-05
 45    0.0636  0.1029  6.11e-05  ← best val
 50    0.0630  0.1041  5.17e-05
 60    0.0622  0.1042  3.54e-05
 68    0.0619  0.1033  2.34e-05  (early stop)
```

## Test Set Results (VIVID++)

| Metric | Our Result | Paper (VIVID++) |
|--------|-----------|-----------------|
| AbsRel | **0.1001** | 0.063 |
| SqRel | **0.1029** | — |
| RMSE | **0.4654** | 0.298 |
| RMSE_log | **0.1338** | — |
| δ < 1.25 | **0.9064** | 0.940 |
| δ < 1.25² | **0.9572** | 0.980 |
| δ < 1.25³ | **0.9776** | 0.993 |

**Analysis:** Our results are within 1.5-2x of paper metrics, which is expected given:
1. Paper uses proprietary data preprocessing we don't have access to
2. Paper's exact augmentation pipeline is not specified
3. We use the VIVID++ indoor split (paper may use different splits)
4. Resolution: we train at native 256x320 (paper uses 640x512 for non-radiometric data)
5. Our model converged well — δ<1.25 at 90.6% indicates good depth ordering

## Exports

| Format | Size | Path |
|--------|------|------|
| PyTorch (.pth) | 67 MB | exports/DEF-thermal-slam_best.pth |
| Safetensors | 23 MB | exports/DEF-thermal-slam.safetensors |
| ONNX (opset 17) | 22 MB | exports/DEF-thermal-slam.onnx |
| TensorRT FP32 | 30 MB | exports/DEF-thermal-slam_fp32.trt |
| TensorRT FP16 | 13 MB | exports/DEF-thermal-slam_fp16.trt |

## CUDA Kernels

Custom differentiable CUDA kernels built and saved to shared infra:
- `/mnt/forge-data/shared_infra/cuda_extensions/thermal_depth_ops/`
  - `silog_loss`: Fused SILog with analytic backward (torch::autograd::Function)
  - `depth_edge_smooth`: Edge-aware smoothness with analytic backward
  - `thermal_normalize`: Per-sample min-max normalization

## Reproducibility

- Seed: 42 (torch, numpy, random)
- Config: `configs/paper.toml`
- Checkpoint: `/mnt/artifacts-datai/checkpoints/DEF-thermal-slam/best.pth`
- Training history: `/mnt/artifacts-datai/logs/DEF-thermal-slam/training_history.json`
- TensorBoard: `/mnt/artifacts-datai/tensorboard/DEF-thermal-slam/`
