# 31_Thermal-SLAM

## Module Identity
- Module: `DEF-thermal-slam`
- Scope: Thermal depth estimation with recurrent networks for monocular ORB-SLAM3
- Primary source: `papers/2603.14998.pdf` — Sahin et al., ICRA 2026
- Upstream repo: https://github.com/hurkansah/RBs-thermal2depth

## Paper Summary

The paper presents a framework for autonomous navigation in GPS-denied, visually
degraded environments using thermal (LWIR) imagery on UAVs. It consists of:

1. **T-RefNet** — Thermal Refinement Network that takes 16-bit raw thermal frames and
   produces (a) normalized thermal images for depth estimation and (b) 8-bit color-mapped
   images for ORB feature extraction in SLAM.

2. **Depth Estimation Network** — Lightweight encoder (EfficientNet-B0 / MobileNet /
   ResNet-8) paired with a recurrent block (ConvGRU or Reservoir Computing with LIF
   neurons) and an up-projection decoder to produce dense metric depth maps.

3. **ORB-SLAM3 Integration** — Refined thermal images + predicted depth feed into
   monocular ORB-SLAM3 for thermal-only localization in darkness.

## Architecture

```
16-bit thermal frame
       |
  [ T-RefNet ]
       |--- normalized thermal --> [ Encoder (EfficientNet-B0) ]
       |--- 8-bit colormap ------> [ ORB-SLAM3 ]
                                          |
                                   [ Recurrent Block (ConvGRU / RC-LIF) ]
                                          |
                                   [ Decoder (UpProjection) ]
                                          |
                                   Dense depth map
```

### Encoder Variants
| Backbone | Params | Notes |
|----------|--------|-------|
| EfficientNet-B0 | ~5.3M | Best accuracy on VIVID++ |
| MobileNetV2 | ~3.4M | Lightweight alternative |
| ResNet-8 | ~0.08M | Ultra-light |

### Recurrent Block Variants
| Block | Params | Notes |
|-------|--------|-------|
| ConvGRU | ~800K | Standard gated recurrent unit (convolutional) |
| RC-LIF | ~50K | Reservoir Computing with leaky-integrate-and-fire neurons |

## Loss Functions

Total loss: L = 0.9 * L_SIlog + 0.4 * L_SSIM + 0.1 * L_ord + 0.1 * L_sm

| Loss | Weight | Source |
|------|--------|--------|
| Scale-Invariant Log (SIlog) | 0.9 | Eigen et al. 2014 |
| Structural Similarity (SSIM) | 0.4 | Godard et al. 2019 |
| Ordinal Depth (ord) | 0.1 | Xian et al. 2020 |
| Edge-Aware Smoothness (sm) | 0.1 | Xu et al. 2022 |

## Hyperparameters (paper + reasonable defaults)
- Optimizer: AdamW (paper does not specify; standard for depth estimation)
- Learning rate: 1e-4 (with cosine decay + 5% warmup)
- Batch size: auto (L4 target 65% VRAM)
- Input resolution: 640x512 (FLIR Boson+)
- Precision: bf16 mixed on CUDA
- Gradient clipping: max_norm=1.0
- Epochs: 100 (with early stopping patience=20)

## Datasets
- **VIVID++** — Radiometric thermal indoor-dark sequences with depth GT
- **Custom Non-Radiometric** — FLIR Boson+ 640x512, ~65K samples, diverse lighting
  - Kaggle: bab5526b08d7023b20b28947c31b9e8f9f71d9ec6fc6aa55debcce221a08c305

## Evaluation Metrics
| Metric | Best (VIVID++) | Best (Non-Rad) |
|--------|----------------|----------------|
| AbsRel | 0.063 | 0.076 |
| RMSE | 0.298 | 0.439 |
| delta < 1.25 (a1) | 0.940 | 0.929 |
| delta < 1.25^2 (a2) | 0.980 | 0.965 |
| delta < 1.25^3 (a3) | 0.993 | 0.981 |
| SLAM trajectory error | <0.4m | — |

## Local Conventions
- Python package root: `src/thermal_slam/`
- Configs: `configs/*.toml`
- CLIs: `scripts/train.py`, `scripts/evaluate.py`
- Tests: `tests/`
