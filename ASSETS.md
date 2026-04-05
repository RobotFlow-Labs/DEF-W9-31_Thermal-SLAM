# Thermal-SLAM — Asset Manifest

## Paper
- Title: Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3
- ArXiv: 2603.14998
- Authors: Hürkan Şahin, Huy Xuan Pham, Van Huyen Dang, Alper Yegenoglu, Erdal Kayacan
- URL: https://arxiv.org/abs/2603.14998

## Status: ALMOST

## Reference Repositories
| Repo | Purpose | URL | Local Path | Status |
|------|---------|-----|-----------|--------|
| RBs-thermal2depth | Paper reference implementation (ConvGRU + Liquid/RC variants, training/testing scripts) | https://github.com/hurkansah/RBs-thermal2depth | repositories/RBs-thermal2depth | DONE |
| flir_boson | Camera driver/acquisition stack mentioned by authors | https://github.com/hurkansah/flir_boson | N/A | OPTIONAL |
| ORB-SLAM3 | Target SLAM backend | https://github.com/UZ-SLAMLab/ORB_SLAM3 | N/A | MISSING |

## Datasets
| Dataset | Split/Scope | Source | Expected Path on Server | Status |
|---------|-------------|--------|-------------------------|--------|
| VIVID++ indoor-dark subset | Depth benchmark (radiometric thermal) | https://arxiv.org/abs/2204.06183 | /mnt/forge-data/datasets/vividpp | MISSING |
| Custom non-radiometric thermal-depth dataset | ~65k samples; bright/dark/semi-lit; hot/cold objects | https://kaggle.com/datasets/bab5526b08d7023b20b28947c31b9e8f9f71d9ec6fc6aa55debcce221a08c305 | /mnt/forge-data/datasets/thermal_depth_orbslam3 | MISSING |

## Pretrained Weights
| Model | Source | Path on Server | Status |
|------|--------|----------------|--------|
| EfficientNet-B0 encoder weights | timm pretrained | /mnt/forge-data/models/timm/efficientnet_b0 | MISSING |
| MobileNet / ResNet alternatives | timm pretrained | /mnt/forge-data/models/timm/ | MISSING |
| Module checkpoints (best_model_ema.pt) | reference repo training output | /mnt/artifacts-datai/checkpoints/thermal_slam/ | TODO |

## Hyperparameters (paper + reference scripts)
| Param | Value | Source |
|------|-------|--------|
| Input thermal format | 16-bit LWIR -> normalized; 8-bit color-mapped for ORB features | Paper Fig.1 + §III |
| Sequence model | ConvGRU or Reservoir Computing (RC) block | Paper Fig.1 + §III |
| Composite loss weights | Silog 0.9, SSIM 0.4, ordering 0.1, smoothness 0.1 | Paper §III-B |
| Encoder options | EfficientNet-B0, MobileNet, ResNet-8 | Paper §III-B + Table I/II |
| Sequence length (reference) | 5 | reference `train.sh` |
| Training epochs (reference) | 400 | reference `train.sh` |
| LR (reference) | 2e-4 with cosine warm restarts | reference `train.sh` + `train.py` |
| Batch size (reference) | 8 | reference `train.sh` |

## Expected Metrics (paper targets)
| Benchmark | Metric | Reported | Target |
|-----------|--------|----------|--------|
| VIVID++ indoor-dark | AbsRel | 0.063 (Eff-B0+GRU) | <= 0.07 |
| VIVID++ indoor-dark | RMSE | 0.298 (Eff-B0+GRU) | <= 0.32 |
| VIVID++ indoor-dark | a1 | 0.940 (Eff-B0+GRU) | >= 0.93 |
| Non-radiometric indoor (custom) | AbsRel | 0.076 (Eff-B0+RC) | <= 0.09 |
| Non-radiometric indoor (custom) | a1 | 0.929 (Eff-B0+RC) | >= 0.91 |
| Thermal-only ORB-SLAM3 | Mean trajectory error (corridor UAV) | < 0.4 m | <= 0.4 m |

## Notes
- `papers/2501.14584.pdf` is unrelated to this module and should not be used for implementation decisions.
- `papers/2603.14998.pdf` is now downloaded and used as the source of truth.
