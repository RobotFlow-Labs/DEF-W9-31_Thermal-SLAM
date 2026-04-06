# ASSETS.md — DEF-thermal-slam Asset Inventory

## Datasets

### VIVID++ Indoor-Dark (Radiometric Thermal)
- **Description**: Radiometric thermal sequences with aligned depth ground truth
- **Usage**: Primary evaluation benchmark (Table I in paper)
- **Source**: VIVID++ dataset (academic release)
- **Size**: ~5-10GB (estimated)
- **Path**: `/mnt/forge-data/datasets/vivid_pp/`
- **Status**: NOT DOWNLOADED
- **Download**: Requires academic access; check VIVID++ project page

### Custom Non-Radiometric Thermal Dataset
- **Description**: FLIR Boson+ 640x512 non-radiometric thermal + depth pairs, ~65K samples
- **Usage**: Training and evaluation (Table II in paper)
- **Source**: Kaggle (public)
- **URL**: https://kaggle.com/datasets/bab5526b08d7023b20b28947c31b9e8f9f71d9ec6fc6aa55debcce221a08c305
- **Size**: ~15-30GB (estimated for 65K 640x512 16-bit frames + depth)
- **Path**: `/mnt/forge-data/datasets/thermal_slam_nonrad/`
- **Status**: NOT DOWNLOADED

## Pretrained Models / Backbones

### EfficientNet-B0 (ImageNet pretrained)
- **Usage**: Default encoder backbone
- **Source**: torchvision / timm
- **Path**: Auto-downloaded by torchvision on first use (cached in torch hub)
- **Size**: ~20MB
- **Status**: Available via torchvision

### MobileNetV2 (ImageNet pretrained)
- **Usage**: Alternative lightweight encoder
- **Source**: torchvision / timm
- **Path**: Auto-downloaded by torchvision on first use
- **Size**: ~14MB
- **Status**: Available via torchvision

## Shared Infrastructure

### CUDA Kernels (from MAP.md)
- **Depth projection + Z-buffer** (5.4x) — may be useful for depth rendering
- **Fused grid warp + sample** (43.5x) — useful for depth warping
- **Fused image preprocess** — normalize, augment, resize
- Install: `uv pip install /mnt/forge-data/shared_infra/cuda_extensions/wheels_py311_cu128/*.whl`

### Pre-computed Caches
- No directly applicable thermal dataset caches exist in shared_infra
- KITTI depth cache exists but is RGB-based, not thermal

## Output Paths
| Type | Path |
|------|------|
| Checkpoints | `/mnt/artifacts-datai/checkpoints/DEF-thermal-slam/` |
| Logs | `/mnt/artifacts-datai/logs/DEF-thermal-slam/` |
| TensorBoard | `/mnt/artifacts-datai/tensorboard/DEF-thermal-slam/` |
| Exports | `/mnt/artifacts-datai/exports/DEF-thermal-slam/` |
| Reports | `/mnt/artifacts-datai/reports/DEF-thermal-slam/` |

## Downloads Needed Before Training
1. **Kaggle thermal dataset** (~15-30GB) — `kaggle datasets download -d <id>` or manual
2. **VIVID++ indoor-dark** — check academic access
