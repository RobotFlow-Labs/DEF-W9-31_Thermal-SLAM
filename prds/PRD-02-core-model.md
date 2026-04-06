# PRD-02: Core Model

## Objective
Implement the full Thermal-SLAM architecture: T-RefNet, encoder backbones, recurrent
blocks (ConvGRU + RC-LIF), and depth decoder.

## Architecture Components

### T-RefNet (Thermal Refinement Network)
- Input: 16-bit raw thermal (1, H, W)
- Output: normalized thermal (1, H, W) + 8-bit colormap (3, H, W)
- Learnable normalization with histogram equalization-like processing
- Small convolutional network (3 conv layers + instance norm)

### Encoder
- EfficientNet-B0 (default), MobileNetV2, ResNet-8
- Multi-scale feature extraction at 4 levels
- ImageNet pretrained weights via timm

### Recurrent Block
- **ConvGRU**: ~800K params, standard convolutional GRU
- **RC-LIF**: ~50K params, reservoir computing with leaky-integrate-and-fire neurons
  - State update: x(t+1) = f(W_in * u(t+1) + W * x(t))
  - LIF: tau_m * dV/dt = -V(t) + R_m * I(t)

### Depth Decoder
- Up-projection blocks at 4 scales
- Skip connections from encoder
- Final sigmoid + depth scaling

## Deliverables
- [x] `src/thermal_slam/model.py` — all architecture components
- [x] Forward pass works with random input (1, 1, 512, 640)
- [x] Both ConvGRU and RC variants selectable via config

## Acceptance Criteria
- Model instantiates and runs forward pass without error
- Output depth map shape matches input spatial dimensions
- Parameter counts match paper (~5.3M encoder + ~800K GRU or ~50K RC)
