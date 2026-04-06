# PRD-03: Loss Functions

## Objective
Implement all four loss functions from the paper with correct weighting.

## Loss Functions

### 1. Scale-Invariant Log Loss (SIlog) — weight 0.9
From Eigen et al. 2014:
```
d_i = log(y_hat_i) - log(y_i)
L_SIlog = (1/n) * sum(d_i^2) - (lambda/n^2) * (sum(d_i))^2
```
where lambda=0.5 (variance reduction term).

### 2. Structural Similarity Loss (SSIM) — weight 0.4
From Godard et al. 2019:
```
L_SSIM = (1 - SSIM(y_hat, y)) / 2
```
Computed per-patch with 3x3 or 7x7 window.

### 3. Ordinal Depth Loss — weight 0.1
From Xian et al. 2020:
Enforces correct relative depth ordering between sampled pixel pairs.

### 4. Edge-Aware Smoothness Loss — weight 0.1
From Xu et al. 2022:
```
L_sm = |d_x(depth)| * exp(-|d_x(image)|) + |d_y(depth)| * exp(-|d_y(image)|)
```

### Composite
```
L_total = 0.9 * L_SIlog + 0.4 * L_SSIM + 0.1 * L_ord + 0.1 * L_sm
```

## Deliverables
- [x] `src/thermal_slam/losses.py` — all four losses + composite
- [x] Each loss tested independently with random tensors
- [x] Weights configurable via TOML

## Acceptance Criteria
- All losses produce valid gradients (no NaN)
- Composite loss weight sum matches paper
- Each loss can be disabled via config (weight=0)
