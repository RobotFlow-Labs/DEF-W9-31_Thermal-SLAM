# PRD-04: Training Pipeline

## Objective
Complete training loop with config-driven hyperparameters, checkpointing,
early stopping, and logging.

## Components

### Dataset
- `ThermalDepthDataset`: loads 16-bit thermal frames + depth GT
- Supports VIVID++ and custom non-radiometric formats
- Train/val/test split (90/5/5) with saved indices

### Training Loop
- AdamW optimizer with cosine annealing + 5% linear warmup
- bf16 mixed precision via torch.amp
- Gradient clipping (max_norm=1.0)
- Per-step logging: loss, lr, throughput
- Per-epoch: val_loss, val_metrics, checkpoint decision

### Checkpointing
- Save every 500 steps
- Keep top 2 by val_loss + separate best.pth
- Full state: model, optimizer, scheduler, epoch, step, metrics, config

### Early Stopping
- Patience: 20 epochs
- Min delta: 1e-4
- NaN detection with immediate stop

## Deliverables
- [x] `src/thermal_slam/dataset.py` — thermal-depth dataset loader
- [x] `src/thermal_slam/train.py` — full training loop
- [x] `scripts/train.py` — CLI entry point
- [x] Checkpoint save/load cycle works

## Acceptance Criteria
- Training loop runs 2 steps on synthetic data without error
- Checkpoint saves and resumes correctly
- Config drives all hyperparameters (nothing hardcoded)
