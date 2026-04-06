# PRD-06: Export Pipeline

## Objective
Export trained models to ONNX, safetensors, and TensorRT formats.

## Export Targets
| Format | Use Case | Tool |
|--------|----------|------|
| safetensors | HuggingFace upload, fast loading | safetensors library |
| ONNX | Cross-platform inference | torch.onnx.export |
| TensorRT FP16 | GPU edge deployment (Jetson) | shared_infra/trt_toolkit |
| TensorRT FP32 | GPU inference (accuracy) | shared_infra/trt_toolkit |

## Export Flow
1. Load best.pth checkpoint
2. Export to safetensors
3. Export to ONNX (opset 17, dynamic batch)
4. Convert ONNX to TRT FP16 + FP32 via shared toolkit
5. Push all to HuggingFace

## Deliverables
- [x] Export logic in `src/thermal_slam/utils.py`
- [x] ONNX export tested with dummy input

## Acceptance Criteria
- ONNX export produces valid .onnx file
- safetensors round-trip matches original weights
- Export paths: `/mnt/artifacts-datai/exports/DEF-thermal-slam/`
