# Thermal-SLAM Production Plan

## Export Path
1. Train/eval PyTorch checkpoint (`best.pt`)
2. Export ONNX (`opset >= 17`, dynamic batch optional)
3. Build TensorRT FP16 engine
4. Build TensorRT FP32 fallback engine
5. Validate parity against PyTorch on holdout batch

## Runtime Hardening
- Add watchdog for feature-count collapse (dark/no-texture frames)
- Add NUC artifact detection and frame skip/debounce policy
- Add input sanity checks (saturation, dead frame, shape mismatch)
- Enable stage-wise latency telemetry (refine, encode, recurrent, decode, adapter)

## Deployment Checklist
- [ ] model artifact versioned
- [ ] dataset hash logged
- [ ] metrics meet ASSETS.md targets
- [ ] API `/ready` tied to model load success
- [ ] ROS2 topic QoS documented
