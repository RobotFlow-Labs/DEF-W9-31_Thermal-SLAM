# Benchmarks

This folder stores reproducible benchmark outputs for Thermal-SLAM.

Suggested workflow:

```bash
uv run python scripts/profile_infer.py --iters 100 --warmup 20
```

Record:
- Device and CUDA/cuDNN versions
- Input resolution and sequence length
- Average forward latency (ms)
- Throughput (fps)

