#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from anima_thermal_slam.models import ThermalDepthNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Thermal-SLAM model forward latency")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalDepthNet().to(device).eval()
    x = torch.randn(1, args.seq_len, 1, args.height, args.width, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x, decode_all=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x, decode_all=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters

    print(f"device={device.type} avg_forward_ms={dt * 1000:.3f}")


if __name__ == "__main__":
    main()
