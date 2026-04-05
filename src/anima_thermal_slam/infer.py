from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .models import ThermalDepthNet
from .slam_adapter import ORBSLAM3ThermalAdapter


def _load_thermal(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    arr = img.astype(np.float32)
    maxv = 16383.0 if arr.max() <= 16383.0 else 65535.0
    arr = np.clip(arr / maxv, 0.0, 1.0)
    arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thermal depth inference CLI")
    p.add_argument("--input_dir", type=Path, required=False)
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--backbone", type=str, default="simple_cnn")
    p.add_argument("--recurrent", type=str, default="convgru")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--decode_all", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalDepthNet(backbone=args.backbone, recurrent=args.recurrent).to(device)
    model.eval()

    if args.checkpoint is not None and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    adapter = ORBSLAM3ThermalAdapter()

    if args.input_dir is None:
        print("No --input_dir provided. Exiting after CLI validation.")
        return

    files = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}])
    if len(files) < args.seq_len:
        raise RuntimeError(f"Need at least {args.seq_len} frames in input_dir")

    for i in range(args.seq_len - 1, len(files)):
        seq_paths = files[i - args.seq_len + 1 : i + 1]
        seq = []
        for p in seq_paths:
            t = _load_thermal(p, size=(args.width, args.height))
            seq.append(t)
        x = np.stack(seq, axis=0)[None, :, None, :, :]
        x_t = torch.from_numpy(x.astype(np.float32)).to(device)

        with torch.no_grad():
            out = model(x_t, decode_all=args.decode_all)

        depth = out.depth.squeeze(0).squeeze(0)
        refined = out.refined_last.squeeze(0).squeeze(0)

        depth_np = adapter.depth_to_float32(depth)
        thermal_u8 = adapter.to_orb_thermal_u8(refined)

        stem = seq_paths[-1].stem
        np.save(args.output_dir / f"{stem}_depth.npy", depth_np)
        cv2.imwrite(str(args.output_dir / f"{stem}_thermal_u8.png"), thermal_u8)

    print(f"Saved outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
