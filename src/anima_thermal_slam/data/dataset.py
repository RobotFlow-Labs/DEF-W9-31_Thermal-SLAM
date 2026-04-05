from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _list_sorted_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def _list_sorted_npy(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.npy"))


def _read_thermal(path: Path, percentile: float | None = None, clahe_clip: float | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    arr = img.astype(np.float32)
    maxv = 16383.0 if arr.max() <= 16383.0 else 65535.0
    arr = np.clip(arr / maxv, 0.0, 1.0)

    if percentile is not None:
        lo = np.percentile(arr, percentile)
        hi = np.percentile(arr, 100.0 - percentile)
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    if clahe_clip is not None and clahe_clip > 0:
        arr8 = (arr * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        arr8 = clahe.apply(arr8)
        arr = arr8.astype(np.float32) / 255.0

    return arr


def _read_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _resize_pair(thermal: np.ndarray, depth: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    t = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_AREA)
    d = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return t, d


@dataclass
class _ClipIndex:
    seq_path: Path
    start: int


class SequenceThermalDepthDataset(Dataset[Any]):
    """Sequence dataset compatible with paper and reference implementation format.

    Expected structure under each sequence folder:
    - Thermal/*.png|jpg
    - Depth_T/*.npy
    """

    def __init__(
        self,
        root_dir: str,
        list_file: str,
        seq_len: int = 5,
        stride: int = 1,
        out_h: int = 256,
        out_w: int = 320,
        return_last_only: bool = True,
        use_percentile_stretch: bool = False,
        percentile: float = 5.0,
        use_clahe: bool = False,
        clahe_clip: float = 3.0,
        augment_flip: bool = False,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.return_last_only = bool(return_last_only)
        self.use_percentile_stretch = bool(use_percentile_stretch)
        self.percentile = float(percentile)
        self.use_clahe = bool(use_clahe)
        self.clahe_clip = float(clahe_clip)
        self.augment_flip = bool(augment_flip)
        self.rng = random.Random(seed)

        seq_names = [
            ln.strip()
            for ln in Path(list_file).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]

        self.index: list[_ClipIndex] = []
        for seq_name in seq_names:
            seq_path = self.root_dir / seq_name
            thermal_files = _list_sorted_images(seq_path / "Thermal")
            depth_files = _list_sorted_npy(seq_path / "Depth_T")
            n = min(len(thermal_files), len(depth_files))
            if n == 0:
                continue
            max_start = n - (self.seq_len - 1) * self.stride
            for start in range(max_start):
                self.index.append(_ClipIndex(seq_path=seq_path, start=start))

        if not self.index:
            raise RuntimeError("No clips found. Check root_dir/list_file and folder layout.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip = self.index[idx]
        thermal_files = _list_sorted_images(clip.seq_path / "Thermal")
        depth_files = _list_sorted_npy(clip.seq_path / "Depth_T")
        n = min(len(thermal_files), len(depth_files))

        inds = [clip.start + i * self.stride for i in range(self.seq_len)]
        if inds[-1] >= n:
            raise IndexError("Clip index overflow")

        thermal_stack: list[np.ndarray] = []
        depth_stack: list[np.ndarray] = []
        mask_stack: list[np.ndarray] = []

        for i in inds:
            thermal = _read_thermal(
                thermal_files[i],
                percentile=(self.percentile if self.use_percentile_stretch else None),
                clahe_clip=(self.clahe_clip if self.use_clahe else None),
            )
            depth = _read_depth(depth_files[i])
            thermal, depth = _resize_pair(thermal, depth, self.out_h, self.out_w)
            mask = (depth > 0).astype(np.float32)

            thermal_stack.append(thermal[None, ...])
            depth_stack.append(depth)
            mask_stack.append(mask)

        thermal_seq = np.stack(thermal_stack, axis=0)  # (T,1,H,W)
        depth_seq = np.stack(depth_stack, axis=0)  # (T,H,W)
        mask_seq = np.stack(mask_stack, axis=0)  # (T,H,W)

        if self.augment_flip and self.rng.random() < 0.5:
            thermal_seq = thermal_seq[..., ::-1].copy()
            depth_seq = depth_seq[..., ::-1].copy()
            mask_seq = mask_seq[..., ::-1].copy()

        sample: dict[str, torch.Tensor] = {
            "thermal_seq": torch.from_numpy(thermal_seq.astype(np.float32)),
        }
        if self.return_last_only:
            sample["depth"] = torch.from_numpy(depth_seq[-1].astype(np.float32))
            sample["mask"] = torch.from_numpy(mask_seq[-1].astype(np.float32))
        else:
            sample["depth_seq"] = torch.from_numpy(depth_seq.astype(np.float32))
            sample["mask_seq"] = torch.from_numpy(mask_seq.astype(np.float32))
        return sample
