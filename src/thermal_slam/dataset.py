"""Dataset loader for thermal-depth pairs.

Supports:
  - VIVID++ format: sequences with Thermal/*.png + Depth_T/*.npy
  - Generic format: thermal/ + depth/ directory pairs
  - 16-bit thermal images (.png, .tiff, .npy)
  - Aligned depth ground truth (.npy, .png)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VIVIDPlusPlusDataset(Dataset):
    """VIVID++ thermal depth dataset.

    Directory structure:
        root/dataset/train/
          {sequence_name}/
            Thermal/    ← 16-bit uint16 PNG
            Depth_T/    ← float32 NPY (aligned with thermal)
          train.txt     ← sequence names for train split
          valid.txt     ← sequence names for val split
          test.txt      ← sequence names for test split

    Each sequence has matched Thermal/{id}.png and Depth_T/{id}.npy files.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        height: int = 256,
        width: int = 320,
        augmentation: bool = False,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.height = height
        self.width = width
        self.augmentation = augmentation
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.samples: list[tuple[Path, Path]] = []
        self._discover_samples()

    def _discover_samples(self) -> None:
        """Find matched thermal-depth pairs from split file."""
        dataset_dir = self.root / "dataset" / "train"
        if not dataset_dir.exists():
            # Fallback: try root directly
            dataset_dir = self.root
            if not dataset_dir.exists():
                return

        # Read split file
        split_map = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
        split_file = dataset_dir / split_map.get(self.split, f"{self.split}.txt")

        if split_file.exists():
            with open(split_file) as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            # If no split file, use all sequences that have Thermal dir
            sequences = [
                d.name for d in dataset_dir.iterdir()
                if d.is_dir() and (d / "Thermal").exists()
            ] if dataset_dir.exists() else []

        for seq_name in sequences:
            seq_dir = dataset_dir / seq_name
            thermal_dir = seq_dir / "Thermal"
            depth_dir = seq_dir / "Depth_T"

            if not thermal_dir.exists() or not depth_dir.exists():
                continue

            for tf in sorted(thermal_dir.iterdir()):
                if tf.suffix not in (".png", ".tiff", ".tif", ".npy"):
                    continue
                stem = tf.stem
                # Match depth file (always .npy in VIVID++)
                df = depth_dir / f"{stem}.npy"
                if df.exists():
                    self.samples.append((tf, df))

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if len(self.samples) == 0:
            # Synthetic fallback for smoke testing
            thermal = np.random.randn(self.height, self.width).astype(np.float32) * 0.5 + 0.5
            depth = np.random.uniform(
                self.min_depth, self.max_depth, (self.height, self.width)
            ).astype(np.float32)
        else:
            idx = idx % len(self.samples)
            thermal_path, depth_path = self.samples[idx]

            # Load thermal (16-bit uint16 PNG)
            thermal = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
            # Load depth (float32 NPY)
            depth = np.load(str(depth_path)).astype(np.float32)

        # Ensure 2D
        if thermal.ndim == 3:
            thermal = thermal[:, :, 0]
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Resize if needed
        if thermal.shape[0] != self.height or thermal.shape[1] != self.width:
            thermal = cv2.resize(thermal, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # Normalize thermal to [0, 1]
        t_min, t_max = thermal.min(), thermal.max()
        if t_max - t_min > 1e-6:
            thermal = (thermal - t_min) / (t_max - t_min)

        # Clamp depth to valid range
        depth = np.clip(depth, 0.0, self.max_depth)

        # Augment
        if self.augmentation:
            thermal, depth = self._augment(thermal, depth)

        # To tensors: (1, H, W)
        thermal_t = torch.from_numpy(thermal.copy()).unsqueeze(0).float()
        depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).float()

        return {"thermal": thermal_t, "depth": depth_t}

    def _augment(
        self, thermal: np.ndarray, depth: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronized augmentations."""
        if np.random.random() > 0.5:
            thermal = np.fliplr(thermal).copy()
            depth = np.fliplr(depth).copy()
        if np.random.random() > 0.5:
            jitter = np.random.uniform(-0.05, 0.05)
            thermal = np.clip(thermal + jitter, 0.0, 1.0)
        return thermal, depth


class ThermalDepthDataset(Dataset):
    """Generic dataset of aligned thermal image + depth ground truth pairs.

    Expected directory structure:
        root/
          thermal/    <-- 16-bit thermal frames (.npy or .png)
          depth/      <-- depth ground truth (.npy or .png)

    File names must match between thermal/ and depth/ directories.
    """

    def __init__(
        self,
        root: str,
        height: int = 512,
        width: int = 640,
        augmentation: bool = False,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.height = height
        self.width = width
        self.augmentation = augmentation
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.thermal_dir = self.root / "thermal"
        self.depth_dir = self.root / "depth"

        self.samples: list[tuple[Path, Path]] = []
        self._discover_samples()

    def _discover_samples(self) -> None:
        """Find matched thermal-depth pairs."""
        if not self.thermal_dir.exists() or not self.depth_dir.exists():
            return

        thermal_files = sorted(self.thermal_dir.iterdir())
        for tf in thermal_files:
            stem = tf.stem
            for ext in [".npy", ".png", ".tiff", ".tif"]:
                df = self.depth_dir / (stem + ext)
                if df.exists():
                    self.samples.append((tf, df))
                    break

    def _load_frame(self, path: Path) -> np.ndarray:
        """Load a single frame (thermal or depth)."""
        if path.suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        return img.astype(np.float32)

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if len(self.samples) == 0:
            thermal = np.random.randn(self.height, self.width).astype(np.float32) * 0.5 + 0.5
            depth = np.random.uniform(
                self.min_depth, self.max_depth, (self.height, self.width)
            ).astype(np.float32)
        else:
            idx = idx % len(self.samples)
            thermal_path, depth_path = self.samples[idx]
            thermal = self._load_frame(thermal_path)
            depth = self._load_frame(depth_path)

        if thermal.ndim == 3:
            thermal = thermal[:, :, 0]
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if thermal.shape[0] != self.height or thermal.shape[1] != self.width:
            thermal = cv2.resize(thermal, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        t_min, t_max = thermal.min(), thermal.max()
        if t_max - t_min > 1e-6:
            thermal = (thermal - t_min) / (t_max - t_min)

        depth = np.clip(depth, self.min_depth, self.max_depth)

        if self.augmentation:
            if np.random.random() > 0.5:
                thermal = np.fliplr(thermal).copy()
                depth = np.fliplr(depth).copy()
            if np.random.random() > 0.5:
                jitter = np.random.uniform(-0.05, 0.05)
                thermal = np.clip(thermal + jitter, 0.0, 1.0)

        thermal_t = torch.from_numpy(thermal.copy()).unsqueeze(0).float()
        depth_t = torch.from_numpy(depth.copy()).unsqueeze(0).float()

        return {"thermal": thermal_t, "depth": depth_t}


def create_split_indices(
    total: int, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42
) -> dict[str, list[int]]:
    """Create train/val/test split indices."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total).tolist()

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def save_split(split: dict[str, list[int]], path: str) -> None:
    """Save split indices to JSON."""
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> dict[str, list[int]]:
    """Load split indices from JSON."""
    with open(path) as f:
        return json.load(f)
