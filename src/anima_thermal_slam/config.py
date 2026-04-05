from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class ModelConfig:
    name: str = "thermal_depth_net"
    backbone: str = "simple_cnn"
    recurrent: str = "convgru"  # convgru|reservoir|none
    use_trefnet: bool = True
    pretrained_backbone: bool = False
    in_channels: int = 1
    hidden_channels: int = 128
    recurrent_layers: int = 1


@dataclass
class DataConfig:
    root_dir: str = ""
    train_list: str = ""
    val_list: str = ""
    test_list: str = ""
    seq_len: int = 5
    stride: int = 1
    height: int = 256
    width: int = 320
    return_last_only: bool = True
    use_percentile_stretch: bool = False
    percentile: float = 5.0
    use_clahe: bool = False
    clahe_clip: float = 3.0


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4
    workers: int = 2
    lr: float = 2e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    w_silog: float = 0.9
    w_ssim: float = 0.4
    w_order: float = 0.1
    w_smooth: float = 0.1
    temporal_lambda: float = 0.1
    use_silog: bool = True


@dataclass
class InferConfig:
    checkpoint: str = ""
    input_dir: str = ""
    output_dir: str = "outputs"
    decode_all: bool = False


@dataclass
class SLAMConfig:
    percent_low: float = 5.0
    percent_high: float = 95.0
    invert_colormap: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)


def _coerce_section(section_cls: type[Any], values: dict[str, Any] | None) -> Any:
    if not values:
        return section_cls()
    allowed = {f.name for f in fields(section_cls)}
    cleaned = {k: v for k, v in values.items() if k in allowed}
    return section_cls(**cleaned)


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open("rb") as f:
        raw = tomllib.load(f)

    cfg = AppConfig(
        model=_coerce_section(ModelConfig, raw.get("model")),
        data=_coerce_section(DataConfig, raw.get("data")),
        train=_coerce_section(TrainConfig, raw.get("train")),
        infer=_coerce_section(InferConfig, raw.get("infer")),
        slam=_coerce_section(SLAMConfig, raw.get("slam")),
    )
    return cfg
