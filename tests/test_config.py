from __future__ import annotations

from pathlib import Path

from anima_thermal_slam import load_config


def test_load_config_overrides_and_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[model]
recurrent = "reservoir"
hidden_channels = 64

[data]
seq_len = 7
height = 240
width = 320
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.model.recurrent == "reservoir"
    assert cfg.model.hidden_channels == 64
    assert cfg.data.seq_len == 7
    assert cfg.data.height == 240
    assert cfg.data.width == 320
    # Ensure defaults are preserved for fields not specified.
    assert cfg.train.lr == 2e-4
    assert cfg.infer.output_dir == "outputs"

