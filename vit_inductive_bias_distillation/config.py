from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config", "save_config"]

_DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "configs" / "defaults.yaml"


class Config:
    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, Config):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> Config:
    from vit_inductive_bias_distillation.data.datasets import get_dataset_info

    with open(_DEFAULTS_PATH) as f:
        defaults = yaml.safe_load(f)

    with open(config_path) as f:
        overrides = yaml.safe_load(f)

    presets = defaults.pop("presets")

    merged = _deep_merge(defaults, overrides)

    preset_name = merged["model"]["student_preset"]
    if preset_name:
        merged["model"]["vit"].update(presets[preset_name])

    dataset_info = get_dataset_info(merged["data"]["dataset"])
    merged["model"]["num_classes"] = dataset_info["num_classes"]

    return Config(merged)


def save_config(config: Config, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False)
