"""
Configuration management for the canonical BASD training path.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


DATASET_IMG_SIZES = {'imagenet': 224}
_DINO_EMBED_DIMS = {'dinov2_vits14': 384}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _to_plain(value):
    if isinstance(value, SimpleNamespace):
        return {k: _to_plain(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _resolve_derived_values(config: SimpleNamespace) -> None:
    config.vit.img_size = DATASET_IMG_SIZES[config.data.dataset]
    config.basd.teacher_embed_dim = _DINO_EMBED_DIMS[config.basd.teacher_model_name]
    config.basd.cross_attn_num_heads = max(1, config.basd.teacher_embed_dim // 64)
    config.basd.rsd_kappa = 1.0 / config.basd.teacher_embed_dim
    config.basd.attn_layer_pairs = [[layer, layer] for layer in config.basd.token_layers]
    config.basd.spectral_num_bands = config.vit.img_size // config.vit.patch_size


def load_config(config_path):
    config_path = Path(config_path)
    defaults_path = Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'

    with open(defaults_path, 'r') as f:
        defaults_config = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        experiment_config = yaml.safe_load(f)

    raw_config = _deep_merge(defaults_config, experiment_config)
    config = _to_namespace(raw_config)
    _resolve_derived_values(config)
    return config


def save_config(config, save_path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(_to_plain(config), f, default_flow_style=False)
