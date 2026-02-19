"""Configuration schema and YAML IO for BASD experiments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 256
    num_workers: int = 16


@dataclass(frozen=True)
class VitConfig:
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    drop_path_rate: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    num_classes: int = 1000


@dataclass(frozen=True)
class TrainingConfig:
    num_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    label_smoothing: float = 0.1
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    swa_start_epoch: float = 0.8
    swa_lr: float = 1e-4
    grad_clip_norm: float = 1.0
    cosine_eta_min: float = 1e-6
    autocast_dtype: str = "bfloat16"


@dataclass(frozen=True)
class BASDConfig:
    teacher_model_name: str = "dinov2_vits14"
    token_layers: list[int] = field(default_factory=lambda: [3, 6, 9, 11])
    warmup_fraction: float = 0.1
    disable_components: list[str] | None = None
    uwso_temperature: float = 2.0
    vrm_num_pairs: int = 128
    num_teacher_layers: int = 12
    layer_selector_temperature: float = 2.0
    layer_selector_diversity_weight: float = 0.01
    layer_selector_recon_weight: float = 0.01
    layer_selector_grass_proj_dim: int = 128
    layer_selector_grass_rank: int = 16
    layer_selector_grass_cov_eps: float = 1e-4
    # Derived fields are computed in load_config().
    teacher_embed_dim: int = 0
    cross_attn_num_heads: int = 0
    rsd_kappa: float = 0.0
    attn_layer_pairs: list[tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class BASDExperimentConfig:
    seed: int = 42
    experiment_name: str = "basd_imagenet"
    output_dir: str = "./outputs"
    resume_from: str | None = None
    checkpoint: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vit: VitConfig = field(default_factory=VitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    basd: BASDConfig = field(default_factory=BASDConfig)


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a frozen dataclass instance."""
    hints = get_type_hints(cls)
    field_types = {f.name: hints[f.name] for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_types:
            raise ValueError(f"Unknown config key '{key}' for {cls.__name__}")
        ft = field_types[key]
        if dataclasses.is_dataclass(ft) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str | Path) -> BASDExperimentConfig:
    """Load YAML config, validate constraints, and fill derived BASD fields."""
    from vit_inductive_bias_distillation.models.teacher import DINO_EMBED_DIMS

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config = _dict_to_dataclass(BASDExperimentConfig, raw)

    if config.vit.img_size % config.vit.patch_size != 0:
        raise ValueError("img_size must be divisible by patch_size")
    if config.basd.teacher_model_name not in DINO_EMBED_DIMS:
        raise ValueError(
            f"Unknown teacher: {config.basd.teacher_model_name}. "
            f"Registered: {list(DINO_EMBED_DIMS)}"
        )

    teacher_dim = DINO_EMBED_DIMS[config.basd.teacher_model_name]
    new_basd = dataclasses.replace(
        config.basd,
        teacher_embed_dim=teacher_dim,
        cross_attn_num_heads=max(1, teacher_dim // 64),
        rsd_kappa=1.0 / teacher_dim,
        attn_layer_pairs=[(layer, layer) for layer in config.basd.token_layers],
    )
    return dataclasses.replace(config, basd=new_basd)


def save_config(config: BASDExperimentConfig, save_path: str | Path) -> None:
    """Save config to YAML."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(dataclasses.asdict(config), f, default_flow_style=False)
