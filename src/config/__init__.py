"""
Configuration management with hierarchical config loading.
"""

from src.config.schema import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ViTConfig,
    DistillationConfig,
    SelfSupervisedDistillationConfig,
    load_config,
    save_config,
    validate_config,
)
