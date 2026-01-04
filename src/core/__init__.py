"""
Core infrastructure module for the ViT Inductive Bias Distillation framework.

Provides:
- Configuration management with hierarchical loading
- Rank-aware logging for distributed training
- Utility functions for reproducibility and DDP setup
"""

from .config import (
    ConfigManager,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    LoggingConfig,
    ViTConfig,
    DistillationConfig,
    SelfSupervisedDistillationConfig,
)

from .logging import (
    setup_logging,
    setup_logging_for_rank,
    get_logger,
)

from .utils import (
    set_seed,
    find_free_port,
    setup_ddp_environment,
    cleanup_ddp,
    get_world_info,
)

__all__ = [
    # Config classes
    'ConfigManager',
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'LoggingConfig',
    'ViTConfig',
    'DistillationConfig',
    'SelfSupervisedDistillationConfig',
    # Logging
    'setup_logging',
    'setup_logging_for_rank',
    'get_logger',
    # Utils
    'set_seed',
    'find_free_port',
    'setup_ddp_environment',
    'cleanup_ddp',
    'get_world_info',
]
