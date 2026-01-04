"""
Rank-aware logging utilities for distributed training.

Provides logging setup that respects DDP process ranks.
"""
import logging
from pathlib import Path
from typing import Optional

import torch


def setup_logging(config) -> None:
    """Set up logging based on config (non-distributed).

    Args:
        config: LoggingConfig with log_level and log_dir
    """
    log_level = getattr(logging, config.log_level.upper())

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(config.log_dir) / 'training.log'),
            logging.StreamHandler()
        ]
    )

    logging.getLogger(__name__).info("Logging configured successfully")


def setup_logging_for_rank(
    config,
    rank: int,
    world_size: int,
    mode_name: str = "Training"
) -> logging.Logger:
    """Set up logging based on process rank.

    Only the main process (rank 0) gets full logging.
    Other processes only log ERROR level.

    Args:
        config: Config object with logging settings
        rank: Process rank
        world_size: Total number of processes
        mode_name: Training mode name for log header

    Returns:
        Logger instance for the calling module
    """
    is_main_process = (rank == 0)

    if is_main_process:
        setup_logging(config.logging)
        log = logging.getLogger(__name__)
        log.info("=" * 60)
        log.info(f"{mode_name} (DDP)")
        log.info("=" * 60)
        log.info(f"Experiment: {config.experiment_name}")
        log.info(f"Dataset: {config.data.dataset}")
        log.info(f"World Size: {world_size}")
        log.info(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
        log.info("=" * 60)
    else:
        logging.basicConfig(level=logging.ERROR)
        log = logging.getLogger(__name__)

    return log


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, returns root logger.

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
