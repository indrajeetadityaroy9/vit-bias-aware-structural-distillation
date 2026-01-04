"""
Core utility functions for reproducibility and distributed training.

Extracted from main.py for modular organization.
"""
import logging
import os
import random
import socket
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seeds for partial reproducibility.

    Note: cuDNN benchmark=True and deterministic=False are set for training speed.
    This means runs are NOT fully deterministic but benefit from faster kernel selection.
    For strict reproducibility, set benchmark=False and deterministic=True.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def find_free_port() -> int:
    """Find a free port for DDP communication.

    Returns:
        Available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_ddp_environment(rank: int, world_size: int, port: Optional[int] = None) -> None:
    """Set up DDP environment variables and initialize process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        port: Port for DDP communication (default: 29500 or from MASTER_PORT env)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(port or 29500))
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )


def cleanup_ddp() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_info() -> tuple:
    """Get distributed training world info.

    Returns:
        Tuple of (rank, world_size, is_main_process)
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size, rank == 0
