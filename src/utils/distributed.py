"""
Distributed training utilities for H100 GPUs.
"""
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def find_free_port() -> int:
    """Find a free port for DDP communication."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def is_torchrun() -> bool:
    """Check if running under torchrun."""
    return all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])


def init_distributed(rank: int = None, world_size: int = None) -> tuple:
    """
    Initialize distributed training.

    Returns:
        Tuple of (rank, world_size, device)
    """
    if is_torchrun():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = rank or 0
        world_size = world_size or 1
        local_rank = rank
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', str(find_free_port()))
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # H100 optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )
        dist.barrier()

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
