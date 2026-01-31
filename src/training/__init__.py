"""
Training infrastructure module.
"""

from .engine import DDPTrainer
from .checkpointing import build_checkpoint_dict

__all__ = ['DDPTrainer', 'build_checkpoint_dict']
