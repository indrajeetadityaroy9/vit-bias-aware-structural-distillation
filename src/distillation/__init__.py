"""
Knowledge distillation module for CNN->ViT and DINO->ViT transfer.
"""

from .engine import DistillationTrainer, SelfSupervisedDistillationTrainer
from .teachers import load_dino_teacher

__all__ = [
    'DistillationTrainer',
    'SelfSupervisedDistillationTrainer',
    'load_dino_teacher',
]
