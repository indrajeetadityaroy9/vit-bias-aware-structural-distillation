"""
Training infrastructure module.

Provides:
- Optimizers with H100 fused kernels
- LR schedulers
- Training loops (standard, DDP)
- Checkpointing utilities
- GPU-side augmentation (MixUp, CutMix)
- CUDA Graphs for kernel launch optimization
"""

from .components import EarlyStopping, LabelSmoothingCrossEntropy
from .optimizers import build_optimizer, build_scheduler
from .checkpointing import build_checkpoint_dict, restore_rng_state
from .engine import Trainer, DDPTrainer
from .gpu_augment import GPUMixUp, GPUCutMix, GPUMixUpCutMix, CUDAGraphWrapper

__all__ = [
    # Trainers
    'Trainer',
    'DDPTrainer',
    # Utilities
    'EarlyStopping',
    'LabelSmoothingCrossEntropy',
    'build_optimizer',
    'build_scheduler',
    'build_checkpoint_dict',
    'restore_rng_state',
    # H100 GPU optimizations
    'GPUMixUp',
    'GPUCutMix',
    'GPUMixUpCutMix',
    'CUDAGraphWrapper',
]
