"""
Knowledge distillation module for CNN→ViT and DINO→ViT transfer.

Provides:
- Standard DeiT-style distillation (hard/soft KL)
- Self-supervised CST-style distillation (token correlation)
- Structural distillation (CKA, Gram matrix)
- Distillation trainers with DDP support
"""

from .losses import (
    DistillationLoss,
    ProjectionHead,
    TokenRepresentationLoss,
    TokenCorrelationLoss,
    CKALoss,
    GramMatrixLoss,
    LayerWiseStructuralLoss,
    SelfSupervisedDistillationLoss,
)

from .engine import (
    DistillationTrainer,
    SelfSupervisedDistillationTrainer,
)

from .teachers import (
    load_dino_teacher,
    DINO_EMBED_DIMS,
)

__all__ = [
    # Losses - Standard
    'DistillationLoss',
    # Losses - Token-level
    'ProjectionHead',
    'TokenRepresentationLoss',
    'TokenCorrelationLoss',
    # Losses - Structural
    'CKALoss',
    'GramMatrixLoss',
    'LayerWiseStructuralLoss',
    # Losses - Combined
    'SelfSupervisedDistillationLoss',
    # Trainers
    'DistillationTrainer',
    'SelfSupervisedDistillationTrainer',
    # Teachers
    'load_dino_teacher',
    'DINO_EMBED_DIMS',
]
