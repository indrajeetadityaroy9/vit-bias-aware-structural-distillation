"""
Distillation loss functions.

Provides:
- Standard KD losses (hard/soft distillation)
- Token-level losses (L_tok, L_rel)
- Structural losses (CKA, Gram matrix)
- Combined self-supervised distillation loss
"""

from .standard import DistillationLoss
from .token import ProjectionHead, TokenRepresentationLoss, TokenCorrelationLoss
from .structural import CKALoss, GramMatrixLoss, LayerWiseStructuralLoss
from .combined import SelfSupervisedDistillationLoss

__all__ = [
    # Standard distillation
    'DistillationLoss',
    # Token-level losses
    'ProjectionHead',
    'TokenRepresentationLoss',
    'TokenCorrelationLoss',
    # Structural losses
    'CKALoss',
    'GramMatrixLoss',
    'LayerWiseStructuralLoss',
    # Combined
    'SelfSupervisedDistillationLoss',
]
