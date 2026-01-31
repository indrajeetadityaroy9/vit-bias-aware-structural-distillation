"""
Model registry with unified create_model() factory.
"""

from .registry import create_model

# Trigger decorator registrations
from . import vit  # Registers 'deit'
from . import cnn  # Registers 'adaptive_cnn', 'resnet18_cifar', 'convnext_v2_tiny'

__all__ = ['create_model']
