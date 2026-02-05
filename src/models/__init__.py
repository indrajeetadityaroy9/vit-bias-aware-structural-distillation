"""
Model registry with unified create_model() factory.
"""

from src.models.registry import create_model

# Trigger decorator registrations
import src.models.deit         # Registers 'deit'
import src.models.adaptive_cnn # Registers 'adaptive_cnn'
import src.models.resnet       # Registers 'resnet18_cifar'
import src.models.convnext     # Registers 'convnext_v2_tiny'
