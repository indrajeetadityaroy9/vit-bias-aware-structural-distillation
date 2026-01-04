"""
Unified model zoo for the ViT Inductive Bias Distillation framework.

Provides:
- Model registry with @register_model decorator
- Unified create_model() factory function
- All models are automatically registered on import

Available models:
- deit: Data-efficient Image Transformer
- adaptive_cnn: AdaptiveCNN with SE blocks
- resnet18_cifar: ResNet-18 for CIFAR
- convnext_v2_tiny: ConvNeXt V2 Tiny (requires timm)

Usage:
    from src.modeling import create_model, list_models

    # List available models
    print(list_models())  # ['deit', 'adaptive_cnn', 'resnet18_cifar', ...]

    # Create a model
    model = create_model('deit', config)
"""
# Export factory function (ONLY export this to avoid circular imports)
from .registry import create_model, list_models, register_model

# Trigger decorator registrations by importing subpackages
from . import vit  # Registers 'deit'
from . import cnn  # Registers 'adaptive_cnn', 'resnet18_cifar', 'convnext_v2_tiny'

__all__ = ['create_model', 'list_models', 'register_model']
