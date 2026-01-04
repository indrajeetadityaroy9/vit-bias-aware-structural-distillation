"""
ResNet-18 for CIFAR with registry decorator.
"""
from src.teachers import ResNet18CIFAR
from src.modeling.registry import register_model

# Register ResNet-18 CIFAR with the model registry
register_model('resnet18_cifar')(ResNet18CIFAR)

__all__ = ['ResNet18CIFAR']
