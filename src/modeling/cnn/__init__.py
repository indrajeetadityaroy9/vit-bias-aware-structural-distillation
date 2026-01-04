"""
CNN models for the model registry.

Import this module to register:
- adaptive_cnn
- resnet18_cifar
- convnext_v2_tiny (if timm available)
"""
# Trigger decorator registrations
from . import adaptive  # Registers 'adaptive_cnn'
from . import resnet    # Registers 'resnet18_cifar'

# ConvNeXt is optional (requires timm)
try:
    from . import convnext  # Registers 'convnext_v2_tiny' if timm available
except ImportError:
    pass

# Note: We don't re-export model classes to avoid circular imports
# Use create_model('model_name', config) instead
