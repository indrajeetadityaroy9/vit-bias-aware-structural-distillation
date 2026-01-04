"""
DeiT model with registry decorator.
"""
from src.vit import DeiT
from src.modeling.registry import register_model

# Register DeiT with the model registry
register_model('deit')(DeiT)

__all__ = ['DeiT']
