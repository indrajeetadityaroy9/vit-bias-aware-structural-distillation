"""
Vision Transformer models.

Import this module to register DeiT.
"""
# Trigger decorator registration
from . import deit  # Registers 'deit'

# Note: We don't re-export DeiT class here to avoid circular imports
# Use create_model('deit', config) instead
