"""
ViT building blocks and utilities.

Re-exports components from the original vit.py for modularity.
"""
from src.vit import (
    interpolate_pos_embed,
    DropPath,
    PatchEmbed,
    HybridPatchEmbed,
    MultiHeadSelfAttention,
    MLP,
    TransformerBlock,
    DEIT_CONFIGS,
    get_deit_config,
    HAS_SDPA,
)

__all__ = [
    'interpolate_pos_embed',
    'DropPath',
    'PatchEmbed',
    'HybridPatchEmbed',
    'MultiHeadSelfAttention',
    'MLP',
    'TransformerBlock',
    'DEIT_CONFIGS',
    'get_deit_config',
    'HAS_SDPA',
]
