"""
Data-efficient Image Transformer (DeiT) for H100 GPUs.

Uses Flash Attention via SDPA and gradient checkpointing by default.
Distillation token removed — BASD uses structural losses instead.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DropPath(nn.Module):
    """Stochastic depth regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


class PatchEmbed(nn.Module):
    """Patch embedding via linear projection."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head attention with Flash Attention via SDPA."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if return_attention:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self.attn_weights = attn.detach()
            x = attn @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    """MLP with GELU."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DeiT(nn.Module):
    """
    DeiT without distillation token.

    BASD uses structural losses (CKA, attention, frequency) instead of
    a distillation token. Only CLS token + patch tokens.
    Gradient checkpointing enabled by default for memory efficiency.
    """

    def __init__(self, config: dict):
        super().__init__()

        in_channels = config['in_channels']
        num_classes = config['num_classes']
        img_size = config['img_size']
        patch_size = config['patch_size']
        embed_dim = config['embed_dim']
        depth = config['depth']
        num_heads = config['num_heads']
        drop_path_rate = config['drop_path_rate']

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # CLS + patches

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, 4.0, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        if self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    def forward_with_intermediates(self, x: torch.Tensor, layer_indices: List[int]) -> Dict:
        """
        Forward pass with intermediate outputs and attention weights for BASD.

        For layers in layer_indices, extracts patch token intermediates and
        attention weights. Uses gradient checkpointing for non-captured layers.

        Args:
            x: Input images (B, C, H, W)
            layer_indices: List of block indices to capture intermediates from

        Returns:
            Dict with keys:
                'output': Classification logits (B, num_classes)
                'intermediates': Dict[layer_idx] → (B, N_patches, embed_dim) patch tokens
                'patch_tokens': (B, N_patches, embed_dim) final patch tokens
                'attention_weights': Dict[layer_idx] → (B, H, N, N) attention weights
        """
        layer_set = set(layer_indices)
        intermediates = {}
        attention_weights = {}

        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed

        for idx, block in enumerate(self.blocks):
            if idx in layer_set:
                # Compute attention once and reuse output (no double compute)
                normed = block.norm1(x)
                attn_out = block.attn(normed, return_attention=True)
                attention_weights[idx] = block.attn.attn_weights
                x = x + block.drop_path(attn_out)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
                intermediates[idx] = x[:, 1:, :].clone()  # Patch tokens only (skip CLS)
            elif self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        cls_out = self.head(x[:, 0])

        return {
            'output': cls_out,
            'intermediates': intermediates,
            'patch_tokens': x[:, 1:, :],
            'attention_weights': attention_weights,
        }
