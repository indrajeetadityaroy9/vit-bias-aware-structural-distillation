"""
Data-efficient Image Transformer (DeiT) for H100 GPUs.

Uses Flash Attention via SDPA and gradient checkpointing by default.
"""

import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.registry import register_model


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
    """Patch embedding with optional conv stem."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, use_conv_stem: bool = False):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        if use_conv_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            self.proj = nn.Conv2d(embed_dim, embed_dim, patch_size, patch_size)
        else:
            self.stem = None
            self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stem is not None:
            x = self.stem(x)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head attention with Flash Attention via SDPA."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_weights: Optional[torch.Tensor] = None

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
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    """MLP with GELU."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@register_model('deit')
class DeiT(nn.Module):
    """
    DeiT with distillation token.

    Gradient checkpointing enabled by default for memory efficiency.
    """

    def __init__(self, config: dict):
        super().__init__()

        in_channels = config.get('in_channels', 3)
        num_classes = config.get('num_classes', 10)
        img_size = config.get('img_size', 32)
        patch_size = config.get('patch_size', 4)
        embed_dim = config.get('embed_dim', 192)
        depth = config.get('depth', 12)
        num_heads = config.get('num_heads', 3)
        mlp_ratio = config.get('mlp_ratio', 4.0)
        drop_rate = config.get('drop_rate', 0.0)
        attn_drop_rate = config.get('attn_drop_rate', 0.0)
        drop_path_rate = config.get('drop_path_rate', 0.1)
        use_conv_stem = config.get('use_conv_stem', False)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        # Grayscale expansion
        if in_channels == 1:
            self.channel_expand = nn.Conv2d(1, 3, 1, bias=False)
            in_channels = 3
        else:
            self.channel_expand = None

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim, use_conv_stem)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
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
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        if self.training:
            return cls_out, dist_out
        return (cls_out + dist_out) / 2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward_with_intermediates(self, x: torch.Tensor, layer_indices: List[int] = None) -> Dict:
        """Forward with intermediate outputs for distillation."""
        layer_indices = layer_indices or []
        intermediates = {}

        if self.channel_expand is not None:
            x = self.channel_expand(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in layer_indices:
                intermediates[idx] = x[:, 2:, :].clone()  # Patch tokens only

        x = self.norm(x)
        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        return {
            'output': (cls_out, dist_out) if self.training else (cls_out + dist_out) / 2,
            'intermediates': intermediates,
            'patch_tokens': x[:, 2:, :]
        }

    def get_attention_weights(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract attention weights."""
        if self.channel_expand is not None:
            x = self.channel_expand(x)

        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        weights = {}
        for idx, block in enumerate(self.blocks):
            block.attn(block.norm1(x), return_attention=True)
            if block.attn.attn_weights is not None:
                weights[idx] = block.attn.attn_weights
            x = block(x)

        return weights
