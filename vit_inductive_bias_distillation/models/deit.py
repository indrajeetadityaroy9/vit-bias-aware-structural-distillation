from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vit_inductive_bias_distillation.config import Config

__all__ = ["DeiT", "StudentIntermediates"]


class StudentIntermediates(NamedTuple):
    output: torch.Tensor
    intermediates: dict[int, torch.Tensor]
    attention_weights: dict[int, torch.Tensor]


class DropPath(nn.Module):
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
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if return_attention:
            scale = self.head_dim ** -0.5
            attn_logits = (q @ k.transpose(-2, -1)) * scale
            attn_probs = attn_logits.softmax(dim=-1)
            x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(x), attn_logits

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
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
    def __init__(self, vit_config: Config, model_config: Config):
        super().__init__()

        self.num_classes = model_config.num_classes
        self.embed_dim = vit_config.embed_dim
        self.depth = vit_config.depth

        self.patch_embed = PatchEmbed(
            vit_config.img_size, vit_config.patch_size,
            model_config.in_channels, vit_config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, vit_config.embed_dim))

        dpr = torch.linspace(0, vit_config.drop_path_rate, vit_config.depth).tolist()
        self.blocks = nn.ModuleList([
            Block(vit_config.embed_dim, vit_config.num_heads, 4.0, dpr[i])
            for i in range(vit_config.depth)
        ])

        self.norm = nn.LayerNorm(vit_config.embed_dim)
        self.head = nn.Linear(vit_config.embed_dim, model_config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m: nn.Module) -> None:
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

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        return x + self.pos_embed

    def forward(
        self, x: torch.Tensor, layer_indices: list[int] = ()
    ) -> StudentIntermediates:
        x = self._embed(x)

        layer_set = set(layer_indices)
        intermediates: dict[int, torch.Tensor] = {}
        attention_weights: dict[int, torch.Tensor] = {}

        for idx, block in enumerate(self.blocks):
            if idx in layer_set:
                normed = block.norm1(x)
                attn_out, attn_w = block.attn(normed, return_attention=True)
                attention_weights[idx] = attn_w
                x = x + block.drop_path(attn_out)
                x = x + block.drop_path(block.mlp(block.norm2(x)))
                intermediates[idx] = x[:, 1:, :].clone()
            elif self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm(x)
        cls_out = self.head(x[:, 0])

        return StudentIntermediates(
            output=cls_out,
            intermediates=intermediates,
            attention_weights=attention_weights,
        )
