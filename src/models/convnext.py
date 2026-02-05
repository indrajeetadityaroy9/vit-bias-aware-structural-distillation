"""
ConvNeXt V2-Tiny adapted for CIFAR-10 (32x32 images).

Uses ImageNet-pretrained backbone with modified stem for small images.
This represents the "Modern CNN Bridge" architecture.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import timm  # Required for ConvNeXt V2

from src.models.registry import register_model


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first tensors (B, C, H, W).
    Used in ConvNeXt stem.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@register_model('convnext_v2_tiny')
class ConvNeXtV2Tiny(nn.Module):
    """
    ConvNeXt V2-Tiny adapted for CIFAR-10 (32x32 images).

    Uses ImageNet-pretrained backbone with modified stem for small images.
    This represents the "Modern CNN Bridge" - an architecture that combines:
    - CNN local operations (depthwise convolutions)
    - Transformer-like macro design (stage ratios, LayerNorm, GELU)

    Key modifications:
    - Original stem: 4x4 patchify with stride 4 (for 224x224 -> 56x56)
    - Modified stem: 2x2 patchify with stride 2 (for 32x32 -> 16x16)
    - Stem weights are reinitialized (see Pre-Flight Check E, F)

    Target accuracy: >95% on CIFAR-10 with fine-tuning.

    Parameters: ~28.6M
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 10)
        self.pretrained = config.get("pretrained", True)  # Use ImageNet weights
        self.drop_path_rate = config.get("drop_path_rate", 0.1)

        # Create ConvNeXt V2 Tiny base model with ImageNet weights
        self.model = timm.create_model(
            "convnextv2_tiny",
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            in_chans=self.in_channels,
            drop_path_rate=self.drop_path_rate,
        )

        # Get original stem output channels (96 for tiny)
        stem_out_channels = 96

        # Modify stem for 32x32 images
        # Original: 4x4 conv, stride 4 -> 56x56 from 224x224
        # Modified: 2x2 conv, stride 2 -> 16x16 from 32x32
        self.model.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels, stem_out_channels, kernel_size=2, stride=2, padding=0
            ),
            LayerNorm2d(stem_out_channels),
        )

        # Reinitialize modified stem weights (pretrained weights don't match 2x2 kernel)
        nn.init.trunc_normal_(self.model.stem[0].weight, std=0.02)
        nn.init.zeros_(self.model.stem[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt V2."""
        return self.model(x)
