"""
ResNet-18 modified for CIFAR-10 (32x32 images).

This represents the "Classic CNN" inductive bias for the distillation study.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import register_model


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18.

    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


@register_model('resnet18_cifar')
class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 (32x32 images).

    Key modifications from ImageNet ResNet-18:
    1. First conv: 3x3 kernel with stride 1 (not 7x7 with stride 2)
    2. No max pooling after first conv (preserves spatial resolution)
    3. Better suited for small 32x32 images

    This represents the "Classic CNN" inductive bias for the distillation study.
    Target accuracy: >94% on CIFAR-10.

    Architecture:
        Input (3, 32, 32)
        -> Conv3x3 (64) -> BN -> ReLU      [32x32]
        -> Layer1: 2x BasicBlock(64)       [32x32]
        -> Layer2: 2x BasicBlock(128)      [16x16]
        -> Layer3: 2x BasicBlock(256)      [8x8]
        -> Layer4: 2x BasicBlock(512)      [4x4]
        -> AdaptiveAvgPool -> FC(10)

    Parameters: ~11.2M
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 10)
        self.in_planes = 64

        # Modified first layer for CIFAR (3x3, stride 1, no maxpool)
        self.conv1 = nn.Conv2d(
            self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # NO MaxPool - removed for 32x32 images

        # ResNet-18 layer configuration: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, self.num_classes)

        self._init_weights()

    def _make_layer(
        self, block: type, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build a layer with multiple residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18."""
        out = F.relu(self.bn1(self.conv1(x)))
        # No maxpool for CIFAR
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
