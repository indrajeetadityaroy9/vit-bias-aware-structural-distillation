"""
AdaptiveCNN model with registry decorator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import register_model


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional SE attention."""

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se is not None:
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


@register_model('adaptive_cnn')
class AdaptiveCNN(nn.Module):
    """Adaptive CNN with SE blocks, automatically configured for dataset."""

    def __init__(self, config):
        super().__init__()

        self.in_channels = config.get('in_channels', 3)
        self.num_classes = config.get('num_classes', 10)
        self.dataset = config.get('dataset', 'custom')
        self.use_se = config.get('use_se', True)

        if self.dataset == 'mnist':
            self._build_mnist_architecture()
        elif self.dataset == 'cifar':
            self._build_cifar_architecture()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}. Use 'mnist' or 'cifar'.")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.get('dropout', 0.5))

    def _build_mnist_architecture(self):
        """Build deeper MNIST architecture for 99%+ accuracy."""
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ),
            ResidualBlock(32, 32, stride=1, use_se=self.use_se),
            ResidualBlock(32, 32, stride=1, use_se=self.use_se),
            ResidualBlock(32, 64, stride=2, use_se=self.use_se),
            ResidualBlock(64, 64, stride=1, use_se=self.use_se),
            ResidualBlock(64, 128, stride=2, use_se=self.use_se),
            ResidualBlock(128, 128, stride=1, use_se=self.use_se),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )

    def _build_cifar_architecture(self):
        """Build deeper CIFAR architecture for 90%+ accuracy."""
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),
            ResidualBlock(64, 64, stride=1, use_se=self.use_se),
            ResidualBlock(64, 64, stride=1, use_se=self.use_se),
            ResidualBlock(64, 128, stride=2, use_se=self.use_se),
            ResidualBlock(128, 128, stride=1, use_se=self.use_se),
            ResidualBlock(128, 128, stride=1, use_se=self.use_se),
            ResidualBlock(128, 256, stride=2, use_se=self.use_se),
            ResidualBlock(256, 256, stride=1, use_se=self.use_se),
            ResidualBlock(256, 256, stride=1, use_se=self.use_se),
            ResidualBlock(256, 512, stride=2, use_se=self.use_se),
            ResidualBlock(512, 512, stride=1, use_se=self.use_se),
            ResidualBlock(512, 512, stride=1, use_se=self.use_se),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
