from __future__ import annotations

import torch
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Normalize,
    RandomErasing,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToDtype,
    ToImage,
    TrivialAugmentWide,
)

__all__ = ["DualTransform", "build_train_transform", "build_eval_transform"]


class DualTransform:
    def __init__(
        self,
        clean_transform: Compose,
        augmented_transform: Compose,
    ):
        self.clean_transform = clean_transform
        self.augmented_transform = augmented_transform

    def __call__(self, img):
        return self.clean_transform(img), self.augmented_transform(img)


def build_train_transform(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> DualTransform:
    clean = Compose([
        Resize(image_size + 32),
        CenterCrop(image_size),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])

    augmented = Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        TrivialAugmentWide(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
        RandomErasing(p=0.25),
    ])

    return DualTransform(clean, augmented)


def build_eval_transform(
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Compose:
    return Compose([
        Resize(image_size + 32),
        CenterCrop(image_size),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])
