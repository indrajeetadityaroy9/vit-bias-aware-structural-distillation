from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np
import torch
import torch.utils.data
from datasets import Image, ClassLabel, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader
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

from vit_inductive_bias_distillation.config import Config


_PARENT_DATASETS: dict[str, str] = {
    "barkermrl/imagenet-a": "ILSVRC/imagenet-1k",
    "axiong/imagenet-r": "ILSVRC/imagenet-1k",
    "songweig/imagenet_sketch": "ILSVRC/imagenet-1k",
}


def _detect_column_keys(features: dict) -> tuple[str, str]:
    image_key = next(name for name, feat in features.items() if isinstance(feat, Image))
    label_key = next(name for name, feat in features.items() if isinstance(feat, ClassLabel))
    return image_key, label_key


@lru_cache(maxsize=8)
def _compute_channel_stats(
    hf_path: str, split: str, image_key: str, *, n_samples: int = 5000,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ds = load_dataset(hf_path, split=split, streaming=True)

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        img = example[image_key].convert("RGB")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        pixel_sum += flat.sum(axis=0)
        pixel_sq_sum += (flat ** 2).sum(axis=0)
        pixel_count += flat.shape[0]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return tuple(mean.tolist()), tuple(std.tolist())


@lru_cache(maxsize=16)
def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    builder = load_dataset_builder(dataset_name)
    features = builder.info.features
    available_splits = set(builder.info.splits.keys())

    image_key, label_key = _detect_column_keys(features)
    num_classes = features[label_key].num_classes
    class_names = features[label_key].names

    val_split_name = "validation" if "validation" in available_splits else "test"
    split_map: dict[str, str] = {"train": "train", "val": val_split_name}

    if dataset_name in _PARENT_DATASETS:
        parent_info = get_dataset_info(_PARENT_DATASETS[dataset_name])
        mean, std = parent_info["mean"], parent_info["std"]
    else:
        mean, std = _compute_channel_stats(dataset_name, split_map["train"], image_key)

    result: dict[str, Any] = {
        "image_key": image_key,
        "label_key": label_key,
        "num_classes": num_classes,
        "class_names": class_names,
        "split_map": split_map,
        "mean": mean,
        "std": std,
    }

    if dataset_name in _PARENT_DATASETS:
        result["parent_dataset"] = _PARENT_DATASETS[dataset_name]

    return result


@lru_cache(maxsize=16)
def get_subset_indices(subset_name: str) -> tuple[int, ...]:
    subset_info = get_dataset_info(subset_name)
    parent_info = get_dataset_info(subset_info["parent_dataset"])

    parent_name_to_idx = {name: idx for idx, name in enumerate(parent_info["class_names"])}

    return tuple(sorted(parent_name_to_idx[name] for name in subset_info["class_names"]))


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


def build_augmented_transform(
    image_size: int,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Compose:
    return Compose([
        RandomResizedCrop(image_size),
        RandomHorizontalFlip(),
        TrivialAugmentWide(),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
        RandomErasing(p=0.25),
    ])


def build_train_transform(
    image_size: int,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> DualTransform:
    return DualTransform(
        build_eval_transform(image_size, mean=mean, std=std),
        build_augmented_transform(image_size, mean=mean, std=std),
    )


def build_eval_transform(
    image_size: int,
    *,
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


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, split: str, transform: Any):
        info = get_dataset_info(dataset_name)
        hf_split = info["split_map"][split]
        self.dataset = load_dataset(dataset_name, split=hf_split)
        self.transform = transform
        self._image_key = info["image_key"]
        self._label_key = info["label_key"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item[self._image_key].convert("RGB")
        transformed = self.transform(image)
        if isinstance(transformed, tuple):
            return (*transformed, item[self._label_key])
        return transformed, item[self._label_key]


def create_eval_loader(
    dataset_name: str,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    info = get_dataset_info(dataset_name)
    transform = build_eval_transform(image_size, mean=info["mean"], std=info["std"])
    dataset = HFDataset(dataset_name, "val", transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def create_dataloaders(
    config: Config,
    *,
    view_mode: Literal["dual", "single"] = "dual",
) -> tuple[DataLoader, DataLoader]:
    info = get_dataset_info(config.data.dataset)
    image_size = config.model.vit.img_size

    if view_mode == "dual":
        train_transform = build_train_transform(image_size, mean=info["mean"], std=info["std"])
    else:
        train_transform = build_augmented_transform(image_size, mean=info["mean"], std=info["std"])

    train_dataset = HFDataset(config.data.dataset, "train", train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = create_eval_loader(
        config.data.dataset,
        image_size=image_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    return train_loader, val_loader
