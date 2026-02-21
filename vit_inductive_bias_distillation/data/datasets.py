from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch.utils.data
from datasets import Image, ClassLabel, load_dataset, load_dataset_builder
from torch.utils.data import DataLoader

from vit_inductive_bias_distillation.config import Config
from vit_inductive_bias_distillation.data.transforms import build_eval_transform, build_train_transform

__all__ = ["HFDataset", "get_dataset_info", "get_subset_indices", "create_dataloaders"]

_HF_ALIASES: dict[str, str] = {
    "imagenet1k": "ILSVRC/imagenet-1k",
    "cifar100": "uoft-cs/cifar100",
    "cifar10": "uoft-cs/cifar10",
    "flowers102": "nelorth/oxford-flowers",
    "stanford_cars": "tanganke/stanford_cars",
    "imagenet_a": "barkermrl/imagenet-a",
    "imagenet_r": "axiong/imagenet-r",
    "imagenet_sketch": "songweig/imagenet_sketch",
}

_PARENT_DATASETS: dict[str, str] = {
    "imagenet_a": "imagenet1k",
    "imagenet_r": "imagenet1k",
}


def _detect_column_keys(features: dict) -> tuple[str, str]:
    image_key = None
    label_key = None
    for name, feat in features.items():
        if isinstance(feat, Image) and image_key is None:
            image_key = name
        elif isinstance(feat, ClassLabel) and label_key is None:
            label_key = name
    if image_key is None or label_key is None:
        raise ValueError(
            f"Could not auto-detect image/label columns from features: {list(features)}"
        )
    return image_key, label_key


@lru_cache(maxsize=16)
def _compute_channel_stats(
    hf_path: str, split: str, image_key: str, n_samples: int = 5000,
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
    hf_path = _HF_ALIASES.get(dataset_name, dataset_name)
    builder = load_dataset_builder(hf_path)
    features = builder.info.features
    available_splits = set(builder.info.splits.keys())

    image_key, label_key = _detect_column_keys(features)
    num_classes = features[label_key].num_classes

    split_map: dict[str, str] = {}
    if "train" in available_splits:
        split_map["train"] = "train"
    if "validation" in available_splits:
        split_map["val"] = "validation"
    elif "test" in available_splits:
        split_map["val"] = "test"
    elif "train" in available_splits:
        split_map["val"] = "train"

    stats_split = split_map.get("train", split_map.get("val"))
    mean, std = _compute_channel_stats(hf_path, stats_split, image_key)

    result: dict[str, Any] = {
        "hf_path": hf_path,
        "image_key": image_key,
        "label_key": label_key,
        "num_classes": num_classes,
        "split_map": split_map,
        "mean": mean,
        "std": std,
    }

    parent = _PARENT_DATASETS.get(dataset_name)
    if parent is not None:
        result["parent_dataset"] = parent

    return result


@lru_cache(maxsize=4)
def get_subset_indices(subset_name: str) -> list[int]:
    info = get_dataset_info(subset_name)
    parent_name = info.get("parent_dataset")
    if parent_name is None:
        raise ValueError(f"{subset_name} has no parent_dataset defined")

    parent_info = get_dataset_info(parent_name)

    subset_builder = load_dataset_builder(info["hf_path"])
    parent_builder = load_dataset_builder(parent_info["hf_path"])

    subset_features = subset_builder.info.features
    parent_features = parent_builder.info.features

    _, subset_label_key = _detect_column_keys(subset_features)
    _, parent_label_key = _detect_column_keys(parent_features)

    subset_class_names = subset_features[subset_label_key].names
    parent_class_names = parent_features[parent_label_key].names

    parent_name_to_idx = {name: idx for idx, name in enumerate(parent_class_names)}

    indices = []
    for name in subset_class_names:
        if name in parent_name_to_idx:
            indices.append(parent_name_to_idx[name])

    if len(indices) != len(subset_class_names):
        matched = len(indices)
        expected = len(subset_class_names)
        raise ValueError(
            f"Only matched {matched}/{expected} class names from "
            f"{subset_name} to {parent_name}"
        )

    return sorted(indices)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, split: str, transform: Any):
        info = get_dataset_info(dataset_name)
        hf_split = info["split_map"][split]
        self.dataset = load_dataset(info["hf_path"], split=hf_split)
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


def create_dataloaders(
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    info = get_dataset_info(config.data.dataset)
    image_size = config.model.vit.img_size
    mean = info["mean"]
    std = info["std"]

    train_transform = build_train_transform(image_size, mean, std)
    eval_transform = build_eval_transform(image_size, mean, std)

    train_dataset = HFDataset(config.data.dataset, "train", train_transform)
    val_dataset = HFDataset(config.data.dataset, "val", eval_transform)

    num_workers = config.data.num_workers
    batch_size = config.data.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader
