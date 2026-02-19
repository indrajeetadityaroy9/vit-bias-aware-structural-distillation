from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_inductive_bias_distillation.config import Config, save_config
from vit_inductive_bias_distillation.data.datasets import (
    HFDataset,
    get_dataset_info,
    get_subset_indices,
)
from vit_inductive_bias_distillation.data.transforms import build_eval_transform
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model, measure_efficiency
from vit_inductive_bias_distillation.runtime_log import log_event


def run_eval_suite(
    model: nn.Module,
    config: Config,
    device: torch.device,
    *,
    config_path: str,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    datasets_to_eval = [config.data.dataset] + list(config.data.eval_datasets)
    primary_results: dict[str, Any] = {}
    robustness_results: dict[str, Any] = {}

    for ds_name in datasets_to_eval:
        info = get_dataset_info(ds_name)
        transform = build_eval_transform(config.model.vit.img_size, info["mean"], info["std"])

        dataset = HFDataset(ds_name, "val", transform)
        loader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
        )

        valid_indices = None
        if "parent_dataset" in info:
            valid_indices = get_subset_indices(ds_name)

        metrics = evaluate_model(model, loader, device, criterion, valid_indices=valid_indices)

        if ds_name == config.data.dataset:
            primary_results = metrics
        else:
            robustness_results[ds_name] = metrics

        log_event(
            "eval_dataset",
            dataset=ds_name,
            top1=metrics["val_acc"],
            top5=metrics["val_acc_top5"],
            loss=metrics["loss"],
        )

    efficiency = measure_efficiency(
        model, device,
        image_size=config.model.vit.img_size,
    )

    log_event(
        "eval_efficiency",
        dataset=config.data.dataset,
        params_m=efficiency["param_count_m"],
        gflops=efficiency["gflops"],
        throughput_img_per_sec=efficiency["throughput_img_per_sec"],
    )

    return {
        "run": {"name": config.run.name, "config": config_path},
        "primary": {
            "dataset": config.data.dataset,
            **primary_results,
        },
        "robustness": robustness_results,
        "efficiency": efficiency,
    }


def save_metrics(
    results: dict[str, Any],
    output_dir: Path,
    config: Config,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    return metrics_path
