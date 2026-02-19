"""Standalone evaluation for a trained BASD checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vit_inductive_bias_distillation.config import load_config, save_config
from vit_inductive_bias_distillation.data.datasets import ImageNetDataset
from vit_inductive_bias_distillation.data.transforms import build_eval_transform
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.models.deit import DeiT

def main() -> None:
    parser = argparse.ArgumentParser(description="BASD Evaluation")
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = load_config(args.config)

    if not config.checkpoint:
        raise ValueError("config.checkpoint must be set for evaluation")

    model = DeiT(config.vit, config.model).to(device)

    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(config.checkpoint, ckpt["epoch"])

    eval_transform = build_eval_transform(config)
    val_dataset = ImageNetDataset("val", eval_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    metrics = evaluate_model(model, val_loader, device, criterion)

    print(
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["loss"],
    )

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(str(output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
