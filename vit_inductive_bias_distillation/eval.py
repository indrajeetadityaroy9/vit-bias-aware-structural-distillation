"""Standalone evaluation for a trained BASD checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from vit_inductive_bias_distillation.config import load_config, save_config
from vit_inductive_bias_distillation.data import create_dataloaders
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.models import build_student_model

def main() -> None:
    parser = argparse.ArgumentParser(description="BASD Evaluation")
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    config = load_config(args.config)

    if not config.checkpoint:
        raise ValueError("config.checkpoint must be set for evaluation")

    model = build_student_model(config, device)

    ckpt = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(config.checkpoint, ckpt["epoch"])

    _, _, _, test_loader = create_dataloaders(config, world_size=1, rank=0, with_augmentation=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    results = evaluate_model(model, test_loader, device, criterion)
    metrics = results["metrics"]

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
