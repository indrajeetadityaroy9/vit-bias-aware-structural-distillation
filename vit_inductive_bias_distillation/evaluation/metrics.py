"""Evaluation metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

__all__ = ["evaluate_model"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Evaluate a model and return scalar metrics."""
    model.eval()

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        predictions = outputs.argmax(dim=1)
        correct += predictions.eq(targets).sum().item()
        total += targets.size(0)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    if world_size > 1:
        stats = torch.tensor([total_loss, correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        correct = int(stats[1].item())
        total = int(stats[2].item())

    all_preds_np = torch.cat(all_predictions).numpy()
    all_tgts_np = torch.cat(all_targets).numpy()

    metrics = {
        "accuracy": correct / total,
        "precision_macro": precision_score(all_tgts_np, all_preds_np, average="macro"),
        "recall_macro": recall_score(all_tgts_np, all_preds_np, average="macro"),
        "f1_macro": f1_score(all_tgts_np, all_preds_np, average="macro"),
        "loss": total_loss / total,
        "val_acc": 100.0 * correct / total,
    }

    return metrics
