from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

__all__ = ["evaluate_model", "measure_efficiency"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    num_classes: int,
    valid_indices: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, device=device, dtype=torch.long)
    top_k = min(5, num_classes)

    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs).output

        if valid_indices is not None:
            outputs = outputs[:, valid_indices]

        total_loss += criterion(outputs, targets).item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        correct_top1 += preds.eq(targets).sum().item()
        correct_top5 += (
            outputs.topk(top_k, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        )
        total += targets.size(0)

        confusion.view(-1).index_add_(
            0, targets * num_classes + preds, torch.ones_like(preds),
        )

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    return {
        "val_acc": 100.0 * correct_top1 / total,
        "val_acc_top5": 100.0 * correct_top5 / total,
        "precision_macro": precision.mean().item(),
        "recall_macro": recall.mean().item(),
        "f1_macro": f1.mean().item(),
        "loss": total_loss / total,
    }


@torch.no_grad()
def measure_efficiency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    in_channels: int = 3,
    batch_size: int = 64,
    num_warmup: int = 10,
    num_batches: int = 50,
) -> dict[str, float]:
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6

    dummy = torch.randn(1, in_channels, image_size, image_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy)
    gflops = flop_counter.get_total_flops() / 1e9

    dummy_batch = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
    for _ in range(num_warmup):
        model(dummy_batch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_batches):
        model(dummy_batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    throughput = (batch_size * num_batches) / elapsed

    return {
        "param_count": param_count,
        "param_count_m": param_count_m,
        "gflops": gflops,
        "throughput_img_per_sec": throughput,
    }
