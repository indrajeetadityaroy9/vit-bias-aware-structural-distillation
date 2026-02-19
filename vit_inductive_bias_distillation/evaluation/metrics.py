from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.flop_counter import FlopCounterMode

__all__ = ["evaluate_model", "measure_efficiency"]


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    valid_indices: list[int] | None = None,
) -> dict[str, Any]:
    model.eval()

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    correct_top5 = 0
    total = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs).output

        if valid_indices is not None:
            outputs = outputs[:, valid_indices]

        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        all_predictions.append(outputs.argmax(dim=1).cpu())
        all_targets.append(targets.cpu())

        # Top-5 is computed per batch since sklearn has no multiclass top-k API.
        top5_preds = outputs.topk(5, dim=1).indices
        correct_top5 += top5_preds.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)

    all_preds_np = torch.cat(all_predictions).numpy()
    all_tgts_np = torch.cat(all_targets).numpy()

    acc = accuracy_score(all_tgts_np, all_preds_np)

    return {
        "val_acc": 100.0 * acc,
        "val_acc_top5": 100.0 * correct_top5 / total,
        "precision_macro": precision_score(all_tgts_np, all_preds_np, average="macro", zero_division=0),
        "recall_macro": recall_score(all_tgts_np, all_preds_np, average="macro", zero_division=0),
        "f1_macro": f1_score(all_tgts_np, all_preds_np, average="macro", zero_division=0),
        "loss": total_loss / total,
    }


@torch.no_grad()
def measure_efficiency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 64,
    num_warmup: int = 10,
    num_batches: int = 50,
) -> dict[str, float]:
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(dummy)
    gflops = flop_counter.get_total_flops() / 1e9

    dummy_batch = torch.randn(batch_size, 3, image_size, image_size, device=device)
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
