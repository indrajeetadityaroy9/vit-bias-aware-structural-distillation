"""Evaluation metrics used by the BASD training path."""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate a model and compute primary classification metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_targets = []
    total_loss = 0.0

    for inputs, targets in tqdm(data_loader, desc="Evaluating"):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()

    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision_macro': precision_score(all_targets, all_predictions, average='macro'),
        'recall_macro': recall_score(all_targets, all_predictions, average='macro'),
        'f1_macro': f1_score(all_targets, all_predictions, average='macro'),
        'loss': total_loss / len(data_loader),
    }

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'metrics': metrics,
    }
