import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(model, data_loader, device, criterion=None, class_names=None):
    """
    Evaluate a model on a dataset and compute comprehensive metrics.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        criterion: Loss function (default: CrossEntropyLoss)
        class_names: Optional list of class names for reporting

    Returns:
        Dict containing predictions, targets, probabilities, and metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for inputs, targets in tqdm(data_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)

    metrics = _calculate_metrics(all_targets, all_predictions, all_probabilities, class_names)
    metrics['loss'] = total_loss / len(data_loader)

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'metrics': metrics
    }


def _calculate_metrics(targets, predictions, probabilities, class_names=None):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}

    metrics['accuracy'] = np.mean(predictions == targets)
    metrics['precision_macro'] = precision_score(targets, predictions, average='macro')
    metrics['recall_macro'] = recall_score(targets, predictions, average='macro')
    metrics['f1_macro'] = f1_score(targets, predictions, average='macro')

    metrics['precision_per_class'] = precision_score(targets, predictions, average=None)
    metrics['recall_per_class'] = recall_score(targets, predictions, average=None)
    metrics['f1_per_class'] = f1_score(targets, predictions, average=None)

    metrics['confusion_matrix'] = confusion_matrix(targets, predictions)

    for k in [1, 3, 5]:
        if k <= probabilities.shape[1]:
            metrics[f'top_{k}_accuracy'] = _top_k_accuracy(targets, probabilities, k)

    metrics['classification_report'] = classification_report(
        targets, predictions,
        target_names=class_names if class_names else None,
        output_dict=True
    )

    num_classes = probabilities.shape[1]
    targets_one_hot = np.eye(num_classes)[targets]

    metrics['auc_macro'] = roc_auc_score(targets_one_hot, probabilities, average='macro')
    metrics['auc_weighted'] = roc_auc_score(targets_one_hot, probabilities, average='weighted')

    return metrics


def _top_k_accuracy(targets, probabilities, k):
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
    correct = 0
    for i, target in enumerate(targets):
        if target in top_k_preds[i]:
            correct += 1
    return correct / len(targets)


def print_evaluation_summary(results, class_names=None):
    """Print structured evaluation summary."""
    if 'metrics' not in results:
        return

    metrics = results['metrics']

    loss_str = f" loss={metrics['loss']:.4f}" if 'loss' in metrics else ""
    print(f"eval accuracy={metrics['accuracy']:.4f} precision={metrics['precision_macro']:.4f} "
          f"recall={metrics['recall_macro']:.4f} f1={metrics['f1_macro']:.4f}{loss_str}")

    topk_parts = []
    for k in [1, 3, 5]:
        if f'top_{k}_accuracy' in metrics:
            topk_parts.append(f"top{k}={metrics[f'top_{k}_accuracy']:.4f}")
    if topk_parts:
        print(f"eval {' '.join(topk_parts)}")

    if 'auc_macro' in metrics:
        print(f"eval auc_macro={metrics['auc_macro']:.4f} auc_weighted={metrics['auc_weighted']:.4f}")
