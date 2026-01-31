"""
Training utility components.

Provides:
- EarlyStopping: Callback for early stopping
- LabelSmoothingCrossEntropy: Smooth label loss function
"""

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping callback to stop training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for the metric being monitored
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing support."""

    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_pred = torch.log_softmax(pred, dim=-1)

        # Handle soft labels (from CutMix/MixUp) - target is [batch, num_classes] float
        if target.dim() > 1 and target.size(-1) == n_classes:
            # Soft labels: use KL divergence style loss
            loss = -(target * log_pred).sum(dim=-1)
            return loss.mean()

        # Handle hard labels (integer indices) - original behavior
        loss = -log_pred.sum(dim=-1)
        nll = -log_pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = loss / n_classes
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()


__all__ = ['EarlyStopping', 'LabelSmoothingCrossEntropy']
