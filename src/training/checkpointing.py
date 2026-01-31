"""
Checkpoint utilities for training state persistence.

Provides:
- build_checkpoint_dict: Create checkpoint dictionary
"""

import random

import torch
import numpy as np


def build_checkpoint_dict(model, optimizer, scheduler, scaler, swa_model,
                          epoch, metrics, config, best_val_acc, metrics_history,
                          extra_metadata=None):
    """
    Build checkpoint dictionary with all training state.

    Args:
        model: PyTorch model (unwrapped from DDP)
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        scaler: GradScaler (optional)
        swa_model: SWA model (optional)
        epoch: Current epoch
        metrics: Current metrics dict
        config: Config object
        best_val_acc: Best validation accuracy
        metrics_history: Training history
        extra_metadata: Additional distillation-specific metadata (optional)

    Returns:
        Checkpoint dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'best_val_acc': best_val_acc,
        'metrics_history': dict(metrics_history)
    }

    # Save RNG states for reproducible resume
    checkpoint['rng_state'] = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
        'numpy': np.random.get_state(),
        'python': random.getstate()
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if swa_model is not None:
        checkpoint['swa_model_state_dict'] = swa_model.state_dict()

    if extra_metadata is not None:
        checkpoint.update(extra_metadata)

    return checkpoint


__all__ = ['build_checkpoint_dict']
