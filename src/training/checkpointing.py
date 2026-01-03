"""
Checkpoint utilities for training state persistence.

Provides:
- build_checkpoint_dict: Create checkpoint dictionary
- restore_rng_state: Restore RNG states for reproducible resume
"""

import random
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


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
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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


def restore_rng_state(checkpoint):
    """
    Restore RNG states from checkpoint for reproducible resume.

    Args:
        checkpoint: Checkpoint dictionary with 'rng_state' key
    """
    if 'rng_state' not in checkpoint:
        logger.warning("Checkpoint does not contain RNG state - resume may not be reproducible")
        return

    rng_state = checkpoint['rng_state']

    if 'torch' in rng_state:
        torch.set_rng_state(rng_state['torch'])

    if 'torch_cuda' in rng_state and rng_state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])

    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])

    if 'python' in rng_state:
        random.setstate(rng_state['python'])

    logger.info("Restored RNG state from checkpoint")


__all__ = ['build_checkpoint_dict', 'restore_rng_state']
