"""
Optimizer and scheduler utilities for H100.

Uses fused kernels by default.
"""

import torch.optim as optim


def build_optimizer(model, config, device):
    """Create optimizer with fused kernels."""
    name = config.training.optimizer.lower()
    lr = config.training.learning_rate
    wd = config.training.weight_decay

    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, fused=True)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
    elif name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True, fused=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    name = config.training.scheduler.lower()
    params = config.training.lr_scheduler_params

    if name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=params.get('step_size', 10), gamma=params.get('gamma', 0.1))
    elif name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.get('T_max', config.training.num_epochs), eta_min=params.get('eta_min', 0.0001))
    elif name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params.get('factor', 0.1), patience=params.get('patience', 5))
    else:
        raise ValueError(f"Unknown scheduler: {name}")
