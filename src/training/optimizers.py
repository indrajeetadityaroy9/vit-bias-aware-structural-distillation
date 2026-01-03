"""
Optimizer and scheduler utilities with H100 optimizations.

Provides:
- build_optimizer: Create optimizer with fused kernel support
- build_scheduler: Create learning rate scheduler
"""

import logging

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

# Check PyTorch version for H100 optimizations
PYTORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
HAS_PYTORCH_2 = PYTORCH_VERSION >= (2, 0)
HAS_PYTORCH_2_1 = PYTORCH_VERSION >= (2, 1)


def build_optimizer(model, config, device):
    """
    Create optimizer with H100 optimizations.

    Args:
        model: PyTorch model (unwrapped)
        config: Config object with training settings
        device: Target device

    Returns:
        Configured optimizer
    """
    opt_name = config.training.optimizer.lower()
    lr = config.training.learning_rate
    wd = config.training.weight_decay

    # Fused optimizers require PyTorch 2.0+ and CUDA
    use_fused = (config.training.use_fused_optimizer and
                 device.type == 'cuda' and HAS_PYTORCH_2)

    if not HAS_PYTORCH_2 and config.training.use_fused_optimizer:
        logger.warning("Fused optimizers require PyTorch 2.0+. Using standard optimizers.")

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, fused=use_fused)
    elif opt_name == 'adamw':
        if use_fused:
            logger.info("Using fused AdamW optimizer (H100 optimized)")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=use_fused)
    elif opt_name == 'sgd':
        if use_fused:
            logger.info("Using fused SGD optimizer (H100 optimized)")
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                         momentum=0.9, nesterov=True, fused=use_fused)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, config):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Config object with training settings

    Returns:
        Configured scheduler or None
    """
    scheduler_name = config.training.scheduler.lower()
    params = config.training.lr_scheduler_params

    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 10),
            gamma=params.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get('T_max', config.training.num_epochs),
            eta_min=params.get('eta_min', 0.0001)
        )
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.get('factor', 0.1),
            patience=params.get('patience', 5),
            min_lr=params.get('min_lr', 1e-7)
        )
    elif scheduler_name == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get('gamma', 0.95)
        )
    elif scheduler_name == 'cyclic':
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=params.get('base_lr', 0.0001),
            max_lr=params.get('max_lr', config.training.learning_rate),
            step_size_up=params.get('step_size_up', 2000),
            mode=params.get('mode', 'triangular2')
        )
    else:
        return None


__all__ = ['build_optimizer', 'build_scheduler', 'HAS_PYTORCH_2', 'HAS_PYTORCH_2_1']
