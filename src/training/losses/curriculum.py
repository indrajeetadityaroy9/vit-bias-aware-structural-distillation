"""
Adaptive scheduling and learned loss weighting for distillation.

Implements:
- UncertaintyWeighting: Learned multi-task loss weighting (Kendall et al., CVPR 2018)
  L_weighted = sum_k [ L_k / (2*sigma_k^2) + log(sigma_k) ]
- WarmupSchedule: Unified cosine warmup with staggered component activation

UncertaintyWeighting replaces manual lambda_max tuning by learning per-component
log-variance parameters. WarmupSchedule replaces 8 per-component warmup boundaries
with a single warmup_fraction and fixed stagger offsets.
"""

import math

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """
    Learned multi-task loss weighting (Kendall et al., CVPR 2018).

    Each component gets a learned log_sigma parameter. The weighted loss is:
    L_k_weighted = L_k / (2 * sigma_k^2) + log(sigma_k)

    This is equivalent to maximizing a Gaussian likelihood with learned variance,
    automatically balancing components by their homoscedastic uncertainty.
    """

    def __init__(self, component_names):
        """
        Args:
            component_names: List of loss component names (e.g., ['rsd', 'gram', 'attn', 'spectral'])
        """
        super().__init__()
        self.component_names = list(component_names)
        self.log_sigmas = nn.Parameter(torch.zeros(len(component_names)))

    def forward(self, losses):
        """
        Apply uncertainty weighting to loss components.

        Args:
            losses: Dict[str, Tensor] mapping component names to raw loss values

        Returns:
            total: Weighted sum of all components (scalar tensor)
            info: Dict with learned sigma and effective weight per component
        """
        total = torch.tensor(0.0, device=self.log_sigmas.device)
        info = {}

        for i, name in enumerate(self.component_names):
            log_sigma = self.log_sigmas[i]
            precision = torch.exp(-2 * log_sigma)
            total = total + 0.5 * precision * losses[name] + log_sigma
            info[f'sigma_{name}'] = torch.exp(log_sigma).item()
            info[f'weight_{name}'] = (0.5 * precision).item()

        return total, info


class WarmupSchedule:
    """
    Unified cosine warmup with staggered component activation.

    Replaces 8 per-component warmup start/end steps with a single warmup_fraction.
    Components activate at fixed stagger offsets within the warmup window:
    - rsd:      0% (immediate)
    - gram:    25% into warmup
    - attn:    25% into warmup
    - spectral: 50% into warmup
    """

    STAGGER = {'rsd': 0.0, 'gram': 0.25, 'attn': 0.25, 'spectral': 0.5}

    def __init__(self, total_steps, warmup_fraction=0.1):
        """
        Args:
            total_steps: Total training steps (len(train_loader) * num_epochs)
            warmup_fraction: Fraction of total steps for warmup window
        """
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.starts = {k: int(self.warmup_steps * s) for k, s in self.STAGGER.items()}

    def get_ramp(self, component, step):
        """
        Get warmup ramp value for a component at the given step.

        Returns:
            float in [0, 1]: 0 before component start, cosine ramp during warmup, 1 after
        """
        start = self.starts.get(component, 0)
        if step < start:
            return 0.0
        if step >= self.warmup_steps:
            return 1.0
        progress = (step - start) / max(self.warmup_steps - start, 1)
        return 0.5 * (1 - math.cos(math.pi * progress))
