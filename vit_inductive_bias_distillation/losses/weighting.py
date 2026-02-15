"""Learned component weighting and warmup ramps for BASD losses."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn as nn

__all__ = ["UncertaintyWeighting", "WarmupSchedule"]


class UncertaintyWeighting(nn.Module):
    """Uncertainty-based multi-task weighting."""

    def __init__(self, component_names: list[str]):
        """Initialize one learnable log-sigma per component."""
        super().__init__()
        self.component_names = list(component_names)
        self.log_sigmas = nn.Parameter(torch.zeros(len(component_names)))

    def forward(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Apply L/(2*sigma^2) + log(sigma) per component."""
        total = self.log_sigmas.new_zeros(())
        info: dict[str, float] = {}

        for i, name in enumerate(self.component_names):
            log_sigma = self.log_sigmas[i]
            precision = torch.exp(-2 * log_sigma)
            total = total + 0.5 * precision * losses[name] + log_sigma
            info[f"sigma_{name}"] = torch.exp(log_sigma).item()
            info[f"weight_{name}"] = (0.5 * precision).item()

        return total, info


class WarmupSchedule:
    """Cosine warmup with per-component staggered start offsets."""

    _DEFAULT_STAGGER: ClassVar[dict[str, float]] = {
        "rsd": 0.0,
        "gram": 0.25,
        "attn": 0.25,
        "spectral": 0.5,
    }

    def __init__(
        self,
        total_steps: int,
        warmup_fraction: float = 0.1,
        stagger: dict[str, float] | None = None,
    ):
        """Build warmup schedule."""
        self._stagger = stagger or self._DEFAULT_STAGGER
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.starts = {k: int(self.warmup_steps * s) for k, s in self._stagger.items()}

    def get_ramp(self, component: str, step: int) -> float:
        """Return the component ramp value in [0, 1]."""
        start = self.starts[component]
        if step < start:
            return 0.0
        if step >= self.warmup_steps:
            return 1.0
        progress = (step - start) / max(self.warmup_steps - start, 1)
        return 0.5 * (1 - math.cos(math.pi * progress))
