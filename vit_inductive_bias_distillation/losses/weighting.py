"""Analytical UW-SO weighting and warmup ramps for BASD losses."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F

__all__ = ["UncertaintyWeighting", "WarmupSchedule"]


class UncertaintyWeighting:
    """Analytical inverse-loss softmax weighting (UW-SO).

    Replaces learned log-sigma parameters with parameter-free analytical
    weighting (arXiv 2408.07985). Filters out zero-ramped components to
    avoid infinite weights during warmup, and scales by N_active to
    compensate for softmax normalization.
    """

    def __init__(self, component_names: list[str], temperature: float = 2.0):
        """Initialize with component names and temperature."""
        self.component_names = list(component_names)
        self.temperature = temperature

    def forward(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute UW-SO weighted sum of active (non-zero) losses."""
        info: dict[str, float] = {}

        # Filter out zero-ramped components to avoid 1/0 = inf
        active = {k: v for k, v in losses.items() if v.item() > 1e-8}

        if not active:
            # All components are zero (very early warmup) — return zero loss
            ref = next(iter(losses.values()))
            for name in self.component_names:
                info[f"weight_{name}"] = 0.0
            return ref.new_zeros(()), info

        # Inverse-loss weights with stop-gradient
        raw_w = torch.stack(
            [1.0 / v.detach().clamp(min=1e-8) for v in active.values()]
        )
        omega = F.softmax(raw_w / self.temperature, dim=0)

        # Scale by N_active so weighted sum is not suppressed by softmax normalization
        n_active = len(active)
        total = torch.zeros((), device=raw_w.device, dtype=raw_w.dtype)
        for i, (name, loss) in enumerate(active.items()):
            scaled_w = omega[i] * n_active
            total = total + scaled_w * loss
            info[f"weight_{name}"] = scaled_w.item()

        # Log zero weight for inactive components
        for name in self.component_names:
            if name not in active:
                info[f"weight_{name}"] = 0.0

        return total, info


class WarmupSchedule:
    """Cosine warmup with per-component staggered start offsets."""

    _DEFAULT_STAGGER: ClassVar[dict[str, float]] = {
        "rsd": 0.0,
        "rel": 0.25,
        "attn": 0.25,
        "spectral": 0.5,
    }

    def __init__(
        self,
        total_steps: int,
        warmup_fraction: float = 0.1,
    ):
        """Build warmup schedule."""
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.starts = {k: int(self.warmup_steps * s) for k, s in self._DEFAULT_STAGGER.items()}

    def get_ramp(self, component: str, step: int) -> float:
        """Return the component ramp value in [0, 1]."""
        start = self.starts[component]
        if step < start:
            return 0.0
        if step >= self.warmup_steps:
            return 1.0
        progress = (step - start) / max(self.warmup_steps - start, 1)
        return 0.5 * (1 - math.cos(math.pi * progress))
