from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["UncertaintyWeighting", "WarmupSchedule"]


class UncertaintyWeighting:
    def __init__(self, component_names: list[str], temperature: float = 2.0):
        self.component_names = list(component_names)
        self.temperature = temperature

    def forward(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        info: dict[str, float] = {}

        # Zero-ramped losses are excluded to avoid unstable inverse weighting.
        active = {k: v for k, v in losses.items() if v.item() > 1e-8}

        if not active:
            ref = next(iter(losses.values()))
            for name in self.component_names:
                info[f"weight_{name}"] = 0.0
            return ref.new_zeros(()), info

        raw_w = torch.stack(
            [1.0 / v.detach().clamp(min=1e-8) for v in active.values()]
        )
        omega = F.softmax(raw_w / self.temperature, dim=0)

        # Scale by active count to preserve loss magnitude after softmax normalization.
        n_active = len(active)
        total = torch.zeros((), device=raw_w.device, dtype=raw_w.dtype)
        for i, (name, loss) in enumerate(active.items()):
            scaled_w = omega[i] * n_active
            total = total + scaled_w * loss
            info[f"weight_{name}"] = scaled_w.item()

        for name in self.component_names:
            if name not in active:
                info[f"weight_{name}"] = 0.0

        return total, info


class WarmupSchedule:
    def __init__(
        self,
        total_steps: int,
        warmup_fraction: float,
        stagger: dict[str, float],
    ):
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.starts = {k: int(self.warmup_steps * s) for k, s in stagger.items()}

    def get_ramp(self, component: str, step: int) -> float:
        start = self.starts[component]
        if step < start:
            return 0.0
        if step >= self.warmup_steps:
            return 1.0
        progress = (step - start) / max(self.warmup_steps - start, 1)
        return 0.5 * (1 - math.cos(math.pi * progress))
