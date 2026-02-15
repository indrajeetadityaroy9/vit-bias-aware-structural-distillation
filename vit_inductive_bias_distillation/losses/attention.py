"""Attention-map distillation loss."""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttentionDistillationLoss"]

_ATTN_EPS: Final[float] = 1e-8


class AttentionDistillationLoss(nn.Module):
    """KL divergence between aligned teacher and student attention maps."""

    def __init__(self, init_temperature: float = 1.0):
        """Initialize learnable temperature."""
        super().__init__()
        self._raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.exp(init_temperature) - 1.0))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature via softplus reparameterization."""
        return F.softplus(self._raw_temperature)

    def _align_heads(
        self, teacher_attn: torch.Tensor, num_student_heads: int
    ) -> torch.Tensor:
        """Average teacher heads into groups matching student head count."""
        B, H_t, N, _ = teacher_attn.shape
        if H_t % num_student_heads != 0:
            raise ValueError(
                f"Teacher heads ({H_t}) must be divisible by "
                f"student heads ({num_student_heads})"
            )
        group_size = H_t // num_student_heads
        return teacher_attn.reshape(B, num_student_heads, group_size, N, N).mean(dim=2)

    def _align_spatial(self, attn: torch.Tensor, target_size: int) -> torch.Tensor:
        """Interpolate attention matrices to the target token resolution."""
        B, H, N, _ = attn.shape
        attn_flat = attn.reshape(B * H, 1, N, N)
        attn_resized = F.interpolate(
            attn_flat, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        attn_resized = attn_resized.reshape(B, H, target_size, target_size)
        return attn_resized / (attn_resized.sum(dim=-1, keepdim=True) + _ATTN_EPS)

    def forward(
        self,
        student_attns: dict[int, torch.Tensor],
        teacher_attns: dict[int, torch.Tensor],
        layer_pairs: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Compute mean KL divergence across configured layer pairs."""
        device = student_attns[layer_pairs[0][0]].device
        total_loss = torch.tensor(0.0, device=device)

        for s_layer, t_layer in layer_pairs:
            s_attn = student_attns[s_layer]
            t_attn = teacher_attns[t_layer]

            H_s = s_attn.shape[1]
            N_s = s_attn.shape[2]

            t_attn = self._align_heads(t_attn, H_s)
            t_attn = self._align_spatial(t_attn, N_s)

            s_log_prob = F.log_softmax(s_attn / self.temperature, dim=-1)
            t_prob = F.softmax(t_attn / self.temperature, dim=-1)

            kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            total_loss = total_loss + kl

        return total_loss / len(layer_pairs)
