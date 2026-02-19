"""Attention-map distillation with learned head alignment (A2D)."""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttentionDistillationLoss"]

_ATTN_EPS: Final[float] = 1e-8


class HeadAligner(nn.Module):
    """Learnable 1x1 Conv2d for cross-layer head alignment (A2D).

    Maps total student attention heads to total teacher attention heads
    with only (S_heads * T_heads) parameters.
    """

    def __init__(self, total_student_heads: int, total_teacher_heads: int):
        """Initialize 1x1 conv for head alignment."""
        super().__init__()
        self.conv = nn.Conv2d(
            total_student_heads, total_teacher_heads,
            kernel_size=1, bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Align student heads to teacher head space.

        Args:
            x: (B, total_student_heads, N, N) student attention logits
        Returns:
            (B, total_teacher_heads, N, N) aligned attention logits
        """
        return self.conv(x)


class AttentionDistillationLoss(nn.Module):
    """KL divergence between aligned student and teacher attention maps.

    Student inputs are raw attention logits (pre-softmax).
    Teacher inputs are post-softmax probabilities (from hooks).
    """

    def __init__(
        self,
        student_heads_per_layer: int,
        teacher_heads_per_layer: int,
        num_layers: int,
        init_temperature: float = 1.0,
    ):
        """Initialize head aligner and learnable temperature."""
        super().__init__()
        self.student_heads_per_layer = student_heads_per_layer
        self.teacher_heads_per_layer = teacher_heads_per_layer
        self.num_layers = num_layers

        total_student_heads = student_heads_per_layer * num_layers
        total_teacher_heads = teacher_heads_per_layer * num_layers

        self.head_aligner = HeadAligner(total_student_heads, total_teacher_heads)
        self._raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.exp(init_temperature) - 1.0))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature via softplus reparameterization."""
        return F.softplus(self._raw_temperature)

    def _align_spatial(self, attn: torch.Tensor, target_size: int) -> torch.Tensor:
        """Interpolate attention matrices to the target token resolution."""
        B, H, N, _ = attn.shape
        if N == target_size:
            return attn
        attn_flat = attn.reshape(B * H, 1, N, N)
        attn_resized = F.interpolate(
            attn_flat, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        attn_resized = attn_resized.reshape(B, H, target_size, target_size)
        # Renormalize after interpolation (teacher probs must sum to 1)
        return attn_resized / (attn_resized.sum(dim=-1, keepdim=True) + _ATTN_EPS)

    def forward(
        self,
        student_attns: dict[int, torch.Tensor],
        teacher_attns: dict[int, torch.Tensor],
        layer_pairs: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Compute mean KL divergence across configured layer pairs.

        Args:
            student_attns: {layer_idx: (B, H_s, N_s, N_s)} raw logits (pre-softmax)
            teacher_attns: {layer_idx: (B, H_t, N_t, N_t)} probabilities (post-softmax)
            layer_pairs: [(student_layer, teacher_layer), ...]
        """
        # Stack all student attention logits: (B, total_student_heads, N_s, N_s)
        s_layers = [student_attns[s_layer] for s_layer, _ in layer_pairs]
        s_stacked = torch.cat(s_layers, dim=1)

        # Align student heads to teacher head space via 1x1 conv
        # (B, total_student_heads, N_s, N_s) -> (B, total_teacher_heads, N_s, N_s)
        aligned_logits = self.head_aligner(s_stacked)
        N_s = s_stacked.shape[2]

        # Split aligned logits back into per-layer groups
        aligned_per_layer = aligned_logits.split(self.teacher_heads_per_layer, dim=1)

        T = self.temperature
        device = s_stacked.device
        total_loss = torch.tensor(0.0, device=device)

        for i, (_, t_layer) in enumerate(layer_pairs):
            t_attn = teacher_attns[t_layer]

            # Spatially align teacher probs to student resolution
            t_attn = self._align_spatial(t_attn, N_s)

            # Student: apply softmax with temperature to raw logits (single softmax)
            s_log_prob = F.log_softmax(aligned_per_layer[i] / T, dim=-1)

            # Teacher: already post-softmax probabilities, use directly
            # (no re-softmax — avoids softmax-of-softmax bug)
            t_prob = t_attn

            kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            total_loss = total_loss + kl

        return total_loss / len(layer_pairs)
