"""Gram-matrix structural distillation loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GramAnchoringLoss"]


class GramAnchoringLoss(nn.Module):
    """Match pairwise token affinity via normalized Gram matrices."""

    def forward(self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Return mean squared distance between student/teacher Gram matrices."""
        student_tokens = F.normalize(student_tokens, dim=-1)
        teacher_tokens = F.normalize(teacher_tokens, dim=-1)

        G_s = torch.bmm(student_tokens, student_tokens.transpose(1, 2))
        G_t = torch.bmm(teacher_tokens, teacher_tokens.transpose(1, 2))

        return (G_s - G_t).pow(2).mean()
