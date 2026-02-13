"""
Structural distillation loss using Gram matrix anchoring.

Implements GramAnchoringLoss: Gram matrix structural consistency (from DINOv3).
Preserves pairwise token relationships (N×N structure) via L2-normalized
Gram matrix matching.

Reference: DINOv3 (arXiv:2508.10104) — Gram matrix anchoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramAnchoringLoss(nn.Module):
    """
    Gram matrix structural consistency (from DINOv3).

    L_gram = ||X_s·X_s^T - X_t·X_t^T||_F^2

    Operates on L2-normalized token features. Preserves pairwise
    token relationships (N×N structure) without HSIC normalization overhead.
    This is orthogonal to the D×D cross-correlation in RedundancySuppressionLoss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute Gram anchoring loss.

        Args:
            student_tokens: (B, N, D_s) student features (raw, pre-projection)
            teacher_tokens: (B, N, D_t) teacher features (raw, pre-projection)

        Returns:
            loss: Frobenius norm squared of Gram matrix difference
        """
        student_tokens = F.normalize(student_tokens, dim=-1)
        teacher_tokens = F.normalize(teacher_tokens, dim=-1)

        # N×N Gram matrices (pairwise token similarities)
        G_s = torch.bmm(student_tokens, student_tokens.transpose(1, 2))  # (B, N, N)
        G_t = torch.bmm(teacher_tokens, teacher_tokens.transpose(1, 2))  # (B, N, N)

        loss = (G_s - G_t).pow(2).mean()
        return loss
