"""VRM-style relational distillation loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RelationalLoss"]


class RelationalLoss(nn.Module):
    """Dimension-preserving pairwise relational matching (VRM).

    Replaces Gram matrix matching with sampled pairwise token differences
    (arXiv 2502.20760). Preserves per-dimension feature structure and
    avoids O(N^2) memory via random pair sampling.
    """

    def __init__(self, num_pairs: int = 128):
        """Initialize with number of token pairs to sample."""
        super().__init__()
        self.num_pairs = num_pairs

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber loss between student and teacher pairwise relations."""
        B, N, D_s = student_tokens.shape
        _, _, D_t = teacher_tokens.shape
        K = min(self.num_pairs, N * (N - 1) // 2)

        # Sample random pairs ensuring idx_i != idx_j
        idx_i = torch.randint(0, N, (K,), device=student_tokens.device)
        idx_j = torch.randint(0, N - 1, (K,), device=student_tokens.device)
        idx_j = idx_j + (idx_j >= idx_i).long()

        # Pairwise differences: (B, K, D)
        s_diff = F.normalize(
            student_tokens[:, idx_i, :] - student_tokens[:, idx_j, :], dim=-1
        )
        t_diff = F.normalize(
            teacher_tokens[:, idx_i, :] - teacher_tokens[:, idx_j, :], dim=-1
        )

        # Align channel dimensions via adaptive_avg_pool1d on last dim (D)
        target_d = min(D_s, D_t)
        if D_s != target_d:
            s_diff = F.adaptive_avg_pool1d(s_diff, target_d)
        if D_t != target_d:
            t_diff = F.adaptive_avg_pool1d(t_diff, target_d)

        return F.smooth_l1_loss(s_diff, t_diff)
