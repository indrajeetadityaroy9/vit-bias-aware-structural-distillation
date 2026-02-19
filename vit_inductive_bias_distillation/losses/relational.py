from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RelationalLoss"]


class RelationalLoss(nn.Module):
    def __init__(self, num_pairs: int = 128):
        super().__init__()
        self.num_pairs = num_pairs

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, N, D_s = student_tokens.shape
        _, _, D_t = teacher_tokens.shape
        K = min(self.num_pairs, N * (N - 1) // 2)

        idx_i = torch.randint(0, N, (K,), device=student_tokens.device)
        idx_j = torch.randint(0, N - 1, (K,), device=student_tokens.device)
        idx_j = idx_j + (idx_j >= idx_i).long()

        s_diff = F.normalize(
            student_tokens[:, idx_i, :] - student_tokens[:, idx_j, :], dim=-1
        )
        t_diff = F.normalize(
            teacher_tokens[:, idx_i, :] - teacher_tokens[:, idx_j, :], dim=-1
        )

        target_d = min(D_s, D_t)
        if D_s != target_d:
            s_diff = F.adaptive_avg_pool1d(s_diff, target_d)
        if D_t != target_d:
            t_diff = F.adaptive_avg_pool1d(t_diff, target_d)

        return F.smooth_l1_loss(s_diff, t_diff)
