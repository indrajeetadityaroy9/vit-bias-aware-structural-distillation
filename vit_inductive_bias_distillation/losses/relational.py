from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RelationalLoss"]


class RelationalLoss(nn.Module):
    # Match token geometry with pairwise directions and triplet angles.

    def __init__(self, num_pairs: int = 128, num_triplets: int = 64, angle_weight: float = 0.5):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_triplets = num_triplets
        self.angle_weight = angle_weight

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, N, D_s = student_tokens.shape
        _, _, D_t = teacher_tokens.shape
        device = student_tokens.device
        K = min(self.num_pairs, N * (N - 1) // 2)

        idx_i = torch.randint(0, N, (K,), device=device)
        idx_j = torch.randint(0, N - 1, (K,), device=device)
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

        dist_loss = F.smooth_l1_loss(s_diff, t_diff)

        T = min(self.num_triplets, K)
        idx_k = torch.randint(0, N, (T,), device=device)
        # Enforce distinct triplets to avoid zero-length direction vectors.
        collides = (idx_k == idx_i[:T]) | (idx_k == idx_j[:T])
        while collides.any():
            idx_k[collides] = torch.randint(0, N, (collides.sum(),), device=device)
            collides = (idx_k == idx_i[:T]) | (idx_k == idx_j[:T])

        e_ij_s = F.normalize(student_tokens[:, idx_i[:T]] - student_tokens[:, idx_j[:T]], dim=-1)
        e_kj_s = F.normalize(student_tokens[:, idx_k] - student_tokens[:, idx_j[:T]], dim=-1)
        s_cos = (e_ij_s * e_kj_s).sum(dim=-1)

        e_ij_t = F.normalize(teacher_tokens[:, idx_i[:T]] - teacher_tokens[:, idx_j[:T]], dim=-1)
        e_kj_t = F.normalize(teacher_tokens[:, idx_k] - teacher_tokens[:, idx_j[:T]], dim=-1)
        t_cos = (e_ij_t * e_kj_t).sum(dim=-1)

        angle_loss = F.smooth_l1_loss(s_cos, t_cos.detach())

        return dist_loss + self.angle_weight * angle_loss
