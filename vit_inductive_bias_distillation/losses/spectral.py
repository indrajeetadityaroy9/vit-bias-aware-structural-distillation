from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpectralMatchingLoss"]


class SpectralMatchingLoss(nn.Module):
    @staticmethod
    def _spectrum(tokens: torch.Tensor) -> torch.Tensor:
        sigma = torch.linalg.svdvals(tokens.float())
        return F.normalize(sigma, p=2, dim=-1)

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        s_sigma = self._spectrum(student_tokens)
        t_sigma = self._spectrum(teacher_tokens)

        K_s, K_t = s_sigma.shape[1], t_sigma.shape[1]
        target_k = min(K_s, K_t)
        if K_s != target_k:
            s_sigma = F.adaptive_avg_pool1d(
                s_sigma.unsqueeze(1), target_k
            ).squeeze(1)
        if K_t != target_k:
            t_sigma = F.adaptive_avg_pool1d(
                t_sigma.unsqueeze(1), target_k
            ).squeeze(1)

        return F.mse_loss(s_sigma, t_sigma)
