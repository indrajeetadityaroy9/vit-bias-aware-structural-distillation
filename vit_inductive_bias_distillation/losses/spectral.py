from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BuresWassersteinLoss"]

_BW_EPS: float = 1e-5
_NEWTON_ITERS: int = 8


def _matrix_sqrt_newton_schulz(A: torch.Tensor, num_iters: int = _NEWTON_ITERS) -> torch.Tensor:
    # Approximate sqrt(A) for batched SPD matrices with Newton-Schulz.
    D = A.shape[-1]
    # Trace normalization keeps eigenvalues in a stable Newton-Schulz regime.
    trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    A_norm = A / trace.clamp(min=_BW_EPS)

    I = torch.eye(D, device=A.device, dtype=A.dtype).unsqueeze(0)
    Y = A_norm
    Z = I.expand_as(A)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # Restore original scale after sqrt(A / trace).
    return Y * trace.sqrt()


class BuresWassersteinLoss(nn.Module):
    # Match token Gaussian statistics with Bures-Wasserstein distance.

    def __init__(self, diagonal: bool = True, mean_weight: float = 1.0, eps: float = _BW_EPS):
        super().__init__()
        self.diagonal = diagonal
        self.mean_weight = mean_weight
        self.eps = eps

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        s = student_tokens.float()
        t = teacher_tokens.detach().float()

        D_s, D_t = s.shape[2], t.shape[2]

        # Align feature width when student and teacher dimensions differ.
        if D_s != D_t:
            target_d = min(D_s, D_t)
            if D_s != target_d:
                s = F.adaptive_avg_pool1d(s, target_d)
            if D_t != target_d:
                t = F.adaptive_avg_pool1d(t, target_d)

        mu_s = s.mean(dim=1)
        mu_t = t.mean(dim=1)
        mean_loss = (mu_s - mu_t).pow(2).sum(dim=-1).mean()

        s_centered = s - mu_s.unsqueeze(1)
        t_centered = t - mu_t.unsqueeze(1)

        if self.diagonal:
            var_s = s_centered.pow(2).mean(dim=1) + self.eps
            var_t = t_centered.pow(2).mean(dim=1) + self.eps
            cov_loss = (var_s.sqrt() - var_t.sqrt()).pow(2).sum(dim=-1).mean()
        else:
            N = s.shape[1]
            D = s.shape[2]
            cov_s = s_centered.transpose(1, 2) @ s_centered / N
            cov_t = t_centered.transpose(1, 2) @ t_centered / N

            eye = torch.eye(D, device=s.device, dtype=s.dtype).unsqueeze(0) * self.eps
            cov_s = cov_s + eye
            cov_t = cov_t + eye

            sqrt_s = _matrix_sqrt_newton_schulz(cov_s)
            M = sqrt_s @ cov_t @ sqrt_s
            sqrt_M = _matrix_sqrt_newton_schulz(M)

            trace = torch.diagonal
            cov_loss = (
                trace(cov_s, dim1=-2, dim2=-1).sum(-1)
                + trace(cov_t, dim1=-2, dim2=-1).sum(-1)
                - 2.0 * trace(sqrt_M, dim1=-2, dim2=-1).sum(-1)
            ).clamp(min=0.0).mean()

        return self.mean_weight * mean_loss + cov_loss
