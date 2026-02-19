"""SVD-based spectral feature matching (architecture-agnostic).

Matches the singular-value spectra of student and teacher token-feature
matrices. The singular values capture how representational energy is
distributed across principal directions — the architecture-agnostic
analogue of a spatial frequency spectrum.

Replaces the prior 2D-FFT approach (SpectralKD, arXiv 2412.19055) which
required tokens to lie on a square spatial grid. This formulation operates
on generic (B, N, D) token sequences with no structural assumptions.

Parameter-free: no learnable weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpectralMatchingLoss"]


class SpectralMatchingLoss(nn.Module):
    """Match student and teacher singular-value spectra.

    For each batch element, computes the singular values of the (N, D)
    token-feature matrix, L2-normalizes them to compare distribution
    shape rather than absolute magnitude, aligns spectrum lengths via
    adaptive average pooling, and returns the MSE between the two
    normalized spectra.
    """

    @staticmethod
    def _spectrum(tokens: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized singular values from token matrix.

        Args:
            tokens: (B, N, D) token-feature matrix.

        Returns:
            (B, min(N, D)) normalized singular value vector.
        """
        sigma = torch.linalg.svdvals(tokens.float())  # (B, min(N, D))
        return F.normalize(sigma, p=2, dim=-1)

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE between normalized singular value spectra.

        Args:
            student_tokens: (B, N_s, D_s) student token features.
            teacher_tokens: (B, N_t, D_t) teacher token features.

        Returns:
            Scalar MSE loss between the two singular value distributions.
        """
        s_sigma = self._spectrum(student_tokens)  # (B, min(N_s, D_s))
        t_sigma = self._spectrum(teacher_tokens)  # (B, min(N_t, D_t))

        # Align spectrum lengths to the shorter one
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
