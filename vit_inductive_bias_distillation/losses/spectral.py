"""Phase-preserving 2D spectral feature matching (SpectralKD)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpectralMatchingLoss"]


class SpectralMatchingLoss(nn.Module):
    """Match student and teacher frequency representations via rfft2.

    Uses real+imaginary component matching instead of magnitude-only,
    preserving phase information (SpectralKD, arXiv 2412.19055).
    Parameter-free: no learnable weights.
    """

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape (B, N, D) tokens to (B, D, H, W) spatial grid."""
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        return tokens.permute(0, 2, 1).reshape(B, D, H, W)

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE on stacked real+imag rfft2 components."""
        s_2d = self._to_spatial(student_tokens)
        t_2d = self._to_spatial(teacher_tokens)

        # Half-spectrum FFT with orthonormal normalization
        s_freq = torch.fft.rfft2(s_2d, norm="ortho")
        t_freq = torch.fft.rfft2(t_2d, norm="ortho")

        # Stack real and imaginary: (B, D, H, W//2+1) complex -> (B, 2, D, H, W//2+1)
        s_ri = torch.stack([s_freq.real, s_freq.imag], dim=1)
        t_ri = torch.stack([t_freq.real, t_freq.imag], dim=1)

        # Align channel dimensions if D_s != D_t
        D_s = s_ri.shape[2]
        D_t = t_ri.shape[2]
        if D_s != D_t:
            target_d = min(D_s, D_t)
            # adaptive_avg_pool1d pools the LAST dimension, so we need D last.
            # Input: (B, 2, D, H, W_half) -> merge batch+spatial -> (B*2*H*W_half, D)
            # -> unsqueeze for pool1d -> (B*2*H*W_half, 1, D) -> pool -> reshape back
            B, two, _, H, W_half = s_ri.shape
            if D_s > target_d:
                # (B, 2, D_s, H, W_half) -> (B, 2, H, W_half, D_s) -> flatten -> pool -> reshape
                s_flat = s_ri.permute(0, 1, 3, 4, 2).reshape(-1, 1, D_s)
                s_flat = F.adaptive_avg_pool1d(s_flat, target_d)
                s_ri = s_flat.reshape(B, two, H, W_half, target_d).permute(0, 1, 4, 2, 3)
            if D_t > target_d:
                t_flat = t_ri.permute(0, 1, 3, 4, 2).reshape(-1, 1, D_t)
                t_flat = F.adaptive_avg_pool1d(t_flat, target_d)
                t_ri = t_flat.reshape(B, two, H, W_half, target_d).permute(0, 1, 4, 2, 3)

        return F.mse_loss(s_ri, t_ri)
