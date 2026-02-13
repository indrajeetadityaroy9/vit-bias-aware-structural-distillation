"""
2D spectral feature matching loss with learnable radial band weights.

Reshapes tokens (B, N, D) -> (B, D, H, W) where H=W=sqrt(N), applies
2D FFT, partitions into radial frequency bands, and computes weighted
MSE per band. Learnable band weights allow the model to discover which
frequency bands matter for DINOv2->DeiT transfer.

Reference: SDKD (arXiv:2507.02939) — Spectral Decoupling Knowledge Distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMatchingLoss(nn.Module):
    """
    2D spectral feature matching with learnable radial band weights.

    Reshapes tokens to their 2D spatial grid, applies 2D FFT, partitions
    the frequency domain into radial bands, and matches magnitude spectra
    with per-band learnable weights.
    """

    def __init__(self, num_bands=4):
        """
        Args:
            num_bands: Number of radial frequency bands to partition into
        """
        super().__init__()
        self.num_bands = num_bands
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        self._band_cache = {}

    def _to_spatial(self, tokens):
        """Reshape tokens (B, N, D) -> (B, D, H, W) for 2D FFT."""
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Token count {N} is not a perfect square"
        return tokens.permute(0, 2, 1).reshape(B, D, H, W)

    def _get_radial_bands(self, H, W, device):
        """Compute radial frequency band masks, cached by spatial dims."""
        cache_key = (H, W, device)
        if cache_key in self._band_cache:
            return self._band_cache[cache_key]

        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx
        radius = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
        max_r = radius.max()

        bands = []
        for k in range(self.num_bands):
            lo = max_r * k / self.num_bands
            hi = max_r * (k + 1) / self.num_bands
            mask = ((radius >= lo) & (radius < hi)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            bands.append(mask.float())

        self._band_cache[cache_key] = bands
        return bands

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute spectral matching loss.

        Args:
            student_tokens: (B, N, D_s) student token representations (spatially aligned)
            teacher_tokens: (B, N, D_t) teacher token representations (spatially aligned)
                           D_s and D_t may differ; magnitude is averaged over D.

        Returns:
            loss: Weighted MSE across radial frequency bands
        """
        s_2d = self._to_spatial(student_tokens)  # (B, D_s, H, W)
        t_2d = self._to_spatial(teacher_tokens)  # (B, D_t, H, W)

        # 2D FFT and shift DC to center
        s_freq = torch.fft.fftshift(torch.fft.fft2(s_2d))
        t_freq = torch.fft.fftshift(torch.fft.fft2(t_2d))

        # Average magnitude over feature dimension to handle D_s != D_t
        s_mag = s_freq.abs().mean(dim=1)  # (B, H, W)
        t_mag = t_freq.abs().mean(dim=1)  # (B, H, W)

        _, H, W = s_mag.shape
        bands = self._get_radial_bands(H, W, s_mag.device)
        weights = F.softplus(self.band_weights)

        loss = 0.0
        for k, mask in enumerate(bands):
            # Squeeze mask from (1, 1, H, W) to (1, H, W) for broadcasting
            m = mask.squeeze(1)
            band_loss = F.mse_loss(s_mag * m, t_mag * m)
            loss = loss + weights[k] * band_loss

        return loss / self.num_bands
