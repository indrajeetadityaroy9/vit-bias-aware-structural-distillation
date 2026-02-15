"""2D spectral feature matching with learnable radial band weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpectralMatchingLoss"]


class SpectralMatchingLoss(nn.Module):
    """Match student and teacher FFT magnitudes across radial bands."""

    def __init__(self, num_bands: int = 4):
        """Initialize band weights and mask cache."""
        super().__init__()
        self.num_bands = num_bands
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        self._band_cache: dict[tuple[int, int], list[torch.Tensor]] = {}

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape tokens to a square grid for 2D FFT."""
        B, N, D = tokens.shape
        # BASD uses square patch grids, so N must be H*W with H=W.
        H = W = int(N ** 0.5)
        return tokens.permute(0, 2, 1).reshape(B, D, H, W)

    def _get_radial_bands(
        self, H: int, W: int, device: torch.device
    ) -> list[torch.Tensor]:
        """Return cached radial frequency band masks."""
        key = (H, W)
        if key not in self._band_cache:
            cy, cx = H // 2, W // 2
            y = torch.arange(H, dtype=torch.float32) - cy
            x = torch.arange(W, dtype=torch.float32) - cx
            radius = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
            max_r = radius.max()

            bands = []
            for k in range(self.num_bands):
                lo = max_r * k / self.num_bands
                hi = max_r * (k + 1) / self.num_bands
                mask = ((radius >= lo) & (radius < hi)).unsqueeze(0).unsqueeze(0)
                bands.append(mask.float())

            self._band_cache[key] = bands

        return [b.to(device) for b in self._band_cache[key]]

    def forward(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted spectral MSE over radial bands."""
        s_2d = self._to_spatial(student_tokens)
        t_2d = self._to_spatial(teacher_tokens)

        s_freq = torch.fft.fftshift(torch.fft.fft2(s_2d))
        t_freq = torch.fft.fftshift(torch.fft.fft2(t_2d))

        s_mag = s_freq.abs().mean(dim=1)
        t_mag = t_freq.abs().mean(dim=1)

        _, H, W = s_mag.shape
        bands = self._get_radial_bands(H, W, s_mag.device)
        weights = F.softplus(self.band_weights)

        loss = torch.tensor(0.0, device=student_tokens.device)
        for k, mask in enumerate(bands):
            m = mask.squeeze(1)
            band_loss = F.mse_loss(s_mag * m, t_mag * m)
            loss = loss + weights[k] * band_loss

        return loss / self.num_bands
