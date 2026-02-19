"""Adaptive spectral-similarity-guided teacher layer selection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_inductive_bias_distillation.analysis.spectral_profile import spatial_spectral_profile

__all__ = ["SpectralLayerSelector"]


class SpectralLayerSelector(nn.Module):
    """Spectral-similarity-guided soft teacher layer selection.

    For each student extraction point, computes 2D spatial FFT energy profiles
    and uses cosine similarity with a learned per-point temperature to produce
    mixing weights over all teacher layers.

    Uses spatial FFT (rfft2 on H×W grid) rather than channel-wise FFT because:
    - Spatial frequencies are physically meaningful (edges vs textures vs global)
    - Both student and teacher share the same spatial grid (14×14), so profiles
      have identical shape — no dimension alignment needed
    - Channel ordering in ViT embeddings is arbitrary, making channel-wise FFT
      physically meaningless

    Entropy regularization prevents degenerate uniform averaging by gently
    encouraging peaked (sparse) weight distributions.
    """

    def __init__(
        self,
        num_extraction_points: int,
        num_teacher_layers: int = 12,
        init_temperature: float = 2.0,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.num_teacher_layers = num_teacher_layers
        self.entropy_weight = entropy_weight

        # Per-extraction-point learnable log-temperature (softplus → always positive)
        self.log_temperatures = nn.Parameter(
            torch.full(
                (num_extraction_points,),
                math.log(math.exp(init_temperature) - 1),
            )
        )

    @property
    def temperatures(self) -> torch.Tensor:
        """Softplus reparameterization ensures tau > 0."""
        return F.softplus(self.log_temperatures)

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[str, float], torch.Tensor]:
        """Select and mix teacher layers for each student extraction point.

        Args:
            student_tokens_per_layer: {student_layer: (B, N_s, D_s)} from selected layers.
            all_teacher_tokens: {0..11: (B, N_t, D_t)} raw tokens from all teacher layers.
            extraction_indices: list of student layer indices, e.g. [3, 6, 9, 11].

        Returns:
            mixed_teachers: {student_layer: (B, N_t, D_t)} one mixed teacher per point.
            info: per-layer weight values and diagnostics for logging.
            entropy_loss: scalar entropy regularization penalty.
        """
        teacher_indices = sorted(all_teacher_tokens.keys())
        stacked_teacher = torch.stack(
            [all_teacher_tokens[idx] for idx in teacher_indices]
        )  # (L, B, N_t, D_t)

        # Pre-compute teacher profiles (shared across all extraction points)
        teacher_profiles = torch.stack([
            spatial_spectral_profile(all_teacher_tokens[idx].detach())
            for idx in teacher_indices
        ])  # (L, profile_dim)

        mixed_teachers: dict[int, torch.Tensor] = {}
        info: dict[str, float] = {}
        total_entropy = stacked_teacher.new_tensor(0.0)

        for i, s_layer in enumerate(extraction_indices):
            s_tokens = student_tokens_per_layer[s_layer]
            s_profile = spatial_spectral_profile(s_tokens.detach())  # (profile_dim,)

            # Cosine similarity with all teacher layers
            sims = F.cosine_similarity(
                s_profile.unsqueeze(0),  # (1, profile_dim)
                teacher_profiles,         # (L, profile_dim)
            )  # (L,)

            # Temperature-controlled softmax
            tau = self.temperatures[i]
            weights = F.softmax(sims / tau, dim=0)  # (L,)

            # Weighted combination of teacher layers
            mixed = (weights.view(-1, 1, 1, 1) * stacked_teacher).sum(dim=0)  # (B, N_t, D_t)
            mixed_teachers[s_layer] = mixed

            # Entropy: H(w) = -Σ w_k log(w_k). Penalize high entropy (uniform).
            entropy = -(weights * (weights + 1e-8).log()).sum()
            total_entropy = total_entropy + entropy

            for j, t_idx in enumerate(teacher_indices):
                info[f"selector_s{s_layer}_t{t_idx}_weight"] = weights[j].item()
            info[f"selector_s{s_layer}_temperature"] = tau.item()
            info[f"selector_s{s_layer}_entropy"] = entropy.item()

        entropy_loss = self.entropy_weight * total_entropy / len(extraction_indices)
        return mixed_teachers, info, entropy_loss
