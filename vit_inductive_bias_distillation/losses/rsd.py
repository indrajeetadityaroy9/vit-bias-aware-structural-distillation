from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RedundancySuppressionLoss(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_layers: int,
        *,
        kappa: float = 0.01,
    ):
        super().__init__()
        self.kappa = kappa
        self.teacher_dim = teacher_dim

        hidden_dim = max(student_dim, teacher_dim) * 2
        self.aad_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        student_intermediates: dict[int, torch.Tensor],
        teacher_intermediates: dict[int, torch.Tensor],
        layer_indices: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = student_intermediates[layer_indices[0]].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, float] = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = teacher_intermediates[layer_idx]

            B, N, D_s = student_tokens.shape
            D_t = self.teacher_dim

            s_flat = self.aad_projectors[i](
                student_tokens.reshape(-1, D_s)
            ).reshape(B, N, D_t)

            s_norm = F.normalize(s_flat.reshape(-1, D_t), dim=0)
            # Keep gradients on student-side AAD projectors.
            t_norm = F.normalize(teacher_tokens.detach().reshape(-1, D_t), dim=0)

            cc = s_norm.T @ t_norm
            target = torch.eye(D_t, device=cc.device)
            diff = (cc - target).pow(2)

            off_diag = ~torch.eye(D_t, dtype=torch.bool, device=cc.device)
            diff[off_diag] *= self.kappa

            layer_loss = diff.mean()
            total_loss = total_loss + layer_loss
            loss_dict[f"rsd_layer_{layer_idx}"] = layer_loss.item()

        total_loss = total_loss / len(layer_indices)

        return total_loss, loss_dict
