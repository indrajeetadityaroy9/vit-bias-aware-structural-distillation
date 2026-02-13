"""
Token-level distillation via redundancy-suppressed cross-correlation.

Implements:
- RedundancySuppressionLoss: Cross-correlation alignment with off-diagonal
  suppression (Barlow Twins / RSD principle). Uses an Architecture-Agnostic
  Decoupler (AAD) MLP to map student features to teacher dimension, then
  drives the cross-correlation matrix toward identity.

Reference: RSD (arXiv:2507.21844) — Redundancy Suppression Distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RedundancySuppressionLoss(nn.Module):
    """
    Cross-correlation alignment with redundancy suppression.

    Maximizes diagonal (invariance) and minimizes off-diagonal (decorrelation)
    of the cross-correlation matrix between student and teacher features.

    L_rsd = ||CC(h(z_s), z_t) - I||_F^2, off-diagonal weighted by kappa.

    The AAD (Architecture-Agnostic Decoupler) is a 2-layer MLP that projects
    student features to teacher dimension without requiring teacher-side
    projectors (eliminates stop-gradient bug).
    """

    def __init__(self, student_dim, teacher_dim, num_layers, kappa=0.01):
        """
        Args:
            student_dim: Student embedding dimension (D_s)
            teacher_dim: Teacher embedding dimension (D_t)
            num_layers: Number of distillation layers (one AAD per layer)
            kappa: Off-diagonal penalty weight (default 0.01)
        """
        super().__init__()
        self.kappa = kappa
        self.teacher_dim = teacher_dim

        # One AAD per layer: student_dim -> hidden_dim -> teacher_dim
        hidden_dim = max(student_dim, teacher_dim) * 2
        self.aad_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, student_intermediates, teacher_intermediates, layer_indices):
        """
        Compute redundancy suppression loss across layers.

        Args:
            student_intermediates: Dict[layer_idx] -> (B, N, D_s)
            teacher_intermediates: Dict[layer_idx] -> (B, N, D_t) spatially aligned
            layer_indices: List of layer indices

        Returns:
            loss: Scalar loss value
            loss_dict: Per-layer losses
        """
        total_loss = 0.0
        loss_dict = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]  # (B, N, D_s)
            teacher_tokens = teacher_intermediates[layer_idx]  # (B, N, D_t)

            B, N, D_s = student_tokens.shape
            D_t = self.teacher_dim

            # AAD: project student to teacher dimension
            s_flat = self.aad_projectors[i](
                student_tokens.reshape(-1, D_s)
            ).reshape(B, N, D_t)

            # L2 normalize along batch*token dimension
            s_norm = F.normalize(s_flat.reshape(-1, D_t), dim=0)  # (B*N, D_t)
            t_norm = F.normalize(teacher_tokens.detach().reshape(-1, D_t), dim=0)

            # Cross-correlation matrix (D_t x D_t)
            cc = s_norm.T @ t_norm  # (D_t, D_t)

            # Target: identity matrix
            target = torch.eye(D_t, device=cc.device)
            diff = (cc - target).pow(2)

            # Weight off-diagonal by kappa
            off_diag = ~torch.eye(D_t, dtype=torch.bool, device=cc.device)
            diff[off_diag] *= self.kappa

            layer_loss = diff.mean()
            total_loss += layer_loss
            loss_dict[f'rsd_layer_{layer_idx}'] = layer_loss.item()

        total_loss = total_loss / len(layer_indices)
        loss_dict['rsd_loss_total'] = total_loss.item()

        return total_loss, loss_dict
