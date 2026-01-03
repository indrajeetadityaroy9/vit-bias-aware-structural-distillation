"""
Token-level distillation losses for self-supervised knowledge transfer.

Implements:
- ProjectionHead: Learnable dimension alignment
- TokenRepresentationLoss (L_tok): Primary signal - matches intermediate embeddings
- TokenCorrelationLoss (L_rel): Regularizer - matches token correlation structures

Reference: Chen et al., "Cross-Architecture Self-supervised Video Representation
Learning", CVPR 2022 (CST paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Learnable projection head to align student/teacher embedding dimensions.

    Architecture: Linear -> LayerNorm -> GELU -> Linear -> LayerNorm
    This stabilizes cosine similarity and allows dimension mismatch handling.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim * 2

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class TokenRepresentationLoss(nn.Module):
    """
    Token representation distillation loss (L_tok) - PRIMARY SIGNAL.

    Matches intermediate layer embeddings between teacher and student
    using learnable projection heads for dimension alignment.
    """

    def __init__(self, student_dim, teacher_dim, projection_dim, num_layers, loss_type='cosine'):
        super().__init__()
        self.loss_type = loss_type
        self.num_layers = num_layers

        # Separate projectors per layer - allows layer-specific alignment
        self.student_projectors = nn.ModuleList([
            ProjectionHead(student_dim, projection_dim)
            for _ in range(num_layers)
        ])
        self.teacher_projectors = nn.ModuleList([
            ProjectionHead(teacher_dim, projection_dim)
            for _ in range(num_layers)
        ])

    def forward(self, student_intermediates, teacher_intermediates, layer_indices):
        """
        Compute token representation loss.

        Args:
            student_intermediates: Dict[layer_idx] -> (B, N_s, d_s)
            teacher_intermediates: Dict[layer_idx] -> (B, N_s, d_t) - already interpolated
            layer_indices: List of layer indices

        Returns:
            loss: Scalar loss value
            loss_dict: Per-layer losses
        """
        total_loss = 0.0
        loss_dict = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]  # (B, N, d_s)
            teacher_tokens = teacher_intermediates[layer_idx]  # (B, N, d_t)

            # Project to common space
            proj_student = self.student_projectors[i](student_tokens)  # (B, N, proj_dim)
            proj_teacher = self.teacher_projectors[i](teacher_tokens)  # (B, N, proj_dim)

            # Compute loss
            if self.loss_type == 'cosine':
                # Negative cosine similarity (minimize = maximize similarity)
                proj_student_norm = F.normalize(proj_student, dim=-1)
                proj_teacher_norm = F.normalize(proj_teacher, dim=-1)
                layer_loss = 1 - (proj_student_norm * proj_teacher_norm).sum(dim=-1).mean()
            else:  # mse
                layer_loss = F.mse_loss(proj_student, proj_teacher)

            total_loss += layer_loss
            loss_dict[f'tok_loss_layer_{layer_idx}'] = layer_loss.item()

        # Average over layers
        total_loss = total_loss / len(layer_indices)
        loss_dict['tok_loss_total'] = total_loss.item()

        return total_loss, loss_dict


class TokenCorrelationLoss(nn.Module):
    """
    Token correlation distillation loss (L_rel) - LIGHTWEIGHT REGULARIZER.

    Matches token-token correlation matrices between teacher and student
    for structural consistency. Uses patch-mean pooling to avoid O(N²) cost.
    """

    def __init__(self, temperature=0.1, loss_type='kl', use_pooled=True):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_pooled = use_pooled

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute token correlation loss.

        Args:
            student_tokens: (B, N_s, d_s) patch tokens from student
            teacher_tokens: (B, N_t, d_t) patch tokens from teacher

        Returns:
            loss: Scalar loss value
        """
        if self.use_pooled:
            # Patch-mean pooling: (B, N, D) → (B, D) - avoids N² correlation matrix
            student_pooled = student_tokens.mean(dim=1)  # (B, D)
            teacher_pooled = teacher_tokens.mean(dim=1)  # (B, D)

            # Compute batch correlation: (B, B) matrix
            student_norm = F.normalize(student_pooled, dim=-1)
            teacher_norm = F.normalize(teacher_pooled, dim=-1)

            student_corr = student_norm @ student_norm.T  # (B, B)
            teacher_corr = teacher_norm @ teacher_norm.T  # (B, B)
        else:
            # Full correlation (expensive for large N)
            student_norm = F.normalize(student_tokens, dim=-1)
            teacher_norm = F.normalize(teacher_tokens, dim=-1)

            # (B, N, N) correlation matrices
            student_corr = torch.bmm(student_norm, student_norm.transpose(1, 2))
            teacher_corr = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))

        # Apply temperature and normalize
        # Use log_softmax for student (numerically stable) and softmax for teacher
        student_log_prob = F.log_softmax(student_corr / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_corr / self.temperature, dim=-1)

        if self.loss_type == 'kl':
            # KL divergence (more stable for probability matrices)
            loss = F.kl_div(
                student_log_prob,
                teacher_prob,
                reduction='batchmean'
            )
        else:  # frobenius
            student_prob = F.softmax(student_corr / self.temperature, dim=-1)
            loss = torch.norm(student_prob - teacher_prob, p='fro') / student_prob.numel()

        return loss


__all__ = ['ProjectionHead', 'TokenRepresentationLoss', 'TokenCorrelationLoss']
