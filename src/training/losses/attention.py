"""
Attention distribution distillation loss.

Matches attention patterns between teacher and student via KL divergence
on temperature-scaled attention distributions. Handles head count and
spatial dimension mismatches between architectures.

Temperature is a learnable nn.Parameter with softplus positivity constraint,
allowing the model to discover the optimal sharpening/smoothing level.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistillationLoss(nn.Module):
    """
    KL divergence on temperature-scaled attention distributions.

    Handles mismatches between teacher and student:
    - Head count: averages teacher heads into groups matching student head count
    - Spatial dimensions: bilinear interpolation of attention matrices + re-normalization

    Temperature is learned via nn.Parameter with softplus constraint.
    """

    def __init__(self, init_temperature=1.0):
        """
        Args:
            init_temperature: Initial temperature value (learned during training)
        """
        super().__init__()
        # Store in inverse-softplus space so softplus(raw) = init_temperature
        self._raw_temperature = nn.Parameter(
            torch.tensor(math.log(math.exp(init_temperature) - 1.0))
        )

    @property
    def temperature(self):
        """Learnable temperature with softplus positivity constraint."""
        return F.softplus(self._raw_temperature)

    def _align_heads(self, teacher_attn, num_student_heads):
        """
        Align teacher head count to student by averaging groups of teacher heads.

        Args:
            teacher_attn: (B, H_t, N, N) teacher attention maps
            num_student_heads: int, number of student attention heads

        Returns:
            aligned: (B, H_s, N, N) teacher attention averaged into H_s groups
        """
        B, H_t, N, _ = teacher_attn.shape
        group_size = H_t // num_student_heads
        aligned = teacher_attn.reshape(B, num_student_heads, group_size, N, N).mean(dim=2)
        return aligned  # (B, H_s, N, N)

    def _align_spatial(self, attn, target_size):
        """
        Bilinear interpolation of attention matrices to match spatial dimensions.

        Args:
            attn: (B, H, N_src, N_src) attention maps
            target_size: int, target spatial dimension N_tgt

        Returns:
            aligned: (B, H, N_tgt, N_tgt) interpolated attention maps
        """
        B, H, N, _ = attn.shape

        # Reshape for interpolation: (B*H, 1, N, N) → bilinear → (B*H, 1, N_tgt, N_tgt)
        attn_flat = attn.reshape(B * H, 1, N, N)
        attn_resized = F.interpolate(attn_flat, size=(target_size, target_size), mode='bilinear', align_corners=False)
        attn_resized = attn_resized.reshape(B, H, target_size, target_size)

        # Re-normalize rows to sum to 1 (attention is a distribution)
        attn_resized = attn_resized / (attn_resized.sum(dim=-1, keepdim=True) + 1e-8)

        return attn_resized

    def forward(self, student_attns, teacher_attns, layer_pairs):
        """
        Compute attention distillation loss across layer pairs.

        Args:
            student_attns: Dict[layer_idx] → (B, H_s, N_s, N_s) student attention weights
            teacher_attns: Dict[layer_idx] → (B, H_t, N_t, N_t) teacher attention weights
            layer_pairs: List of (student_layer, teacher_layer) index pairs

        Returns:
            loss: Scalar KL divergence loss averaged across layers
        """
        total_loss = 0.0

        for s_layer, t_layer in layer_pairs:
            s_attn = student_attns[s_layer]  # (B, H_s, N_s, N_s)
            t_attn = teacher_attns[t_layer]  # (B, H_t, N_t, N_t)

            H_s = s_attn.shape[1]
            N_s = s_attn.shape[2]

            # Align teacher heads to student head count
            t_attn = self._align_heads(t_attn, H_s)  # (B, H_s, N_t, N_t)

            # Align spatial dimensions to student size
            t_attn = self._align_spatial(t_attn, N_s)  # (B, H_s, N_s, N_s)

            # Temperature-scaled KL divergence
            s_log_prob = F.log_softmax(s_attn / self.temperature, dim=-1)
            t_prob = F.softmax(t_attn / self.temperature, dim=-1)

            # KL(teacher || student) per row, averaged across all dimensions
            kl = F.kl_div(s_log_prob, t_prob, reduction='batchmean')
            total_loss += kl

        return total_loss / len(layer_pairs)
