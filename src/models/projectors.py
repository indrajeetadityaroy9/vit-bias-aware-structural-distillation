"""
Cross-attention projector for semantic teacher-student token alignment.

Replaces bilinear spatial interpolation with learnable cross-attention,
preserving DINOv2's semantic token structure when aligning to student
token counts with different spatial resolutions.
"""

import torch
import torch.nn as nn


class CrossAttentionProjector(nn.Module):
    """
    Cross-attention based projector for teacher→student token alignment.

    Uses learnable query tokens (one per student spatial position) that
    attend to teacher tokens via multi-head cross-attention. This handles
    arbitrary token count mismatches semantically rather than spatially.

    Architecture:
        queries: (N_s, D_s) learnable parameters
        cross_attn: MultiheadAttention(D_s, num_heads)
        norm: LayerNorm(D_s)
        proj: Linear(D_t, D_s)  [projects teacher dim to student dim first]

    Note (v2): In BASD-v2, student_dim is set to teacher_dim so the
    projector preserves teacher dimension throughout. The AAD in
    RedundancySuppressionLoss handles the actual D_t → D_s mapping.
    """

    def __init__(self, num_student_tokens, teacher_dim, student_dim, num_heads=4):
        """
        Args:
            num_student_tokens: Number of student patch tokens (N_s)
            teacher_dim: Teacher embedding dimension (D_t)
            student_dim: Student embedding dimension (D_s)
            num_heads: Number of attention heads for cross-attention
        """
        super().__init__()
        self.num_student_tokens = num_student_tokens

        # Project teacher dim to student dim for cross-attention compatibility
        self.teacher_proj = nn.Linear(teacher_dim, student_dim)

        # Learnable queries — one per student token position
        self.queries = nn.Parameter(torch.randn(1, num_student_tokens, student_dim) * 0.02)

        # Cross-attention: queries attend to projected teacher tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=student_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(student_dim)

    def forward(self, teacher_tokens):
        """
        Project teacher tokens to student token space via cross-attention.

        Args:
            teacher_tokens: (B, N_t, D_t) teacher patch token representations

        Returns:
            aligned_tokens: (B, N_s, D_s) aligned to student token count and dim
        """
        B = teacher_tokens.shape[0]

        # Project teacher to student dimension
        kv = self.teacher_proj(teacher_tokens)  # (B, N_t, D_s)

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # (B, N_s, D_s)

        # Cross-attention: student queries attend to teacher key/values
        attn_out, _ = self.cross_attn(queries, kv, kv)  # (B, N_s, D_s)

        # Residual + LayerNorm
        aligned_tokens = self.norm(queries + attn_out)  # (B, N_s, D_s)

        return aligned_tokens
