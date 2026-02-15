"""Cross-attention projector for teacher-student token alignment."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CrossAttentionProjector"]


class CrossAttentionProjector(nn.Module):
    """Align teacher tokens to student token count with learned queries."""

    def __init__(self, num_student_tokens: int, teacher_dim: int, student_dim: int, num_heads: int = 4):
        """Build the projector."""
        super().__init__()
        self.num_student_tokens = num_student_tokens

        self.teacher_proj = nn.Linear(teacher_dim, student_dim)
        self.queries = nn.Parameter(torch.randn(1, num_student_tokens, student_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=student_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(student_dim)

    def forward(self, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Project teacher tokens into the student token layout."""
        B = teacher_tokens.shape[0]

        kv = self.teacher_proj(teacher_tokens)
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        aligned_tokens = self.norm(queries + attn_out)

        return aligned_tokens
