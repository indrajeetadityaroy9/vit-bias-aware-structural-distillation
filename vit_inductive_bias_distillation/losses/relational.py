from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class GeometricRelationalLoss(nn.Module):
    """Procrustes loss with optional CLS-attention weighting.

    Teacher tokens remain attached to preserve projector gradients.
    """

    def __init__(self, *, attn_weighted: Literal["weighted", "unweighted"] = "weighted"):
        super().__init__()
        self.attn_weighted = attn_weighted

    def forward(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        teacher_attn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s = student_tokens.float()
        t = teacher_tokens.float()

        if self.attn_weighted == "weighted" and teacher_attn is not None:
            w = teacher_attn[:, :, 0, 1:].mean(dim=1)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            s = w.unsqueeze(-1) * s
            t = w.unsqueeze(-1) * t

        s_c = s - s.mean(dim=1, keepdim=True)
        t_c = t - t.mean(dim=1, keepdim=True)

        tr_s = (s_c * s_c).sum(dim=(1, 2))
        tr_t = (t_c * t_c).sum(dim=(1, 2))
        cross = torch.bmm(s_c.transpose(1, 2), t_c)
        nuclear = torch.linalg.svdvals(cross).sum(dim=-1)

        return (tr_s + tr_t - 2.0 * nuclear).clamp(min=0.0).mean()
