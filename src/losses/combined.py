import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.layer_selector import GrassmannianLayerSelector
from src.losses.relational import geometric_relational_loss


def _align_token_count(tokens: torch.Tensor, target_n: int) -> torch.Tensor:
    if tokens.shape[1] == target_n:
        return tokens
    return F.interpolate(
        tokens.transpose(1, 2), size=target_n, mode="linear", align_corners=False,
    ).transpose(1, 2)


class BASDLoss(nn.Module):
    def __init__(
        self,
        base_criterion: nn.Module,
        student_dim: int,
        teacher_dim: int,
        student_depth: int,
        num_student_tokens: int,
        *,
        config,
        teacher_has_cls_token: bool,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_has_cls_token = teacher_has_cls_token
        self.num_student_tokens = num_student_tokens

        if config.num_extraction_points == 1:
            self.token_layers = [student_depth - 1]
        else:
            self.token_layers = [
                round(i * (student_depth - 1) / (config.num_extraction_points - 1))
                for i in range(config.num_extraction_points)
            ]

        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        )

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        ce_loss = self.base_criterion(student_output, targets)

        mixed_tokens, mixed_attns = self.layer_selector(
            student_intermediates, all_teacher_tokens, all_teacher_attns,
            self.token_layers,
        )

        aligned_tokens = {}
        for layer_idx in self.token_layers:
            aligned_tokens[layer_idx] = _align_token_count(
                mixed_tokens[layer_idx], self.num_student_tokens,
            )

        geo_losses = []
        for layer_idx in self.token_layers:
            geo_losses.append(geometric_relational_loss(
                student_intermediates[layer_idx], aligned_tokens[layer_idx],
                mixed_attns[layer_idx],
                has_cls_token=self.teacher_has_cls_token,
            ))
        geo_loss = torch.stack(geo_losses).mean()

        vals = [ce_loss, geo_loss]

        # UW-SO weighting (Kirchdorfer et al. 2024): w_i = (1/L_i) / Σ(1/L_j)
        eps = torch.finfo(vals[0].dtype).eps
        inv = torch.stack([1.0 / v.detach().clamp(min=eps) for v in vals])
        w = inv / inv.sum()

        return sum(w[i] * vals[i] for i in range(len(vals)))
