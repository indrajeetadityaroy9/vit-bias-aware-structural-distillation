"""Combined BASD objective."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from vit_inductive_bias_distillation.losses.attention import AttentionDistillationLoss
from vit_inductive_bias_distillation.losses.gram import GramAnchoringLoss
from vit_inductive_bias_distillation.losses.rsd import RedundancySuppressionLoss
from vit_inductive_bias_distillation.losses.spectral import SpectralMatchingLoss
from vit_inductive_bias_distillation.losses.weighting import UncertaintyWeighting, WarmupSchedule

__all__ = ["BASDLoss", "LossComponent"]


class LossComponent(str, Enum):
    """Distillation loss component identifiers."""
    RSD = "rsd"
    GRAM = "gram"
    ATTN = "attn"
    SPECTRAL = "spectral"


class BASDLoss(nn.Module):
    """Unified BASD objective with all four distillation components."""

    def __init__(
        self,
        base_criterion: nn.Module,
        student_dim: int,
        teacher_dim: int,
        config,
        total_steps: int,
        disable_components: list[str] | None = None,
    ):
        """Build the combined loss and schedules."""
        super().__init__()
        self.base_criterion = base_criterion
        self.token_layers = list(config.token_layers)
        self.attn_layer_pairs = [tuple(p) for p in config.attn_layer_pairs]
        self.disabled: set[str] = {c for c in (disable_components or [])}

        active_components = [c.value for c in LossComponent if c.value not in self.disabled]

        if LossComponent.RSD.value not in self.disabled:
            self.rsd_loss = RedundancySuppressionLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                num_layers=len(self.token_layers),
                kappa=config.rsd_kappa,
            )
        if LossComponent.GRAM.value not in self.disabled:
            self.gram_loss = GramAnchoringLoss()
        if LossComponent.ATTN.value not in self.disabled:
            self.attn_loss = AttentionDistillationLoss()
        if LossComponent.SPECTRAL.value not in self.disabled:
            self.spectral_loss = SpectralMatchingLoss(num_bands=config.spectral_num_bands)

        self.uncertainty = UncertaintyWeighting(active_components)
        self.warmup_schedule = WarmupSchedule(total_steps, config.warmup_fraction)

    def _compute_gram_loss(
        self,
        student_intermediates: dict[int, torch.Tensor],
        raw_teacher_intermediates: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = student_intermediates[self.token_layers[0]].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, float] = {}

        for layer_idx in self.token_layers:
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = raw_teacher_intermediates[layer_idx]
            teacher_interp = torch.nn.functional.interpolate(
                teacher_tokens.transpose(1, 2),
                size=student_tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            layer_loss = self.gram_loss(student_tokens, teacher_interp)
            total_loss = total_loss + layer_loss
            loss_dict[f"gram_layer_{layer_idx}"] = layer_loss.item()

        return total_loss / len(self.token_layers), loss_dict

    def _compute_spectral_loss(
        self,
        student_intermediates: dict[int, torch.Tensor],
        teacher_intermediates: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = student_intermediates[self.token_layers[0]].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, float] = {}

        for layer_idx in self.token_layers:
            layer_loss = self.spectral_loss(
                student_intermediates[layer_idx],
                teacher_intermediates[layer_idx],
            )
            total_loss = total_loss + layer_loss
            loss_dict[f"spectral_layer_{layer_idx}"] = layer_loss.item()

        return total_loss / len(self.token_layers), loss_dict

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        teacher_intermediates: dict[int, torch.Tensor],
        raw_teacher_intermediates: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        teacher_attns: dict[int, torch.Tensor],
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the full BASD objective."""
        ce_loss = self.base_criterion(student_output, targets)

        raw_losses: dict[str, torch.Tensor] = {}
        loss_info: dict[str, float] = {
            "ce_loss": ce_loss.item(),
            "rsd_loss": 0.0,
            "gram_loss": 0.0,
            "attn_loss": 0.0,
            "spectral_loss": 0.0,
        }

        if LossComponent.RSD.value not in self.disabled:
            rsd_loss, rsd_dict = self.rsd_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            raw_losses[LossComponent.RSD.value] = (
                self.warmup_schedule.get_ramp(LossComponent.RSD.value, global_step) * rsd_loss
            )
            loss_info["rsd_loss"] = rsd_loss.item()
            loss_info.update(rsd_dict)

        if LossComponent.GRAM.value not in self.disabled:
            gram_loss, gram_dict = self._compute_gram_loss(
                student_intermediates, raw_teacher_intermediates
            )
            raw_losses[LossComponent.GRAM.value] = (
                self.warmup_schedule.get_ramp(LossComponent.GRAM.value, global_step) * gram_loss
            )
            loss_info["gram_loss"] = gram_loss.item()
            loss_info.update(gram_dict)

        if LossComponent.ATTN.value not in self.disabled:
            attn_loss = self.attn_loss(student_attns, teacher_attns, self.attn_layer_pairs)
            raw_losses[LossComponent.ATTN.value] = (
                self.warmup_schedule.get_ramp(LossComponent.ATTN.value, global_step) * attn_loss
            )
            loss_info["attn_loss"] = attn_loss.item()

        if LossComponent.SPECTRAL.value not in self.disabled:
            spectral_loss, spectral_dict = self._compute_spectral_loss(
                student_intermediates, teacher_intermediates
            )
            raw_losses[LossComponent.SPECTRAL.value] = (
                self.warmup_schedule.get_ramp(LossComponent.SPECTRAL.value, global_step)
                * spectral_loss
            )
            loss_info["spectral_loss"] = spectral_loss.item()
            loss_info.update(spectral_dict)

        weighted_sum, weight_info = self.uncertainty(raw_losses)
        total_loss = ce_loss + weighted_sum
        loss_info.update(weight_info)
        loss_info["total_loss"] = total_loss.item()

        return total_loss, loss_info
