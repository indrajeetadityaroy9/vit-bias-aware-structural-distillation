"""Combined BASD objective."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_inductive_bias_distillation.losses.attention import AttentionDistillationLoss
from vit_inductive_bias_distillation.losses.relational import RelationalLoss
from vit_inductive_bias_distillation.losses.layer_selector import SpectralLayerSelector
from vit_inductive_bias_distillation.losses.rsd import RedundancySuppressionLoss
from vit_inductive_bias_distillation.losses.spectral import SpectralMatchingLoss
from vit_inductive_bias_distillation.losses.weighting import UncertaintyWeighting, WarmupSchedule

__all__ = ["BASDLoss", "LossComponent"]


class LossComponent(str, Enum):
    """Distillation loss component identifiers."""
    RSD = "rsd"
    REL = "rel"
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
        student_num_heads: int = 3,
        teacher_num_heads: int = 6,
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
        if LossComponent.REL.value not in self.disabled:
            self.rel_loss = RelationalLoss(num_pairs=config.vrm_num_pairs)
        if LossComponent.ATTN.value not in self.disabled:
            self.attn_loss = AttentionDistillationLoss(
                student_heads_per_layer=student_num_heads,
                teacher_heads_per_layer=teacher_num_heads,
                num_layers=len(self.attn_layer_pairs),
            )
        if LossComponent.SPECTRAL.value not in self.disabled:
            self.spectral_loss = SpectralMatchingLoss()

        self.layer_selector = SpectralLayerSelector(
            num_extraction_points=len(self.token_layers),
            num_teacher_layers=config.num_teacher_layers,
            init_temperature=config.layer_selector_temperature,
            entropy_weight=config.layer_selector_entropy_weight,
        )

        self.uncertainty = UncertaintyWeighting(
            active_components, temperature=config.uwso_temperature
        )
        self.warmup_schedule = WarmupSchedule(total_steps, config.warmup_fraction)

    def _interpolate_tokens_2d(
        self, tokens: torch.Tensor, target_n: int
    ) -> torch.Tensor:
        """2D bilinear interpolation of token sequences."""
        B, N, D = tokens.shape
        if N == target_n:
            return tokens
        H_src = W_src = int(N ** 0.5)
        H_tgt = W_tgt = int(target_n ** 0.5)
        tokens_2d = tokens.transpose(1, 2).reshape(B, D, H_src, W_src)
        tokens_2d = F.interpolate(
            tokens_2d, (H_tgt, W_tgt), mode="bilinear", align_corners=False
        )
        return tokens_2d.reshape(B, D, -1).transpose(1, 2)

    def _compute_rel_loss(
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
            # 2D bilinear interpolation to align token counts
            teacher_tokens = self._interpolate_tokens_2d(
                teacher_tokens, student_tokens.shape[1]
            )
            layer_loss = self.rel_loss(student_tokens, teacher_tokens)
            total_loss = total_loss + layer_loss
            loss_dict[f"rel_layer_{layer_idx}"] = layer_loss.item()

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
            s_tokens = student_intermediates[layer_idx]
            t_tokens = teacher_intermediates[layer_idx]
            # 2D interpolation fallback for robustness
            t_tokens = self._interpolate_tokens_2d(t_tokens, s_tokens.shape[1])
            layer_loss = self.spectral_loss(s_tokens, t_tokens)
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
        all_teacher_tokens: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        teacher_attns: dict[int, torch.Tensor],
        projectors: nn.ModuleList,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the full BASD objective."""
        ce_loss = self.base_criterion(student_output, targets)

        # Adaptive spectral layer selection: mix teacher layers per extraction point
        mixed_teachers, selector_info, entropy_loss = self.layer_selector(
            student_intermediates, all_teacher_tokens, self.token_layers,
        )
        # Build new dicts with mixed teacher tokens (avoid mutating caller's dicts)
        raw_teacher_intermediates = dict(raw_teacher_intermediates)
        teacher_intermediates = dict(teacher_intermediates)
        for i, layer_idx in enumerate(self.token_layers):
            raw_teacher_intermediates[layer_idx] = mixed_teachers[layer_idx]
            teacher_intermediates[layer_idx] = projectors[i](mixed_teachers[layer_idx])

        raw_losses: dict[str, torch.Tensor] = {}
        loss_info: dict[str, float] = {
            "ce_loss": ce_loss.item(),
            "rsd_loss": 0.0,
            "rel_loss": 0.0,
            "attn_loss": 0.0,
            "spectral_loss": 0.0,
        }
        loss_info.update(selector_info)

        if LossComponent.RSD.value not in self.disabled:
            rsd_loss, rsd_dict = self.rsd_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            raw_losses[LossComponent.RSD.value] = (
                self.warmup_schedule.get_ramp(LossComponent.RSD.value, global_step) * rsd_loss
            )
            loss_info["rsd_loss"] = rsd_loss.item()
            loss_info.update(rsd_dict)

        if LossComponent.REL.value not in self.disabled:
            rel_loss, rel_dict = self._compute_rel_loss(
                student_intermediates, raw_teacher_intermediates
            )
            raw_losses[LossComponent.REL.value] = (
                self.warmup_schedule.get_ramp(LossComponent.REL.value, global_step) * rel_loss
            )
            loss_info["rel_loss"] = rel_loss.item()
            loss_info.update(rel_dict)

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

        weighted_sum, weight_info = self.uncertainty.forward(raw_losses)
        total_loss = ce_loss + weighted_sum + entropy_loss
        loss_info.update(weight_info)
        loss_info["entropy_loss"] = entropy_loss.item()
        loss_info["total_loss"] = total_loss.item()

        return total_loss, loss_info
