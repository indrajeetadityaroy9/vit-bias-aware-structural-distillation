from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn

from vit_inductive_bias_distillation.losses.attention import AttentionDistillationLoss
from vit_inductive_bias_distillation.losses.layer_selector import GrassmannianLayerSelector
from vit_inductive_bias_distillation.losses.relational import RelationalLoss
from vit_inductive_bias_distillation.losses.rsd import RedundancySuppressionLoss
from vit_inductive_bias_distillation.losses.spectral import SpectralMatchingLoss
from vit_inductive_bias_distillation.losses.weighting import UncertaintyWeighting, WarmupSchedule
from vit_inductive_bias_distillation.models.projector import CrossAttentionProjector

__all__ = ["BASDLoss", "LossComponent"]


class LossComponent(str, Enum):
    RSD = "rsd"
    REL = "rel"
    ATTN = "attn"
    SPECTRAL = "spectral"


class BASDLoss(nn.Module):
    def __init__(
        self,
        base_criterion: nn.Module,
        student_dim: int,
        teacher_dim: int,
        num_student_tokens: int,
        cross_attn_num_heads: int,
        config,
        total_steps: int,
        disable_components: list[str] = (),
        student_num_heads: int = 3,
        teacher_num_heads: int = 6,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.token_layers = list(config.token_layers)
        self.disabled: set[str] = set(disable_components)

        active_components = [c.value for c in LossComponent if c.value not in self.disabled]

        self.cross_attn_projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=num_student_tokens,
                teacher_dim=teacher_dim,
                student_dim=teacher_dim,
                num_heads=cross_attn_num_heads,
            )
            for _ in self.token_layers
        ])

        if LossComponent.RSD.value not in self.disabled:
            self.rsd_loss = RedundancySuppressionLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                num_layers=len(self.token_layers),
                kappa=1.0 / teacher_dim,
            )
        if LossComponent.REL.value not in self.disabled:
            self.rel_loss = RelationalLoss(num_pairs=config.vrm_num_pairs)
        if LossComponent.ATTN.value not in self.disabled:
            self.attn_loss = AttentionDistillationLoss(
                student_heads_per_layer=student_num_heads,
                teacher_heads_per_layer=teacher_num_heads,
                num_layers=len(self.token_layers),
            )
        if LossComponent.SPECTRAL.value not in self.disabled:
            self.spectral_loss = SpectralMatchingLoss()

        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            init_temperature=config.layer_selector_temperature,
            diversity_weight=config.layer_selector_diversity_weight,
            recon_weight=config.layer_selector_recon_weight,
            proj_dim=config.layer_selector_grass_proj_dim,
            subspace_rank=config.layer_selector_grass_rank,
            cov_eps=config.layer_selector_grass_cov_eps,
        )

        self.uncertainty = UncertaintyWeighting(
            active_components, temperature=config.uwso_temperature
        )
        self.warmup_schedule = WarmupSchedule(
            total_steps, config.warmup_fraction, stagger=vars(config.warmup_stagger),
        )

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.base_criterion(student_output, targets)

        mixed_tokens, mixed_attns, selector_info, reg_loss = self.layer_selector(
            student_intermediates, all_teacher_tokens, all_teacher_attns,
            self.token_layers,
        )

        aligned_tokens: dict[int, torch.Tensor] = {}
        for i, layer_idx in enumerate(self.token_layers):
            aligned_tokens[layer_idx] = self.cross_attn_projectors[i](
                mixed_tokens[layer_idx]
            )

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
                student_intermediates, aligned_tokens, self.token_layers
            )
            raw_losses[LossComponent.RSD.value] = (
                self.warmup_schedule.get_ramp(LossComponent.RSD.value, global_step) * rsd_loss
            )
            loss_info["rsd_loss"] = rsd_loss.item()
            loss_info.update(rsd_dict)

        if LossComponent.REL.value not in self.disabled:
            device = student_intermediates[self.token_layers[0]].device
            rel_total = torch.tensor(0.0, device=device)
            rel_dict: dict[str, float] = {}
            for layer_idx in self.token_layers:
                layer_loss = self.rel_loss(
                    student_intermediates[layer_idx], aligned_tokens[layer_idx]
                )
                rel_total = rel_total + layer_loss
                rel_dict[f"rel_layer_{layer_idx}"] = layer_loss.item()
            rel_loss = rel_total / len(self.token_layers)
            raw_losses[LossComponent.REL.value] = (
                self.warmup_schedule.get_ramp(LossComponent.REL.value, global_step) * rel_loss
            )
            loss_info["rel_loss"] = rel_loss.item()
            loss_info.update(rel_dict)

        if LossComponent.ATTN.value not in self.disabled:
            attn_loss = self.attn_loss(student_attns, mixed_attns, self.token_layers)
            raw_losses[LossComponent.ATTN.value] = (
                self.warmup_schedule.get_ramp(LossComponent.ATTN.value, global_step) * attn_loss
            )
            loss_info["attn_loss"] = attn_loss.item()

        if LossComponent.SPECTRAL.value not in self.disabled:
            device = student_intermediates[self.token_layers[0]].device
            spectral_total = torch.tensor(0.0, device=device)
            spectral_dict: dict[str, float] = {}
            for layer_idx in self.token_layers:
                layer_loss = self.spectral_loss(
                    student_intermediates[layer_idx], aligned_tokens[layer_idx]
                )
                spectral_total = spectral_total + layer_loss
                spectral_dict[f"spectral_layer_{layer_idx}"] = layer_loss.item()
            spectral_loss = spectral_total / len(self.token_layers)
            raw_losses[LossComponent.SPECTRAL.value] = (
                self.warmup_schedule.get_ramp(LossComponent.SPECTRAL.value, global_step)
                * spectral_loss
            )
            loss_info["spectral_loss"] = spectral_loss.item()
            loss_info.update(spectral_dict)

        weighted_sum, weight_info = self.uncertainty.forward(raw_losses)
        total_loss = ce_loss + weighted_sum + reg_loss
        loss_info.update(weight_info)
        loss_info["selector_reg_loss"] = reg_loss.item()
        loss_info["total_loss"] = total_loss.item()

        return total_loss, loss_info
