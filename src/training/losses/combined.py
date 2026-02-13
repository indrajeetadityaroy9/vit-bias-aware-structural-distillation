"""
BASD-v2 combined loss with one canonical implementation.
"""

import torch
import torch.nn as nn

from src.training.losses.attention import AttentionDistillationLoss
from src.training.losses.curriculum import UncertaintyWeighting, WarmupSchedule
from src.training.losses.frequency import SpectralMatchingLoss
from src.training.losses.structural import GramAnchoringLoss
from src.training.losses.token import RedundancySuppressionLoss


class BASDv2Loss(nn.Module):
    """Unified BASD objective with all four distillation components enabled."""

    def __init__(self, base_criterion, student_dim, teacher_dim, config):
        super().__init__()
        self.base_criterion = base_criterion
        self.token_layers = config.token_layers
        self.attn_layer_pairs = [tuple(p) for p in config.attn_layer_pairs]

        self.rsd_loss = RedundancySuppressionLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            num_layers=len(self.token_layers),
            kappa=config.rsd_kappa,
        )
        self.gram_loss = GramAnchoringLoss()
        self.attn_loss = AttentionDistillationLoss()
        self.spectral_loss = SpectralMatchingLoss(num_bands=config.spectral_num_bands)
        self.uncertainty = UncertaintyWeighting(['rsd', 'gram', 'attn', 'spectral'])
        self.warmup_schedule = None
        self.warmup_fraction = config.warmup_fraction

    def set_total_steps(self, total_steps):
        self.warmup_schedule = WarmupSchedule(total_steps, self.warmup_fraction)

    def _compute_gram_loss(self, student_intermediates, raw_teacher_intermediates):
        total_loss = 0.0
        loss_dict = {}

        for layer_idx in self.token_layers:
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = raw_teacher_intermediates[layer_idx]
            teacher_interp = torch.nn.functional.interpolate(
                teacher_tokens.transpose(1, 2),
                size=student_tokens.shape[1],
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            layer_loss = self.gram_loss(student_tokens, teacher_interp)
            total_loss += layer_loss
            loss_dict[f'gram_layer_{layer_idx}'] = layer_loss.item()

        return total_loss / len(self.token_layers), loss_dict

    def _compute_spectral_loss(self, student_intermediates, teacher_intermediates):
        total_loss = 0.0
        loss_dict = {}

        for layer_idx in self.token_layers:
            layer_loss = self.spectral_loss(
                student_intermediates[layer_idx],
                teacher_intermediates[layer_idx],
            )
            total_loss += layer_loss
            loss_dict[f'spectral_layer_{layer_idx}'] = layer_loss.item()

        return total_loss / len(self.token_layers), loss_dict

    def forward(
        self,
        student_output,
        targets,
        student_intermediates,
        teacher_intermediates,
        raw_teacher_intermediates,
        student_attns,
        teacher_attns,
        global_step,
    ):
        """Compute the full BASD objective."""
        ce_loss = self.base_criterion(student_output, targets)
        rsd_loss, rsd_dict = self.rsd_loss(student_intermediates, teacher_intermediates, self.token_layers)
        gram_loss, gram_dict = self._compute_gram_loss(student_intermediates, raw_teacher_intermediates)
        attn_loss = self.attn_loss(student_attns, teacher_attns, self.attn_layer_pairs)
        spectral_loss, spectral_dict = self._compute_spectral_loss(student_intermediates, teacher_intermediates)

        raw_losses = {
            'rsd': self.warmup_schedule.get_ramp('rsd', global_step) * rsd_loss,
            'gram': self.warmup_schedule.get_ramp('gram', global_step) * gram_loss,
            'attn': self.warmup_schedule.get_ramp('attn', global_step) * attn_loss,
            'spectral': self.warmup_schedule.get_ramp('spectral', global_step) * spectral_loss,
        }
        weighted_sum, weight_info = self.uncertainty(raw_losses)
        total_loss = ce_loss + weighted_sum

        return total_loss, {
            'ce_loss': ce_loss.item(),
            'rsd_loss': rsd_loss.item(),
            'gram_loss': gram_loss.item(),
            'attn_loss': attn_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'total_loss': total_loss.item(),
            **weight_info,
            **rsd_dict,
            **gram_dict,
            **spectral_dict,
        }
