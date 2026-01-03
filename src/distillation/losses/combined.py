"""
Combined self-supervised distillation loss.

Combines multiple loss components with staged training support:
- L_ce: Cross-entropy classification loss
- L_tok: Token representation loss (primary signal)
- L_rel: Token correlation loss (regularizer, added after warmup)
- L_cka: CKA structural loss (optional)
- L_gram: Gram matrix loss (ablation)

Reference: Chen et al., "Cross-Architecture Self-supervised Video Representation
Learning", CVPR 2022 (CST paper).
"""

import torch
import torch.nn as nn

from .token import TokenRepresentationLoss, TokenCorrelationLoss
from .structural import LayerWiseStructuralLoss


class SelfSupervisedDistillationLoss(nn.Module):
    """
    Combined loss for CST-style self-supervised distillation.

    L = L_ce + lambda_tok * L_tok + lambda_rel * L_rel + lambda_cka * L_cka + lambda_gram * L_gram

    Supports staged training:
    - Stage A (first rel_warmup_epochs): L = L_ce + L_tok
    - Stage B (remaining epochs): Full loss with L_rel
    - CKA/Gram losses enabled after cka_warmup_epochs

    Structural losses (CKA, Gram) are the PRIMARY signal for structural distillation
    experiments. Token losses (L_tok, L_rel) can be disabled when using structural losses.
    """

    def __init__(self, base_criterion, student_dim, teacher_dim, config):
        """
        Args:
            base_criterion: Base classification loss (e.g., LabelSmoothingCE)
            student_dim: Student embedding dimension
            teacher_dim: Teacher embedding dimension
            config: SelfSupervisedDistillationConfig
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.config = config

        # Token representation loss - PRIMARY
        self.token_rep_loss = TokenRepresentationLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            projection_dim=config.projection_dim,
            num_layers=len(config.token_layers),
            loss_type=config.token_loss_type
        )

        # Token correlation loss - REGULARIZER
        self.token_corr_loss = TokenCorrelationLoss(
            temperature=config.correlation_temperature,
            loss_type=config.correlation_loss_type,
            use_pooled=config.use_pooled_correlation
        )

        # CKA structural loss (optional)
        self.use_cka_loss = getattr(config, 'use_cka_loss', False)
        if self.use_cka_loss:
            self.structural_loss = LayerWiseStructuralLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                projection_dim=config.projection_dim,
                num_layers=len(config.token_layers),
                use_cka=True,
                use_gram=getattr(config, 'use_gram_loss', False),
                cka_kernel=getattr(config, 'cka_kernel_type', 'linear')
            )
            self.lambda_cka = getattr(config, 'lambda_cka', 0.5)
            self.cka_warmup_epochs = getattr(config, 'cka_warmup_epochs', 5)
        else:
            self.structural_loss = None
            self.lambda_cka = 0.0
            self.cka_warmup_epochs = 0

        # Gram matrix loss (ablation - standalone without CKA)
        self.use_gram_loss = getattr(config, 'use_gram_loss', False)
        if self.use_gram_loss and not self.use_cka_loss:
            # Gram loss without CKA - create separate loss module
            self.gram_only_loss = LayerWiseStructuralLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                projection_dim=config.projection_dim,
                num_layers=len(config.token_layers),
                use_cka=False,
                use_gram=True
            )
            self.lambda_gram = getattr(config, 'lambda_gram', 0.5)
        else:
            self.gram_only_loss = None
            self.lambda_gram = getattr(config, 'lambda_gram', 0.5) if self.use_cka_loss else 0.0

        self.lambda_tok = config.lambda_tok
        self.lambda_rel = config.lambda_rel
        self.token_layers = config.token_layers

    def get_effective_lambda_rel(self, epoch):
        """Get effective lambda_rel considering warmup."""
        if epoch < self.config.rel_warmup_epochs:
            return 0.0  # Stage A: no correlation loss
        return self.lambda_rel  # Stage B: add L_rel

    def get_effective_lambda_cka(self, epoch):
        """Get effective lambda_cka considering warmup."""
        if not self.use_cka_loss:
            return 0.0
        if epoch < self.cka_warmup_epochs:
            return 0.0
        return self.lambda_cka

    def forward(self, student_output, targets,
                student_intermediates, teacher_intermediates,
                student_patch_tokens, teacher_patch_tokens,
                epoch):
        """
        Compute combined distillation loss.

        Args:
            student_output: (cls_logits, dist_logits) or cls_logits
            targets: Ground truth labels
            student_intermediates: Dict of intermediate student tokens
            teacher_intermediates: Dict of intermediate teacher tokens
            student_patch_tokens: Final student patch tokens (B, N, d_s)
            teacher_patch_tokens: Final teacher patch tokens (B, N, d_t)
            epoch: Current epoch for staged training

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Classification loss (on student CLS head ONLY - critical!)
        if isinstance(student_output, tuple):
            cls_output, dist_output = student_output
        else:
            cls_output = student_output

        ce_loss = self.base_criterion(cls_output, targets)

        # Token representation loss (PRIMARY - always on)
        tok_loss, tok_loss_dict = self.token_rep_loss(
            student_intermediates, teacher_intermediates, self.token_layers
        )

        # Token correlation loss (REGULARIZER - staged)
        effective_lambda_rel = self.get_effective_lambda_rel(epoch)
        if effective_lambda_rel > 0:
            rel_loss = self.token_corr_loss(student_patch_tokens, teacher_patch_tokens)
        else:
            rel_loss = torch.tensor(0.0, device=ce_loss.device)

        # CKA/Gram structural loss (optional - staged)
        effective_lambda_cka = self.get_effective_lambda_cka(epoch)
        cka_loss = torch.tensor(0.0, device=ce_loss.device)
        gram_loss = torch.tensor(0.0, device=ce_loss.device)
        structural_loss_dict = {}

        if self.structural_loss is not None and effective_lambda_cka > 0:
            struct_loss, structural_loss_dict = self.structural_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            if self.use_cka_loss:
                cka_loss = struct_loss
            if self.use_gram_loss and 'gram_loss_total' in structural_loss_dict:
                gram_loss = torch.tensor(structural_loss_dict['gram_loss_total'], device=ce_loss.device)

        # Gram-only loss (ablation without CKA)
        if self.gram_only_loss is not None and epoch >= self.cka_warmup_epochs:
            gram_struct_loss, gram_loss_dict = self.gram_only_loss(
                student_intermediates, teacher_intermediates, self.token_layers
            )
            gram_loss = gram_struct_loss
            structural_loss_dict.update(gram_loss_dict)

        # Combined loss
        total_loss = (
            ce_loss
            + self.lambda_tok * tok_loss
            + effective_lambda_rel * rel_loss
            + effective_lambda_cka * cka_loss
        )
        if self.gram_only_loss is not None:
            total_loss = total_loss + self.lambda_gram * gram_loss

        loss_dict = {
            'ce_loss': ce_loss.item(),
            'tok_loss': tok_loss.item(),
            'rel_loss': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else 0.0,
            'cka_loss': cka_loss.item() if isinstance(cka_loss, torch.Tensor) else 0.0,
            'gram_loss': gram_loss.item() if isinstance(gram_loss, torch.Tensor) else 0.0,
            'effective_lambda_rel': effective_lambda_rel,
            'effective_lambda_cka': effective_lambda_cka,
            'total_loss': total_loss.item(),
            **tok_loss_dict,
            **structural_loss_dict
        }

        return total_loss, loss_dict


__all__ = ['SelfSupervisedDistillationLoss']
