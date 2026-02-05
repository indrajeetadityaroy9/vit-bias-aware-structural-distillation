"""
Standard knowledge distillation loss functions.

Implements DeiT-style hard/soft distillation using teacher predictions
to guide the student's distillation token.

Reference: Touvron et al., "Training data-efficient image transformers
& distillation through attention", ICML 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for DeiT distillation training.

    Supports two distillation modes:
    1. Hard distillation (default): Uses argmax of teacher predictions
       loss = (1-alpha)*CE(cls_out, targets) + alpha*CE(dist_out, argmax(teacher))

    2. Soft distillation: Uses temperature-scaled KL divergence
       loss = (1-alpha)*CE(cls_out, targets) + alpha*tau^2*KL(dist_out/tau, teacher/tau)

    Args:
        base_criterion: Loss function for ground truth (e.g., LabelSmoothingCrossEntropy)
        distillation_type: 'hard' or 'soft'
        alpha: Weight for distillation loss (0 = no distillation, 1 = only distillation)
        tau: Temperature for soft distillation
    """

    def __init__(self, base_criterion, distillation_type='hard', alpha=0.5, tau=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.distillation_type = distillation_type

        # Validate alpha is in [0, 1]
        if not 0 <= alpha <= 1:
            raise ValueError(f"Distillation alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

        # Validate tau is positive (for soft distillation)
        if tau <= 0:
            raise ValueError(f"Distillation tau must be positive, got {tau}")
        self.tau = tau

    def forward(self, student_cls_output, student_dist_output, targets, teacher_output):
        """
        Compute distillation loss.

        Args:
            student_cls_output: Student [CLS] token predictions (B, num_classes)
            student_dist_output: Student [DIST] token predictions (B, num_classes)
            targets: Ground truth labels (B,) or (B, num_classes) for soft labels
            teacher_output: Teacher model predictions (B, num_classes)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Ground truth loss on [CLS] token
        cls_loss = self.base_criterion(student_cls_output, targets)

        if self.distillation_type == 'hard':
            # Hard labels from teacher (argmax)
            teacher_labels = teacher_output.argmax(dim=1)
            dist_loss = F.cross_entropy(student_dist_output, teacher_labels)
        else:
            # Soft distillation with temperature scaling
            soft_teacher = F.softmax(teacher_output / self.tau, dim=1)
            soft_student = F.log_softmax(student_dist_output / self.tau, dim=1)
            dist_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            # Scale by tau^2 as per Hinton et al.
            dist_loss = dist_loss * (self.tau ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * dist_loss

        loss_dict = {
            'cls_loss': cls_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict
