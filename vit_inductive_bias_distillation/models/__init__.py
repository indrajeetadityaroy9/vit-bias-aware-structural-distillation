"""BASD model components."""

from __future__ import annotations

import torch

from vit_inductive_bias_distillation.config import BASDExperimentConfig
from vit_inductive_bias_distillation.models.deit import DeiT, StudentIntermediates
from vit_inductive_bias_distillation.models.projector import CrossAttentionProjector
from vit_inductive_bias_distillation.models.teacher import (
    TeacherIntermediates,
    TeacherModel,
    extract_intermediates,
    load_teacher,
)

__all__ = [
    "DeiT",
    "StudentIntermediates",
    "CrossAttentionProjector",
    "TeacherModel",
    "TeacherIntermediates",
    "build_student_model",
    "load_teacher",
    "extract_intermediates",
]


def build_student_model(config: BASDExperimentConfig, device: torch.device) -> DeiT:
    """Construct a DeiT student model from config."""
    return DeiT(config.vit, config.model).to(device)
