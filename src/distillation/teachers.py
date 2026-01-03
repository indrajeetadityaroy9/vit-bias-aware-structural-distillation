"""
Teacher model loading utilities for knowledge distillation.

Provides:
- load_dino_teacher: Load pretrained DINO/DINOv2 models from torch.hub
- Teacher embedding dimension lookup
"""

import torch
import logging

logger = logging.getLogger(__name__)


# Embedding dimension lookup for DINO/DINOv2 models
DINO_EMBED_DIMS = {
    # DINOv2 models (14x14 patches)
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
    # DINO models (16x16 or 8x8 patches)
    'dino_vits16': 384,
    'dino_vits8': 384,
    'dino_vitb16': 768,
    'dino_vitb8': 768,
}


def load_dino_teacher(teacher_type, model_name, device):
    """
    Load a pretrained DINO/DINOv2 model as teacher.

    Args:
        teacher_type: 'dino' or 'dinov2'
        model_name: Model identifier (e.g., 'dinov2_vits14', 'dino_vits16')
        device: Target device

    Returns:
        teacher_model: Frozen pretrained ViT teacher
        embed_dim: Teacher embedding dimension
    """
    if teacher_type == 'dinov2':
        logger.info(f"Loading DINOv2 teacher: {model_name}")
        teacher_model = torch.hub.load('facebookresearch/dinov2', model_name)
    elif teacher_type == 'dino':
        logger.info(f"Loading DINO teacher: {model_name}")
        teacher_model = torch.hub.load('facebookresearch/dino:main', model_name)
    else:
        raise ValueError(f"Unknown teacher_type: {teacher_type}. Use 'dino' or 'dinov2'.")

    embed_dim = DINO_EMBED_DIMS.get(model_name, 384)

    # Freeze teacher
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    logger.info(f"Loaded {teacher_type} teacher: {model_name} (embed_dim={embed_dim}, frozen)")
    return teacher_model, embed_dim


__all__ = ['load_dino_teacher', 'DINO_EMBED_DIMS']
