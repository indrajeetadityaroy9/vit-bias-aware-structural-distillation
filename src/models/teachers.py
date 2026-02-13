"""Teacher model loading for the canonical DINOv2 BASD path."""

import torch


# Layer count lookup for supported DINOv2 models
DINO_NUM_LAYERS = {
    'dinov2_vits14': 12,
}

# Embedding dimension lookup for supported DINOv2 models
DINO_EMBED_DIMS = {
    'dinov2_vits14': 384,
}


def load_dino_teacher(model_name, device):
    """
    Load a pretrained DINOv2 model as teacher.

    Args:
        model_name: DINOv2 model identifier (e.g., 'dinov2_vits14')
        device: Target device

    Returns:
        teacher_model: Frozen pretrained ViT teacher
        embed_dim: Teacher embedding dimension
    """
    teacher_model = torch.hub.load('facebookresearch/dinov2', model_name)
    embed_dim = DINO_EMBED_DIMS[model_name]

    # Freeze teacher
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"teacher={model_name} dim={embed_dim}")
    return teacher_model, embed_dim
