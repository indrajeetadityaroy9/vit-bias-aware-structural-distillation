"""DINOv2 teacher loading and intermediate extraction."""

from __future__ import annotations

from typing import NamedTuple

import torch

__all__ = [
    "DINO_EMBED_DIMS",
    "load_teacher",
    "extract_intermediates",
    "TeacherModel",
    "TeacherIntermediates",
]

DINO_EMBED_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
}


class TeacherModel(NamedTuple):
    model: torch.nn.Module
    embed_dim: int


class TeacherIntermediates(NamedTuple):
    projected: dict[int, torch.Tensor]
    raw: dict[int, torch.Tensor]
    attentions: dict[int, torch.Tensor]
    all_raw: dict[int, torch.Tensor]


def load_teacher(model_name: str, device: torch.device) -> TeacherModel:
    """Load a frozen pretrained DINOv2 teacher."""
    teacher_model = torch.hub.load("facebookresearch/dinov2", model_name)
    embed_dim = DINO_EMBED_DIMS[model_name]

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(model_name, embed_dim)
    return TeacherModel(model=teacher_model, embed_dim=embed_dim)


@torch.no_grad()
def extract_intermediates(
    teacher_model: torch.nn.Module,
    x: torch.Tensor,
    token_layers: list[int],
    attn_layers: list[int],
    projectors: torch.nn.ModuleList,
    all_token_layers: list[int] | None = None,
) -> TeacherIntermediates:
    """Extract intermediate tokens and attention maps in one teacher pass.

    Args:
        teacher_model: Frozen DINOv2 teacher.
        x: Input images (B, C, H, W).
        token_layers: Layer indices for projection (paired with projectors).
        attn_layers: Layer indices for attention extraction.
        projectors: Cross-attention projectors (one per token_layer).
        all_token_layers: Additional layers to extract raw tokens from
            (for adaptive layer selection). When provided, raw tokens are
            captured from all specified layers and returned in ``all_raw``.
    """
    hooks: list[torch.utils.hooks.RemovableHook] = []
    captured_tokens: dict[int, torch.Tensor] = {}
    captured_attns: dict[int, torch.Tensor] = {}
    extract_token_layers = set(token_layers) | set(all_token_layers or [])
    extract_attn_layers = set(attn_layers)
    all_layers = extract_token_layers | extract_attn_layers

    for layer_idx in all_layers:
        block = teacher_model.blocks[layer_idx]

        if layer_idx in extract_token_layers:
            def make_token_hook(idx: int):
                def hook(module, input, output):
                    captured_tokens[idx] = output[:, 1:, :]
                return hook
            hooks.append(block.register_forward_hook(make_token_hook(layer_idx)))

        if layer_idx in extract_attn_layers:
            def make_attn_hook(idx: int):
                def hook(module, input, output):
                    batch_size, num_tokens, channels = input[0].shape
                    num_heads = module.num_heads
                    head_dim = channels // num_heads
                    qkv = module.qkv(input[0]).reshape(
                        batch_size, num_tokens, 3, num_heads, head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q, k, _ = qkv.unbind(0)
                    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                    captured_attns[idx] = attn.softmax(dim=-1).detach()
                return hook
            hooks.append(block.attn.register_forward_hook(make_attn_hook(layer_idx)))

    try:
        teacher_model(x)
    finally:
        for hook in hooks:
            hook.remove()

    intermediates: dict[int, torch.Tensor] = {}
    raw_intermediates: dict[int, torch.Tensor] = {}
    for i, layer_idx in enumerate(token_layers):
        raw_intermediates[layer_idx] = captured_tokens[layer_idx]
        intermediates[layer_idx] = projectors[i](captured_tokens[layer_idx])

    # All raw tokens for adaptive layer selection
    all_raw: dict[int, torch.Tensor] = {
        idx: captured_tokens[idx]
        for idx in sorted(captured_tokens)
    }

    return TeacherIntermediates(
        projected=intermediates,
        raw=raw_intermediates,
        attentions=captured_attns,
        all_raw=all_raw,
    )
