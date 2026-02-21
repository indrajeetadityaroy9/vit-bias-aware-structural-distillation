from __future__ import annotations

from typing import NamedTuple

import timm
import torch

__all__ = [
    "load_teacher",
    "extract_intermediates",
    "TeacherModel",
    "TeacherIntermediates",
]


class TeacherModel(NamedTuple):
    model: torch.nn.Module
    embed_dim: int
    num_layers: int
    num_heads: int
    loader: str
    feature_format: str


class TeacherIntermediates(NamedTuple):
    all_tokens: dict[int, torch.Tensor]
    all_attentions: dict[int, torch.Tensor]


def load_teacher(model_name: str, device: torch.device) -> TeacherModel:
    is_dinov2 = model_name.startswith("dinov2_")

    if is_dinov2:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if is_dinov2:
        embed_dim = model.embed_dim
        num_layers = len(model.blocks)
        num_heads = model.blocks[0].attn.num_heads
        feature_format = "token"
        loader = "dinov2"
    else:
        embed_dim = model.num_features
        num_layers = 1
        num_heads = max(1, embed_dim // 64)
        probe = torch.zeros(1, 3, 224, 224, device=device)
        features = model.forward_features(probe)
        if features.dim() == 3:
            feature_format = "token"
        elif features.shape[1] > features.shape[3]:
            feature_format = "nchw"
        else:
            feature_format = "nhwc"
        loader = "timm"

    print(
        f"event=teacher_loaded model={model_name} embed_dim={embed_dim} "
        f"num_layers={num_layers} num_heads={num_heads} feature_format={feature_format}"
    )
    return TeacherModel(
        model=model,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        loader=loader,
        feature_format=feature_format,
    )


def _extract_vit(
    teacher_model: torch.nn.Module,
    x: torch.Tensor,
    num_layers: int,
) -> TeacherIntermediates:
    hooks: list[torch.utils.hooks.RemovableHook] = []
    captured_tokens: dict[int, torch.Tensor] = {}
    captured_attns: dict[int, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        block = teacher_model.blocks[layer_idx]

        def make_token_hook(idx: int):
            def hook(module, input, output):
                captured_tokens[idx] = output[:, 1:, :]
            return hook
        hooks.append(block.register_forward_hook(make_token_hook(layer_idx)))

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

    teacher_model(x)
    for hook in hooks:
        hook.remove()

    return TeacherIntermediates(
        all_tokens={idx: captured_tokens[idx] for idx in sorted(captured_tokens)},
        all_attentions={idx: captured_attns[idx] for idx in sorted(captured_attns)},
    )


@torch.no_grad()
def extract_intermediates(
    teacher_model: torch.nn.Module,
    x: torch.Tensor,
    num_layers: int,
    loader: str,
    feature_format: str,
) -> TeacherIntermediates:
    if loader == "dinov2":
        return _extract_vit(teacher_model, x, num_layers)
    features = teacher_model.forward_features(x)
    if feature_format == "nhwc":
        features = features.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
    else:
        features = features.flatten(2).transpose(1, 2)
    return TeacherIntermediates(all_tokens={0: features}, all_attentions={})
