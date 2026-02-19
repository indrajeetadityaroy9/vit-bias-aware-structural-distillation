"""Spectral-guided layer selection utility for BASD.

Offline CLI tool that computes per-layer 2D spatial spectral energy profiles
for student and teacher models, helping identify optimal extraction layers.

Uses 2D spatial FFT (rfft2) on the token grid — physically meaningful because
spatial frequencies capture real image structure (edges vs textures vs global
patterns). Both student and teacher share the same 14x14 spatial grid, so
frequency bins align perfectly with no dimension mismatch.

Usage:
    python -m vit_inductive_bias_distillation.analysis.spectral_profile \\
        configs/experiment/basd_imagenet.yaml --checkpoint path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from vit_inductive_bias_distillation.config import load_config
from vit_inductive_bias_distillation.models.deit import DeiT
from vit_inductive_bias_distillation.models.teacher import load_teacher


def spectral_energy(tokens: torch.Tensor) -> torch.Tensor:
    """Compute mean spatial spectral energy from token features.

    Uses 2D spatial FFT on the token grid (H, W), averaged over batch
    and channel dimensions.

    Args:
        tokens: (B, N, D) token features from a single layer.

    Returns:
        Scalar mean spectral energy.
    """
    B, N, D = tokens.shape
    H = W = int(N ** 0.5)
    spatial = tokens.permute(0, 2, 1).reshape(B, D, H, W)
    freq = torch.fft.rfft2(spatial, norm="ortho")
    energy = (freq.real ** 2 + freq.imag ** 2).mean()
    return energy


def spatial_spectral_profile(tokens: torch.Tensor) -> torch.Tensor:
    """Compute 2D spatial spectral energy profile.

    Produces an energy distribution over spatial frequency bins, averaged
    over batch and channel dimensions. Output shape depends only on the
    spatial grid size (not embedding dimension), so student and teacher
    profiles are directly comparable.

    Args:
        tokens: (B, N, D) token features.

    Returns:
        (H * (W//2+1),) energy profile. For 14x14 grid: (112,).
    """
    B, N, D = tokens.shape
    H = W = int(N ** 0.5)
    spatial = tokens.permute(0, 2, 1).reshape(B, D, H, W)
    freq = torch.fft.rfft2(spatial, norm="ortho")  # (B, D, H, W//2+1)
    energy = (freq.real ** 2 + freq.imag ** 2).mean(dim=(0, 1))  # (H, W//2+1)
    return energy.flatten()  # (H * (W//2+1),)


def cosine_sim_spectral(s_tokens: torch.Tensor, t_tokens: torch.Tensor) -> float:
    """Compute cosine similarity between 2D spatial spectral energy profiles.

    Both inputs produce profiles of identical shape (determined by spatial
    grid, not embedding dimension), so no truncation or alignment is needed.
    """
    s_profile = spatial_spectral_profile(s_tokens)
    t_profile = spatial_spectral_profile(t_tokens)
    return F.cosine_similarity(s_profile.unsqueeze(0), t_profile.unsqueeze(0)).item()


@torch.no_grad()
def profile_layers(
    config_path: str,
    checkpoint_path: str | None = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> None:
    """Profile spectral energy for all 12 layers of student and teacher."""
    config = load_config(config_path)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load models
    teacher_info = load_teacher(config.basd.teacher_model_name, dev)
    teacher = teacher_info.model
    student = DeiT(config.vit, config.model).to(dev).eval()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])

    # Synthetic batch
    imgs = torch.randn(batch_size, 3, config.vit.img_size, config.vit.img_size, device=dev)

    all_layers = list(range(config.vit.depth))

    # Student intermediates
    student_results = student(imgs, layer_indices=all_layers)
    s_intermediates = student_results.intermediates

    # Teacher intermediates via hooks on all layers
    t_intermediates: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_idx in all_layers:
        def make_hook(idx: int):
            def hook(module, input, output):
                t_intermediates[idx] = output[:, 1:, :].detach()
            return hook
        hooks.append(teacher.blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
    try:
        teacher(imgs)
    finally:
        for h in hooks:
            h.remove()

    # Compute profiles
    print(f"\n{'Layer':>6} | {'Student Energy':>15} | {'Teacher Energy':>15} | {'Cosine Sim':>12}")
    print("-" * 60)

    s_energies = []
    t_energies = []
    for layer in all_layers:
        s_tokens = s_intermediates.get(layer)
        t_tokens = t_intermediates.get(layer)

        if s_tokens is None or t_tokens is None:
            print(f"{layer:>6} | {'N/A':>15} | {'N/A':>15} | {'N/A':>12}")
            continue

        s_e = spectral_energy(s_tokens).item()
        t_e = spectral_energy(t_tokens).item()
        cos_sim = cosine_sim_spectral(s_tokens, t_tokens)

        s_energies.append((layer, s_e))
        t_energies.append((layer, t_e))

        print(f"{layer:>6} | {s_e:>15.4f} | {t_e:>15.4f} | {cos_sim:>12.4f}")

    # Recommend layers with highest combined spectral energy
    if s_energies:
        combined = [(l, se + te) for (l, se), (_, te) in zip(s_energies, t_energies)]
        combined.sort(key=lambda x: x[1], reverse=True)
        top_4 = [l for l, _ in combined[:4]]
        top_4.sort()
        print(f"\nRecommended extraction layers (top-4 by spectral energy): {top_4}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Spectral layer profiling for BASD")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Student checkpoint path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    profile_layers(args.config, args.checkpoint, args.batch_size, args.device)


if __name__ == "__main__":
    main()
