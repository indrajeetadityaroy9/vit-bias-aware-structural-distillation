"""Layer profiling utility for BASD.

Offline CLI tool that computes per-layer Grassmannian projection metric
distances between student and teacher models, helping identify optimal
extraction layers based on subspace geometry alignment.

Usage:
    python -m vit_inductive_bias_distillation.analysis.spectral_profile \\
        configs/experiment/basd_imagenet.yaml --checkpoint path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from vit_inductive_bias_distillation.config import load_config
from vit_inductive_bias_distillation.losses.layer_selector import _grassmann_subspace
from vit_inductive_bias_distillation.models.deit import DeiT
from vit_inductive_bias_distillation.models.teacher import load_teacher


@torch.no_grad()
def grassmann_layer_distance(
    s_tokens: torch.Tensor,
    t_tokens: torch.Tensor,
    proj_dim: int = 128,
    subspace_rank: int = 16,
    cov_eps: float = 1e-4,
) -> float:
    """Compute normalized Grassmannian projection metric distance between two token tensors.

    Creates temporary random orthogonal projections to a common space,
    extracts top-k subspaces, and computes the Grassmannian distance.

    Args:
        s_tokens: (B, N_s, D_s) student tokens.
        t_tokens: (B, N_t, D_t) teacher tokens.
        proj_dim: Dimension of the common projection space.
        subspace_rank: Number of principal directions (k).
        cov_eps: Covariance regularization.

    Returns:
        Normalized Grassmannian distance in [0, 1] (lower = more aligned subspaces).
    """
    D_s = s_tokens.shape[2]
    D_t = t_tokens.shape[2]
    device = s_tokens.device

    # Temporary orthogonal projections
    proj_s = torch.empty(proj_dim, D_s, device=device)
    proj_t = torch.empty(proj_dim, D_t, device=device)
    nn.init.orthogonal_(proj_s)
    nn.init.orthogonal_(proj_t)

    # Project to common space
    s_flat = s_tokens.reshape(-1, D_s).float()
    t_flat = t_tokens.reshape(-1, D_t).float()
    z_s = s_flat @ proj_s.T  # (M_s, d)
    z_t = t_flat @ proj_t.T  # (M_t, d)

    # Extract subspaces
    U_s = _grassmann_subspace(z_s, subspace_rank, cov_eps)  # (d, k)
    U_t = _grassmann_subspace(z_t, subspace_rank, cov_eps)  # (d, k)

    # Grassmannian projection metric
    gram_norm_sq = (U_s.T @ U_t).norm() ** 2
    d_grass_sq = subspace_rank - gram_norm_sq
    return (d_grass_sq / subspace_rank).clamp(min=0.0).item()


@torch.no_grad()
def profile_layers(
    config_path: str,
    checkpoint_path: str | None = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> None:
    """Profile Grassmannian distance between all student-teacher layer pairs."""
    config = load_config(config_path)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    teacher_info = load_teacher(config.basd.teacher_model_name, dev)
    teacher = teacher_info.model
    student = DeiT(config.vit, config.model).to(dev).eval()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])

    imgs = torch.randn(batch_size, 3, config.vit.img_size, config.vit.img_size, device=dev)

    all_layers = list(range(config.vit.depth))

    student_results = student(imgs, layer_indices=all_layers)
    s_intermediates = student_results.intermediates

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

    proj_dim = config.basd.layer_selector_grass_proj_dim
    rank = config.basd.layer_selector_grass_rank
    cov_eps = config.basd.layer_selector_grass_cov_eps

    # Grassmannian distance matrix
    print(f"\nGrassmannian Distance Matrix (d_norm x 100, proj_dim={proj_dim}, rank={rank}):")
    print(f"{'':>6}", end="")
    for tl in all_layers:
        print(f"  T{tl:>2}", end="")
    print()

    for sl in all_layers:
        s_tokens = s_intermediates.get(sl)
        if s_tokens is None:
            continue
        print(f"S{sl:>5}", end="")
        for tl in all_layers:
            t_tokens = t_intermediates.get(tl)
            if t_tokens is None:
                print(f" {'N/A':>4}", end="")
                continue
            d = grassmann_layer_distance(
                s_tokens, t_tokens, proj_dim, rank, cov_eps
            ) * 100
            print(f" {d:>4.1f}", end="")
        print()

    # Recommend extraction layers by minimum Grassmannian distance
    print(f"\nRecommended extraction layers (min Grassmannian distance):")
    for sl in config.basd.token_layers:
        s_tokens = s_intermediates.get(sl)
        if s_tokens is None:
            continue
        best_tl = min(
            all_layers,
            key=lambda tl: (
                grassmann_layer_distance(
                    s_tokens, t_intermediates[tl], proj_dim, rank, cov_eps
                )
                if tl in t_intermediates else float("inf")
            ),
        )
        d = grassmann_layer_distance(
            s_tokens, t_intermediates[best_tl], proj_dim, rank, cov_eps
        ) * 100
        print(f"  Student layer {sl} -> Teacher layer {best_tl} (d_norm x 100 = {d:.2f})")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Layer profiling for BASD")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Student checkpoint path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for profiling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    profile_layers(args.config, args.checkpoint, args.batch_size, args.device)


if __name__ == "__main__":
    main()
