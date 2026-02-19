"""Grassmannian-geometry-guided teacher layer selection with subspace diversity.

Adaptively mixes teacher layers for each student extraction point using the
Grassmannian projection metric on Gr(d, k) as the compatibility measure and
a weight-modulated subspace orthogonality penalty to encourage geometric
coverage across selected layers.

Novel contributions over prior KD layer selection:
1. Learnable common-space projections φ_s, φ_t map heterogeneous student/teacher
   representations to a shared d-dimensional space — fully architecture-agnostic.
2. Grassmannian projection metric d_p²(U, V) = k - ‖U^T V‖_F² is basis-invariant
   and operates on the manifold of k-dimensional subspaces.
3. Subspace orthogonality penalty Σ_{l<l'} w_l w_{l'} ‖U_l^T U_{l'}‖_F² weighted
   by mixing probabilities provides geometrically meaningful diversity.

References:
    arXiv 2511.08628 (AAAI 2026) — Grassmannian deep networks, projection metric
    arXiv 2507.17998 (CVPR 2025) — Grassmannian geodesic alignment
    arXiv 2506.01599 (NeurIPS 2025) — Relative geodesic representations
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GrassmannianLayerSelector"]


def _grassmann_subspace(
    z_flat: torch.Tensor,
    k: int,
    cov_eps: float,
) -> torch.Tensor:
    """Extract top-k eigenvectors of the feature covariance — a point on Gr(d, k).

    Must be called under ``torch.no_grad()``; no backward through eigh.

    Args:
        z_flat: (M, d) flattened, centered tokens in float32.
        k: Subspace rank (number of principal directions to keep).
        cov_eps: Regularization added to covariance diagonal for numerical stability.

    Returns:
        (d, k) orthonormal matrix whose columns span the top-k subspace.
    """
    z = z_flat.float()
    z = z - z.mean(dim=0, keepdim=True)
    M, d = z.shape
    cov = z.T @ z / M  # (d, d)
    cov = cov + cov_eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    # eigh returns eigenvalues in ascending order
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Take the k largest eigenvectors (last k columns)
    return eigvecs[:, -k:]  # (d, k)


class GrassmannianLayerSelector(nn.Module):
    """Grassmannian projection metric–based soft teacher layer selection.

    For each student extraction point, projects student and teacher tokens into
    a learned common space, extracts top-k subspaces via eigendecomposition,
    and computes Grassmannian distances to derive mixing weights. A subspace
    orthogonality penalty encourages diversity, while a reconstruction loss
    provides gradient signal to the learned projections.
    """

    def __init__(
        self,
        num_extraction_points: int,
        student_dim: int,
        teacher_dim: int,
        num_teacher_layers: int = 12,
        init_temperature: float = 2.0,
        diversity_weight: float = 0.01,
        recon_weight: float = 0.01,
        proj_dim: int = 128,
        subspace_rank: int = 16,
        cov_eps: float = 1e-4,
    ):
        super().__init__()
        assert subspace_rank < proj_dim, (
            f"subspace_rank ({subspace_rank}) must be < proj_dim ({proj_dim})"
        )
        self.num_teacher_layers = num_teacher_layers
        self.diversity_weight = diversity_weight
        self.recon_weight = recon_weight
        self.proj_dim = proj_dim
        self.subspace_rank = subspace_rank
        self.cov_eps = cov_eps

        # Learnable common-space projections (no bias — pure linear map)
        self.phi_s = nn.Linear(student_dim, proj_dim, bias=False)
        self.phi_t = nn.Linear(teacher_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.phi_s.weight)
        nn.init.orthogonal_(self.phi_t.weight)

        # Per-extraction-point learnable temperature (softplus reparameterization)
        self.log_temperatures = nn.Parameter(
            torch.full(
                (num_extraction_points,),
                math.log(math.exp(init_temperature) - 1),
            )
        )

    @property
    def temperatures(self) -> torch.Tensor:
        """Softplus ensures τ > 0."""
        return F.softplus(self.log_temperatures)

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[str, float], torch.Tensor]:
        """Select and mix teacher layers for each student extraction point.

        Args:
            student_tokens_per_layer: {student_layer: (B, N_s, D_s)} selected layers.
            all_teacher_tokens: {0..L-1: (B, N_t, D_t)} raw tokens from all teacher layers.
            extraction_indices: list of student layer indices, e.g. [3, 6, 9, 11].

        Returns:
            mixed_teachers: {student_layer: (B, N_t, D_t)} one mixed teacher per point.
            info: per-layer diagnostic values for logging.
            reg_loss: scalar regularization loss (diversity + reconstruction).
        """
        teacher_indices = sorted(all_teacher_tokens.keys())
        L = len(teacher_indices)
        B, N_t, D_t = all_teacher_tokens[teacher_indices[0]].shape

        # Stack raw teacher tokens for weighted mixing later
        stacked_teacher = torch.stack(
            [all_teacher_tokens[idx] for idx in teacher_indices]
        )  # (L, B, N_t, D_t)

        # --- Path A: Project teachers to common space, extract subspaces (detached) ---
        # Teacher tokens are already detached (from @torch.no_grad extract_intermediates)
        # phi_t gets gradient through Path C (reconstruction loss)
        z_teachers: dict[int, torch.Tensor] = {}
        for idx in teacher_indices:
            t_flat = all_teacher_tokens[idx].reshape(-1, D_t)  # (B*N_t, D_t)
            z_teachers[idx] = self.phi_t(t_flat)  # (B*N_t, d), gradient enabled

        # Extract teacher subspaces — fully detached (no eigh backward)
        teacher_subspaces: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for idx in teacher_indices:
                teacher_subspaces[idx] = _grassmann_subspace(
                    z_teachers[idx].detach(), self.subspace_rank, self.cov_eps
                )  # (d, k)

        # Pre-compute teacher cross-Gram matrix for orthogonality penalty (detached)
        # cross_gram[l, l'] = ‖U_l^T U_{l'}‖_F²
        cross_gram = torch.zeros(L, L, device=stacked_teacher.device)
        with torch.no_grad():
            for a in range(L):
                U_a = teacher_subspaces[teacher_indices[a]]
                for b in range(a + 1, L):
                    U_b = teacher_subspaces[teacher_indices[b]]
                    gram_val = (U_a.T @ U_b).norm() ** 2  # ‖U_a^T U_b‖_F²
                    cross_gram[a, b] = gram_val
                    cross_gram[b, a] = gram_val

        # --- Per extraction point ---
        mixed_teachers: dict[int, torch.Tensor] = {}
        info: dict[str, float] = {}
        device = stacked_teacher.device
        input_dtype = stacked_teacher.dtype

        total_orth = torch.tensor(0.0, device=device)
        total_recon = torch.tensor(0.0, device=device)

        k = self.subspace_rank

        for i, s_layer in enumerate(extraction_indices):
            s_tokens = student_tokens_per_layer[s_layer]  # (B, N_s, D_s)
            D_s = s_tokens.shape[2]

            # Project student tokens — DETACHED input, phi_s gets gradient
            s_flat = s_tokens.detach().reshape(-1, D_s)  # (B*N_s, D_s)
            z_s = self.phi_s(s_flat)  # (B*N_s, d), gradient flows to phi_s.weight

            # Student subspace — DETACHED
            with torch.no_grad():
                U_s = _grassmann_subspace(z_s.detach(), k, self.cov_eps)  # (d, k)

            # Grassmannian projection metric distances — detached scalars
            d_grass_sq = torch.zeros(L, device=device)
            with torch.no_grad():
                for j, t_idx in enumerate(teacher_indices):
                    U_t = teacher_subspaces[t_idx]  # (d, k)
                    gram_norm_sq = (U_s.T @ U_t).norm() ** 2  # ‖U_s^T U_t‖_F²
                    d_grass_sq[j] = k - gram_norm_sq  # range [0, k]

            # Normalize to [0, 1]
            d_norm = d_grass_sq / k  # detached

            # Temperature-controlled softmax — gradient flows through τ
            tau = self.temperatures[i]
            weights = F.softmax(-d_norm / tau, dim=0)  # (L,)

            # Mix RAW teacher tokens (D_t-dimensional output)
            weights_mix = weights.to(input_dtype)
            mixed = (weights_mix.view(-1, 1, 1, 1) * stacked_teacher).sum(dim=0)
            mixed_teachers[s_layer] = mixed

            # Orthogonality penalty — gradient through weights → τ
            orth_i = torch.tensor(0.0, device=device)
            for a in range(L):
                for b in range(a + 1, L):
                    orth_i = orth_i + weights[a] * weights[b] * cross_gram[a, b]
            total_orth = total_orth + orth_i

            # Reconstruction loss — gradient to phi_s AND phi_t
            # Mean-pool across tokens to handle N_s ≠ N_t, then compute MSE / d
            # Weights detached so phi_t gets clean gradient (not through tau)
            z_t_mean = torch.stack([
                z_teachers[idx].mean(dim=0) for idx in teacher_indices
            ])  # (L, d)
            z_s_mean = z_s.mean(dim=0)  # (d,)
            z_mixed_mean = (weights.detach().unsqueeze(1) * z_t_mean).sum(dim=0)  # (d,)
            recon_i = (z_mixed_mean - z_s_mean).pow(2).sum() / self.proj_dim
            total_recon = total_recon + recon_i

            # Logging
            for j, t_idx in enumerate(teacher_indices):
                info[f"selector_s{s_layer}_t{t_idx}_weight"] = weights[j].item()
                info[f"selector_s{s_layer}_t{t_idx}_grass_dist"] = d_norm[j].item()
            info[f"selector_s{s_layer}_temperature"] = tau.item()
            info[f"selector_s{s_layer}_orth_penalty"] = orth_i.item()
            info[f"selector_s{s_layer}_recon_loss"] = recon_i.item()

        n_points = len(extraction_indices)
        diversity_loss = self.diversity_weight * total_orth / n_points
        recon_loss = self.recon_weight * total_recon / n_points
        reg_loss = diversity_loss + recon_loss

        info["diversity_loss"] = diversity_loss.item()
        info["recon_loss"] = recon_loss.item()

        return mixed_teachers, info, reg_loss
