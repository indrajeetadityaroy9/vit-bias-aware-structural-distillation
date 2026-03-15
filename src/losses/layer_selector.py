import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def marchenko_pastur_rank(features: torch.Tensor) -> int:
    M, D = features.shape
    q = D / M
    if M >= D:
        cov = features.T @ features / M
    else:
        cov = features @ features.T / M
    eigvals = torch.linalg.eigvalsh(cov)
    sigma2 = eigvals.median().item()
    lambda_plus = sigma2 * (1 + q ** 0.5) ** 2
    rank = (eigvals > lambda_plus).sum().item()
    return int(rank)


def _grassmann_subspace(
    z_flat: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k PCA subspace and corresponding singular values.

    Returns:
        basis: (D, k) orthonormal basis of the top-k subspace
        svals: (k,) singular values (spectral importance weights)
    """
    z = z_flat.float()
    z = z - z.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(z, full_matrices=False)
    return Vt[:k].T, S[:k]


class GrassmannianLayerSelector(nn.Module):
    def __init__(
        self,
        num_extraction_points: int,
        student_dim: int,
        teacher_dim: int,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.subspace_ranks: dict[int, int] = {}

        proj_s = torch.empty(student_dim, student_dim)
        proj_t = torch.empty(student_dim, teacher_dim)
        nn.init.orthogonal_(proj_s)
        nn.init.orthogonal_(proj_t)
        self.register_buffer("proj_s", proj_s)
        self.register_buffer("proj_t", proj_t)

        self.log_temperatures = nn.Parameter(
            torch.full(
                (num_extraction_points,),
                math.log(math.exp(1.0) - 1),
            )
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.log_temperatures)

    @torch.no_grad()
    def _estimate_ranks(self, all_teacher_tokens: dict[int, torch.Tensor]) -> None:
        for idx, tokens in all_teacher_tokens.items():
            sample_z = tokens.reshape(-1, tokens.shape[2]) @ self.proj_t.T
            auto_rank = marchenko_pastur_rank(sample_z)
            self.subspace_ranks[idx] = min(auto_rank, self.student_dim - 1)

    def _mix_for_student_layer(
        self,
        i: int,
        s_tokens: torch.Tensor,
        teacher_indices: list[int],
        stacked_tokens: torch.Tensor,
        stacked_attns: torch.Tensor,
        subspaces: dict[int, torch.Tensor],
        spectral_weights: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        D_s = s_tokens.shape[2]
        s_flat = s_tokens.reshape(-1, D_s)
        z_s = s_flat @ self.proj_s.T

        z_s_c = z_s.float()
        z_s_c = z_s_c - z_s_c.mean(dim=0, keepdim=True)
        _, _, Vt_s = torch.linalg.svd(z_s_c, full_matrices=False)

        d_grass_sq = torch.zeros(len(teacher_indices), device=stacked_tokens.device)
        for j, t_idx in enumerate(teacher_indices):
            k = self.subspace_ranks[t_idx]
            U_s = Vt_s[:k].T
            U_t = subspaces[t_idx]
            sigma = torch.linalg.svdvals(U_s.T @ U_t)
            theta = torch.acos(sigma.clamp(max=1.0 - torch.finfo(sigma.dtype).eps))

            # Spectrally-weighted Grassmannian distance: high-energy
            # directions contribute more to the layer matching decision.
            sw = spectral_weights[t_idx]
            d_grass_sq[j] = (sw * theta.pow(2)).sum() / sw.sum()

        tau = self.temperatures[i]
        weights = F.softmax(-d_grass_sq / tau, dim=0)

        weights = weights.to(stacked_tokens.dtype)
        mixed = (weights.view(-1, 1, 1, 1) * stacked_tokens).sum(dim=0)
        mixed_attn = (weights.view(-1, 1, 1, 1, 1) * stacked_attns).sum(dim=0)

        return mixed, mixed_attn

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        teacher_indices = sorted(all_teacher_tokens.keys())

        self._estimate_ranks(all_teacher_tokens)

        D_t = all_teacher_tokens[teacher_indices[0]].shape[2]
        stacked_tokens = torch.stack([all_teacher_tokens[idx] for idx in teacher_indices])
        stacked_attns = torch.stack([all_teacher_attns[idx] for idx in teacher_indices])

        subspaces = {}
        spectral_weights = {}
        with torch.no_grad():
            for idx in teacher_indices:
                z_t = all_teacher_tokens[idx].reshape(-1, D_t) @ self.proj_t.T
                basis, svals = _grassmann_subspace(z_t, k=self.subspace_ranks[idx])
                subspaces[idx] = basis
                spectral_weights[idx] = svals

        mixed_teachers = {}
        mixed_attentions = {}

        for i, s_layer in enumerate(extraction_indices):
            mixed, attn = self._mix_for_student_layer(
                i, student_tokens_per_layer[s_layer],
                teacher_indices, stacked_tokens, stacked_attns,
                subspaces, spectral_weights,
            )
            mixed_teachers[s_layer] = mixed
            mixed_attentions[s_layer] = attn

        return mixed_teachers, mixed_attentions
