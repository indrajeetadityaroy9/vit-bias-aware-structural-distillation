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
    z = z_flat.float()
    z = z - z.mean(dim=0, keepdim=True)
    M, d = z.shape
    cov = z.T @ z / M
    cov = cov + cov_eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    _, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, -k:]


class GrassmannianLayerSelector(nn.Module):
    def __init__(
        self,
        num_extraction_points: int,
        student_dim: int,
        teacher_dim: int,
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
        self.diversity_weight = diversity_weight
        self.recon_weight = recon_weight
        self.proj_dim = proj_dim
        self.subspace_rank = subspace_rank
        self.cov_eps = cov_eps

        self.phi_s = nn.Linear(student_dim, proj_dim, bias=False)
        self.phi_t = nn.Linear(teacher_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.phi_s.weight)
        nn.init.orthogonal_(self.phi_t.weight)

        self.log_temperatures = nn.Parameter(
            torch.full(
                (num_extraction_points,),
                math.log(math.exp(init_temperature) - 1),
            )
        )

    @property
    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.log_temperatures)

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float], torch.Tensor]:
        teacher_indices = sorted(all_teacher_tokens.keys())
        L = len(teacher_indices)
        _, _, D_t = all_teacher_tokens[teacher_indices[0]].shape

        stacked_teacher = torch.stack(
            [all_teacher_tokens[idx] for idx in teacher_indices]
        )
        stacked_attns = torch.stack(
            [all_teacher_attns[idx] for idx in teacher_indices]
        )

        z_teachers: dict[int, torch.Tensor] = {}
        for idx in teacher_indices:
            t_flat = all_teacher_tokens[idx].reshape(-1, D_t)
            z_teachers[idx] = self.phi_t(t_flat)

        # Eigenspace extraction is detached; gradients flow through reconstruction terms.
        teacher_subspaces: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for idx in teacher_indices:
                teacher_subspaces[idx] = _grassmann_subspace(
                    z_teachers[idx].detach(), self.subspace_rank, self.cov_eps
                )

        cross_gram = torch.zeros(L, L, device=stacked_teacher.device)
        with torch.no_grad():
            for a in range(L):
                U_a = teacher_subspaces[teacher_indices[a]]
                for b in range(a + 1, L):
                    U_b = teacher_subspaces[teacher_indices[b]]
                    gram_val = (U_a.T @ U_b).norm() ** 2
                    cross_gram[a, b] = gram_val
                    cross_gram[b, a] = gram_val

        mixed_teachers: dict[int, torch.Tensor] = {}
        mixed_attentions: dict[int, torch.Tensor] = {}
        info: dict[str, float] = {}
        device = stacked_teacher.device
        input_dtype = stacked_teacher.dtype

        total_orth = torch.tensor(0.0, device=device)
        total_recon = torch.tensor(0.0, device=device)

        k = self.subspace_rank

        for i, s_layer in enumerate(extraction_indices):
            s_tokens = student_tokens_per_layer[s_layer]
            D_s = s_tokens.shape[2]

            s_flat = s_tokens.detach().reshape(-1, D_s)
            z_s = self.phi_s(s_flat)

            with torch.no_grad():
                U_s = _grassmann_subspace(z_s.detach(), k, self.cov_eps)

            d_grass_sq = torch.zeros(L, device=device)
            with torch.no_grad():
                for j, t_idx in enumerate(teacher_indices):
                    U_t = teacher_subspaces[t_idx]
                    gram_norm_sq = (U_s.T @ U_t).norm() ** 2
                    d_grass_sq[j] = k - gram_norm_sq

            d_norm = d_grass_sq / k

            tau = self.temperatures[i]
            weights = F.softmax(-d_norm / tau, dim=0)

            weights_mix = weights.to(input_dtype)
            mixed = (weights_mix.view(-1, 1, 1, 1) * stacked_teacher).sum(dim=0)
            mixed_teachers[s_layer] = mixed
            mixed_attn = (weights_mix.view(-1, 1, 1, 1, 1) * stacked_attns).sum(dim=0)
            mixed_attentions[s_layer] = mixed_attn

            orth_i = torch.tensor(0.0, device=device)
            for a in range(L):
                for b in range(a + 1, L):
                    orth_i = orth_i + weights[a] * weights[b] * cross_gram[a, b]
            total_orth = total_orth + orth_i

            # Detaching weights keeps phi_t gradients independent from temperature dynamics.
            z_t_mean = torch.stack([
                z_teachers[idx].mean(dim=0) for idx in teacher_indices
            ])
            z_s_mean = z_s.mean(dim=0)
            z_mixed_mean = (weights.detach().unsqueeze(1) * z_t_mean).sum(dim=0)
            recon_i = (z_mixed_mean - z_s_mean).pow(2).sum() / self.proj_dim
            total_recon = total_recon + recon_i

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

        return mixed_teachers, mixed_attentions, info, reg_loss
