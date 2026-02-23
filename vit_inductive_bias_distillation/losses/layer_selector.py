from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def _polar_retract(W: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(W, full_matrices=False)
    return U @ Vt


@torch.no_grad()
def _retract_to_stiefel(linear: nn.Linear) -> None:
    linear.weight.copy_(_polar_retract(linear.weight))


@torch.no_grad()
def _marchenko_pastur_rank(features: torch.Tensor, *, min_rank: int = 4) -> int:
    """Estimate effective rank via Marchenko-Pastur threshold."""
    M, D = features.shape
    q = D / M
    cov = features.T @ features / M
    eigvals = torch.linalg.eigvalsh(cov)
    sigma2 = eigvals.median().item()
    lambda_plus = sigma2 * (1 + q ** 0.5) ** 2
    rank = (eigvals > lambda_plus).sum().item()
    return max(min_rank, int(rank))


def _grassmann_subspace(
    z_flat: torch.Tensor,
    *,
    k: int,
    cov_eps: float,
) -> tuple[torch.Tensor, float]:
    """Compute top-k eigenvectors and spectral concentration ratio."""
    z = z_flat.float()
    z = z - z.mean(dim=0, keepdim=True)
    M, d = z.shape
    cov = z.T @ z / M
    cov = cov + cov_eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    gamma = eigvals[-1].item() / (eigvals.sum().item() + 1e-12)
    return eigvecs[:, -k:], gamma


@dataclass
class _TeacherState:
    """Precomputed teacher-side data shared across all student extraction points."""
    teacher_indices: list[int]
    stacked_tokens: torch.Tensor
    stacked_attns: torch.Tensor
    z_teachers: dict[int, torch.Tensor]
    subspaces: dict[int, torch.Tensor]
    concentrations: dict[int, float]
    cross_gram: torch.Tensor


class GrassmannianLayerSelector(nn.Module):
    def __init__(
        self,
        num_extraction_points: int,
        student_dim: int,
        teacher_dim: int,
        *,
        init_temperature: float = 2.0,
        subspace_rank: int = 16,
        cov_eps: float = 1e-4,
        fixed_rank: int = 0,
    ):
        super().__init__()
        proj_dim = teacher_dim // 3
        self.proj_dim = proj_dim
        self.cov_eps = cov_eps

        if fixed_rank > 0:
            self.subspace_rank = fixed_rank
            self._rank_calibrated = True
        else:
            self.subspace_rank = subspace_rank
            self._rank_calibrated = False

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

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        _retract_to_stiefel(self.phi_s)
        _retract_to_stiefel(self.phi_t)

    @property
    def temperatures(self) -> torch.Tensor:
        return F.softplus(self.log_temperatures)

    def _calibrate_rank(self, sample_tokens: torch.Tensor, D_t: int) -> None:
        """One-time Marchenko-Pastur rank calibration."""
        if self._rank_calibrated:
            return
        with torch.no_grad():
            sample_z = self.phi_t(sample_tokens.reshape(-1, D_t))
            auto_rank = _marchenko_pastur_rank(sample_z)
            self.subspace_rank = min(auto_rank, self.proj_dim - 1)
        self._rank_calibrated = True

    def _compute_teacher_state(
        self,
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        teacher_indices: list[int],
    ) -> _TeacherState:
        """Project teacher tokens, compute subspaces, concentrations, and cross-Gram."""
        D_t = all_teacher_tokens[teacher_indices[0]].shape[2]

        stacked_tokens = torch.stack([all_teacher_tokens[idx] for idx in teacher_indices])
        stacked_attns = torch.stack([all_teacher_attns[idx] for idx in teacher_indices])

        z_teachers: dict[int, torch.Tensor] = {}
        for idx in teacher_indices:
            z_teachers[idx] = self.phi_t(all_teacher_tokens[idx].reshape(-1, D_t))

        # Eigenvectors are used only for geometry scores, not for projection gradients.
        subspaces: dict[int, torch.Tensor] = {}
        concentrations: dict[int, float] = {}
        with torch.no_grad():
            for idx in teacher_indices:
                subspace, gamma = _grassmann_subspace(
                    z_teachers[idx].detach(), k=self.subspace_rank, cov_eps=self.cov_eps
                )
                subspaces[idx] = subspace
                concentrations[idx] = gamma

        L = len(teacher_indices)
        cross_gram = torch.zeros(L, L, device=stacked_tokens.device)
        with torch.no_grad():
            for a in range(L):
                U_a = subspaces[teacher_indices[a]]
                for b in range(a + 1, L):
                    U_b = subspaces[teacher_indices[b]]
                    gram_val = (U_a.T @ U_b).norm() ** 2
                    cross_gram[a, b] = gram_val
                    cross_gram[b, a] = gram_val

        return _TeacherState(
            teacher_indices=teacher_indices,
            stacked_tokens=stacked_tokens,
            stacked_attns=stacked_attns,
            z_teachers=z_teachers,
            subspaces=subspaces,
            concentrations=concentrations,
            cross_gram=cross_gram,
        )

    def _mix_for_student_layer(
        self,
        i: int,
        s_layer: int,
        s_tokens: torch.Tensor,
        state: _TeacherState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Compute mixed teacher tokens/attns and regularization for one student layer."""
        teacher_indices = state.teacher_indices
        L = len(teacher_indices)
        k = self.subspace_rank
        device = state.stacked_tokens.device
        input_dtype = state.stacked_tokens.dtype

        D_s = s_tokens.shape[2]
        s_flat = s_tokens.detach().reshape(-1, D_s)
        z_s = self.phi_s(s_flat)

        with torch.no_grad():
            U_s, _ = _grassmann_subspace(z_s.detach(), k=k, cov_eps=self.cov_eps)

        d_grass_sq = torch.zeros(L, device=device)
        with torch.no_grad():
            for j, t_idx in enumerate(teacher_indices):
                U_t = state.subspaces[t_idx]
                gram_norm_sq = (U_s.T @ U_t).norm() ** 2
                d_grass_sq[j] = k - gram_norm_sq

        d_norm = d_grass_sq / k

        # Penalize layers with unusually concentrated spectra.
        concentrations = torch.tensor(
            [state.concentrations[idx] for idx in teacher_indices], device=device
        )
        conc_mean = concentrations.mean()
        conc_std = concentrations.std() + 1e-8
        z_scores = (concentrations - conc_mean) / conc_std
        penalties = torch.sigmoid(-z_scores)
        d_norm = d_norm + penalties

        tau = self.temperatures[i]
        weights = F.softmax(-d_norm / tau, dim=0)

        weights_mix = weights.to(input_dtype)
        mixed = (weights_mix.view(-1, 1, 1, 1) * state.stacked_tokens).sum(dim=0)
        mixed_attn = (weights_mix.view(-1, 1, 1, 1, 1) * state.stacked_attns).sum(dim=0)

        orth_i = torch.tensor(0.0, device=device)
        for a in range(L):
            for b in range(a + 1, L):
                orth_i = orth_i + weights[a] * weights[b] * state.cross_gram[a, b]

        # Detaching weights decouples phi_t gradients from temperature updates.
        z_t_mean = torch.stack([
            state.z_teachers[idx].mean(dim=0) for idx in teacher_indices
        ])
        z_s_mean = z_s.mean(dim=0)
        z_mixed_mean = (weights.detach().unsqueeze(1) * z_t_mean).sum(dim=0)
        recon_i = (z_mixed_mean - z_s_mean).pow(2).sum() / self.proj_dim

        layer_info: dict[str, float] = {}
        for j, t_idx in enumerate(teacher_indices):
            layer_info[f"selector_s{s_layer}_t{t_idx}_weight"] = weights[j].item()
            layer_info[f"selector_s{s_layer}_t{t_idx}_grass_dist"] = d_norm[j].item()
            layer_info[f"selector_s{s_layer}_t{t_idx}_spectral_conc"] = state.concentrations[t_idx]
        layer_info[f"selector_s{s_layer}_temperature"] = tau.item()
        layer_info[f"selector_s{s_layer}_orth_penalty"] = orth_i.item()
        layer_info[f"selector_s{s_layer}_recon_loss"] = recon_i.item()

        return mixed, mixed_attn, orth_i, recon_i, layer_info

    def forward(
        self,
        student_tokens_per_layer: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        extraction_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, float], torch.Tensor, torch.Tensor]:
        teacher_indices = sorted(all_teacher_tokens.keys())
        D_t = all_teacher_tokens[teacher_indices[0]].shape[2]

        self._calibrate_rank(all_teacher_tokens[teacher_indices[0]], D_t)
        state = self._compute_teacher_state(all_teacher_tokens, all_teacher_attns, teacher_indices)

        mixed_teachers: dict[int, torch.Tensor] = {}
        mixed_attentions: dict[int, torch.Tensor] = {}
        info: dict[str, float] = {}
        device = state.stacked_tokens.device
        total_orth = torch.tensor(0.0, device=device)
        total_recon = torch.tensor(0.0, device=device)

        for i, s_layer in enumerate(extraction_indices):
            mixed, attn, orth, recon, layer_info = self._mix_for_student_layer(
                i, s_layer, student_tokens_per_layer[s_layer], state,
            )
            mixed_teachers[s_layer] = mixed
            mixed_attentions[s_layer] = attn
            total_orth = total_orth + orth
            total_recon = total_recon + recon
            info.update(layer_info)

        n_points = len(extraction_indices)
        avg_orth = total_orth / n_points
        avg_recon = total_recon / n_points
        info["raw_diversity"] = avg_orth.item()
        info["raw_recon"] = avg_recon.item()

        return mixed_teachers, mixed_attentions, info, avg_orth, avg_recon
