from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from vit_inductive_bias_distillation.losses.attention import AttentionDistillationLoss
from vit_inductive_bias_distillation.losses.layer_selector import (
    GrassmannianLayerSelector,
    _retract_to_stiefel,
)
from vit_inductive_bias_distillation.losses.relational import GeometricRelationalLoss
from vit_inductive_bias_distillation.losses.rsd import RedundancySuppressionLoss
from vit_inductive_bias_distillation.losses.spectral import BuresWassersteinLoss


class ScaleInvariantWeighting:
    """Zero-hyperparameter multi-task weighting via z-score normalized UW-SO."""

    def __init__(self, component_names: list[str]):
        self.component_names = list(component_names)

    def forward(
        self,
        losses: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        active = [(name, loss) for name, loss in losses.items() if loss.detach().abs() > 1e-8]
        names, vals = zip(*active)
        stacked_detached = torch.stack([v.detach() for v in vals])

        n_active = len(active)
        if n_active == 1:
            z = torch.zeros(1, device=stacked_detached.device)
        else:
            z = (stacked_detached - stacked_detached.mean()) / (stacked_detached.std() + 1e-8)

        w = F.softmax(-z, dim=0) * n_active

        total = sum(w[i] * vals[i] for i in range(n_active))

        info: dict[str, float] = {}
        active_set = set(names)
        for j, name in enumerate(names):
            info[f"weight_{name}"] = w[j].item()
        for name in self.component_names:
            if name not in active_set:
                info[f"weight_{name}"] = 0.0

        return total, info


class GradientGatedSchedule:
    """Activate loss components when their gradient norms stabilize."""

    def __init__(self, component_names: list[str], *, window: int = 100):
        self.component_names = list(component_names)
        self.window = window
        self._grad_norms: dict[str, list[float]] = {n: [] for n in component_names}
        self._activated: dict[str, bool] = {n: False for n in component_names}
        self._activated[component_names[0]] = True

    def get_ramp(self, component: str, step: int) -> float:
        if self._activated[component]:
            return 1.0
        buf = self._grad_norms[component]
        if len(buf) < self.window // 2:
            return 0.0
        t = torch.tensor(buf, dtype=torch.float32)
        mean = t.mean().item()
        if mean < 1e-12:
            return 0.0
        cv = t.std().item() / mean
        if cv < 1.0:
            self._activated[component] = True
            return 1.0
        return 0.0

    def update(self, component: str, grad_norm: float) -> None:
        buf = self._grad_norms[component]
        buf.append(grad_norm)
        if len(buf) > self.window:
            buf.pop(0)

    def state_dict(self) -> dict:
        return {"activated": dict(self._activated)}

    def load_state_dict(self, state: dict) -> None:
        self._activated.update(state["activated"])


class LagrangianRegularizer:
    """Adaptive regularization via dual ascent for layer selector reg weights."""

    def __init__(self):
        self._mu_div: float = 0.01
        self._mu_rec: float = 0.01
        self._lr_dual: float = 0.001

    def compute_reg(
        self, diversity_loss: torch.Tensor, recon_loss: torch.Tensor
    ) -> torch.Tensor:
        return self._mu_div * diversity_loss + self._mu_rec * recon_loss

    def update(self, diversity_loss: float, recon_loss: float) -> None:
        self._mu_div = max(0.0, self._mu_div + self._lr_dual * (diversity_loss - 0.1))
        self._mu_rec = max(0.0, self._mu_rec + self._lr_dual * (recon_loss - 0.1))

    def state_dict(self) -> dict:
        return {"mu_div": self._mu_div, "mu_rec": self._mu_rec}

    def load_state_dict(self, state: dict) -> None:
        self._mu_div = state["mu_div"]
        self._mu_rec = state["mu_rec"]


class CrossAttentionProjector(nn.Module):
    def __init__(self, num_student_tokens: int, teacher_dim: int, student_dim: int, *, num_heads: int = 4):
        super().__init__()
        self.num_student_tokens = num_student_tokens

        self.teacher_proj = nn.Linear(teacher_dim, student_dim)
        self.queries = nn.Parameter(torch.randn(1, num_student_tokens, student_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=student_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(student_dim)

    def forward(self, teacher_tokens: torch.Tensor) -> torch.Tensor:
        B = teacher_tokens.shape[0]

        kv = self.teacher_proj(teacher_tokens)
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, kv, kv)
        aligned_tokens = self.norm(queries + attn_out)

        return aligned_tokens

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        _retract_to_stiefel(self.teacher_proj)


class BASDLoss(nn.Module):
    def __init__(
        self,
        base_criterion: nn.Module,
        student_dim: int,
        teacher_dim: int,
        student_depth: int,
        num_student_tokens: int,
        cross_attn_num_heads: int,
        *,
        config,
        student_num_heads: int = 3,
        teacher_num_heads: int = 6,
    ):
        super().__init__()
        self.base_criterion = base_criterion

        num_points = config.num_extraction_points
        if num_points >= student_depth:
            self.token_layers = list(range(student_depth))
        else:
            self.token_layers = [round(i * (student_depth - 1) / (num_points - 1)) for i in range(num_points)]

        all_components = ["rsd", "rel", "attn", "spectral"]

        self.disabled_components = frozenset(config.disabled_components)
        self.use_scale_invariant_weighting = config.use_scale_invariant_weighting
        self.use_gradient_gating = config.use_gradient_gating
        self.use_lagrangian_reg = config.use_lagrangian_reg

        if "rel" in self.disabled_components:
            logger.warning(
                "Relational loss disabled — CrossAttentionProjector.teacher_proj "
                "receives no gradients; Stiefel retraction is meaningless"
            )

        self.cross_attn_projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=num_student_tokens,
                teacher_dim=teacher_dim,
                student_dim=teacher_dim,
                num_heads=cross_attn_num_heads,
            )
            for _ in self.token_layers
        ])

        self.rsd_loss = RedundancySuppressionLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            num_layers=len(self.token_layers),
            kappa=1.0 / teacher_dim,
        )
        self.rel_loss = GeometricRelationalLoss(
            attn_weighted=config.rel_attn_weighted,
        )
        self.attn_loss = AttentionDistillationLoss(
            student_heads_per_layer=student_num_heads,
            teacher_heads_per_layer=teacher_num_heads,
            num_layers=len(self.token_layers),
        )
        self.spectral_loss = BuresWassersteinLoss()

        self.layer_selector = GrassmannianLayerSelector(
            num_extraction_points=len(self.token_layers),
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            init_temperature=config.layer_selector_temperature,
            cov_eps=config.layer_selector_grass_cov_eps,
            fixed_rank=config.layer_selector_fixed_rank,
        )

        self.uncertainty = ScaleInvariantWeighting(all_components)
        self.schedule = GradientGatedSchedule(
            all_components, window=config.grad_gate_window,
        )
        self.lagrangian_reg = LagrangianRegularizer()

    @torch.no_grad()
    def project_to_stiefel(self) -> None:
        self.layer_selector.project_to_stiefel()
        for proj in self.cross_attn_projectors:
            proj.project_to_stiefel()

    def post_step_update(self, loss_info: dict[str, float]) -> None:
        if self.use_gradient_gating:
            for comp in ["rsd", "rel", "attn", "spectral"]:
                key = f"{comp}_loss"
                if key in loss_info:
                    self.schedule.update(comp, loss_info[key])
        if self.use_lagrangian_reg and "raw_diversity" in loss_info and "raw_recon" in loss_info:
            self.lagrangian_reg.update(loss_info["raw_diversity"], loss_info["raw_recon"])

    def _avg_per_layer(
        self,
        loss_fn: Callable[..., torch.Tensor],
        name: str,
        student_intermediates: dict[int, torch.Tensor],
        aligned_tokens: dict[int, torch.Tensor],
        extras: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = student_intermediates[self.token_layers[0]].device
        total = torch.tensor(0.0, device=device)
        info: dict[str, float] = {}
        for layer_idx in self.token_layers:
            args = [student_intermediates[layer_idx], aligned_tokens[layer_idx]]
            if extras is not None and layer_idx in extras:
                args.append(extras[layer_idx])
            layer_loss = loss_fn(*args)
            total = total + layer_loss
            info[f"{name}_layer_{layer_idx}"] = layer_loss.item()
        return total / len(self.token_layers), info

    def _record_component(
        self,
        name: str,
        loss: torch.Tensor,
        extra_info: dict[str, float],
        raw_losses: dict[str, torch.Tensor],
        loss_info: dict[str, float],
        global_step: int,
    ) -> None:
        loss_info[f"{name}_loss"] = loss.item()
        if extra_info:
            loss_info.update(extra_info)
        if name not in self.disabled_components:
            ramp = self.schedule.get_ramp(name, global_step) if self.use_gradient_gating else 1.0
            raw_losses[name] = ramp * loss

    def forward(
        self,
        student_output: torch.Tensor,
        targets: torch.Tensor,
        student_intermediates: dict[int, torch.Tensor],
        student_attns: dict[int, torch.Tensor],
        all_teacher_tokens: dict[int, torch.Tensor],
        all_teacher_attns: dict[int, torch.Tensor],
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce_loss = self.base_criterion(student_output, targets)

        mixed_tokens, mixed_attns, selector_info, diversity_loss, recon_loss = self.layer_selector(
            student_intermediates, all_teacher_tokens, all_teacher_attns,
            self.token_layers,
        )

        aligned_tokens: dict[int, torch.Tensor] = {}
        for i, layer_idx in enumerate(self.token_layers):
            aligned_tokens[layer_idx] = self.cross_attn_projectors[i](
                mixed_tokens[layer_idx]
            )

        raw_losses: dict[str, torch.Tensor] = {}
        loss_info: dict[str, float] = {"ce_loss": ce_loss.item()}
        loss_info.update(selector_info)

        rsd_loss, rsd_dict = self.rsd_loss(
            student_intermediates, aligned_tokens, self.token_layers
        )
        self._record_component("rsd", rsd_loss, rsd_dict, raw_losses, loss_info, global_step)

        rel_loss, rel_dict = self._avg_per_layer(
            self.rel_loss, "rel", student_intermediates, aligned_tokens, extras=mixed_attns,
        )
        self._record_component("rel", rel_loss, rel_dict, raw_losses, loss_info, global_step)

        attn_loss = self.attn_loss(student_attns, mixed_attns, self.token_layers)
        self._record_component("attn", attn_loss, {}, raw_losses, loss_info, global_step)

        spectral_loss, spectral_dict = self._avg_per_layer(
            self.spectral_loss, "spectral", student_intermediates, aligned_tokens,
        )
        self._record_component("spectral", spectral_loss, spectral_dict, raw_losses, loss_info, global_step)

        if raw_losses:
            if self.use_scale_invariant_weighting:
                weighted_sum, weight_info = self.uncertainty.forward(raw_losses)
            else:
                weighted_sum = sum(raw_losses.values())
                weight_info = {f"weight_{name}": 1.0 for name in raw_losses}
        else:
            weighted_sum = torch.tensor(0.0, device=ce_loss.device)
            weight_info = {}

        if self.use_lagrangian_reg:
            reg_loss = self.lagrangian_reg.compute_reg(diversity_loss, recon_loss)
        else:
            reg_loss = torch.tensor(0.0, device=ce_loss.device)

        total_loss = ce_loss + weighted_sum + reg_loss
        loss_info.update(weight_info)
        loss_info["selector_reg_loss"] = reg_loss.item()
        loss_info["lagrangian_mu_div"] = self.lagrangian_reg._mu_div
        loss_info["lagrangian_mu_rec"] = self.lagrangian_reg._mu_rec
        loss_info["total_loss"] = total_loss.item()

        return total_loss, loss_info
