"""BASD training engine."""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, SWALR

from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

from vit_inductive_bias_distillation.config import BASDExperimentConfig
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.losses.combined import BASDLoss
from vit_inductive_bias_distillation.models.projector import CrossAttentionProjector
from vit_inductive_bias_distillation.models.teacher import extract_intermediates

__all__ = ["BASDTrainer", "init_distributed", "seed_everything"]


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize DDP from torchrun environment variables."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )
    return rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    """Set all random seeds deterministically."""
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class BASDTrainer:
    """Trainer for BASD distillation."""

    def __init__(
        self,
        student_model: DDP,
        teacher_model: nn.Module,
        teacher_embed_dim: int,
        config: BASDExperimentConfig,
        device: torch.device,
        rank: int,
        world_size: int,
        total_steps: int,
    ):
        self.ddp_model = student_model
        self.teacher_model = teacher_model
        self.teacher_embed_dim = teacher_embed_dim
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        self.token_layers = list(config.basd.token_layers)
        self.student_embed_dim = self.ddp_model.module.embed_dim
        self.student_num_tokens = self.ddp_model.module.patch_embed.num_patches

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.optimizer = optim.AdamW(
            self.ddp_model.module.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fused=True,
        )

        self.warmup_epochs = max(1, int(config.training.warmup_fraction * config.training.num_epochs))
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0 / self.warmup_epochs,
                    total_iters=self.warmup_epochs,
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.training.num_epochs,
                    eta_min=config.training.cosine_eta_min,
                ),
            ],
            milestones=[self.warmup_epochs],
        )

        self.swa_model = AveragedModel(self.ddp_model.module)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.training.swa_lr)
        self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = config.training.early_stopping_min_delta
        self._early_stop_counter = 0
        self._early_stop_best_score = -float("inf")

        self._autocast_dtype = getattr(torch, config.training.autocast_dtype)
        self._grad_clip_norm = config.training.grad_clip_norm
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.metrics_history: dict[str, list[Any]] = defaultdict(list)

        self.cross_attn_projectors = nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=self.student_num_tokens,
                teacher_dim=self.teacher_embed_dim,
                student_dim=self.teacher_embed_dim,
                num_heads=config.basd.cross_attn_num_heads,
            )
            for _ in self.token_layers
        ]).to(device)

        self.basd_loss = BASDLoss(
            base_criterion=self.criterion,
            student_dim=self.student_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            config=config.basd,
            total_steps=total_steps,
            disable_components=config.basd.disable_components,
            student_num_heads=config.vit.num_heads,
            teacher_num_heads=config.basd.cross_attn_num_heads,
        ).to(device)

        self.optimizer.add_param_group({
            "params": list(self.cross_attn_projectors.parameters()) + list(self.basd_loss.parameters()),
            "lr": config.training.learning_rate,
        })

        self.attn_layer_pairs = [tuple(p) for p in config.basd.attn_layer_pairs]
        self.teacher_attn_layers = sorted({p[1] for p in self.attn_layer_pairs})
        self.all_teacher_token_layers = list(range(config.basd.num_teacher_layers))

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=0.8, num_classes=config.model.num_classes),
            CutMix(alpha=1.0, num_classes=config.model.num_classes),
        ])

        if self.is_main_process:
            print(
                config.basd.teacher_model_name,
                self.teacher_embed_dim,
                self.student_embed_dim,
                self.token_layers,
            )

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, Any]) -> None:
        """Save a training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.ddp_model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "swa_model_state_dict": self.swa_model.state_dict(),
            "cross_attn_projectors_state_dict": self.cross_attn_projectors.state_dict(),
            "basd_loss_state_dict": self.basd_loss.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_val_acc": self.best_val_acc,
            "metrics_history": dict(self.metrics_history),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

        torch.save(checkpoint, checkpoint_dir / filename)
        print(filename, epoch + 1)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.ddp_model.module.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.swa_model.load_state_dict(ckpt["swa_model_state_dict"])
        self.cross_attn_projectors.load_state_dict(ckpt["cross_attn_projectors_state_dict"])
        self.basd_loss.load_state_dict(ckpt["basd_loss_state_dict"])
        self.best_val_acc = ckpt["best_val_acc"]
        self.metrics_history = defaultdict(list, ckpt["metrics_history"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_state"]["torch_cuda"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        random.setstate(ckpt["rng_state"]["python"])
        return ckpt["epoch"] + 1

    def step_scheduler(self, epoch: int) -> None:
        """Step the canonical LR schedule."""
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(self.ddp_model.module)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()

    def should_early_stop(self, val_loss: float) -> bool:
        """Early stop on non-improving validation loss."""
        score = -val_loss
        if score < self._early_stop_best_score + self.early_stopping_min_delta:
            self._early_stop_counter += 1
        else:
            self._early_stop_best_score = score
            self._early_stop_counter = 0
        return self._early_stop_counter >= self.early_stopping_patience

    def train_epoch_ddp(
        self,
        train_loader: torch.utils.data.DataLoader,
        train_sampler: torch.utils.data.distributed.DistributedSampler,
    ) -> dict[str, Any]:
        """Train one epoch."""
        train_sampler.set_epoch(self.current_epoch)
        self.ddp_model.train()
        self.teacher_model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_rsd_loss = 0.0
        total_rel_loss = 0.0
        total_attn_loss = 0.0
        total_spectral_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for clean_imgs, student_imgs, targets in train_loader:
            clean_imgs = clean_imgs.to(self.device, non_blocking=True)
            student_imgs = student_imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Mixup/CutMix on student images only (teacher uses clean images)
            student_imgs, mixed_targets = self.mixup_cutmix(student_imgs, targets)

            with autocast(device_type="cuda", dtype=self._autocast_dtype):
                student_results = self.ddp_model(
                    student_imgs,
                    layer_indices=self.token_layers,
                )
                student_output = student_results.output
                student_intermediates = student_results.intermediates
                student_attns = student_results.attention_weights

                teacher_results = extract_intermediates(
                    self.teacher_model,
                    clean_imgs,
                    self.token_layers,
                    self.teacher_attn_layers,
                    self.cross_attn_projectors,
                    all_token_layers=self.all_teacher_token_layers,
                )
                loss, loss_dict = self.basd_loss(
                    student_output,
                    mixed_targets,
                    student_intermediates,
                    teacher_results.projected,
                    teacher_results.raw,
                    teacher_results.all_raw,
                    student_attns,
                    teacher_results.attentions,
                    self.cross_attn_projectors,
                    self.global_step,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.ddp_model.parameters())
                + list(self.basd_loss.parameters())
                + list(self.cross_attn_projectors.parameters()),
                self._grad_clip_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_ce_loss += loss_dict["ce_loss"]
            total_rsd_loss += loss_dict["rsd_loss"]
            total_rel_loss += loss_dict["rel_loss"]
            total_attn_loss += loss_dict["attn_loss"]
            total_spectral_loss += loss_dict["spectral_loss"]

            predicted = student_output.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

            self.global_step += 1

        metrics_tensor = torch.tensor([
            total_loss,
            total_ce_loss,
            total_rsd_loss,
            total_rel_loss,
            total_attn_loss,
            total_spectral_loss,
            correct,
            total,
            batch_count,
        ], device=self.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        batch_count_total = int(metrics_tensor[8].item())
        result = {
            "train_loss": metrics_tensor[0].item() / batch_count_total,
            "train_ce_loss": metrics_tensor[1].item() / batch_count_total,
            "train_rsd_loss": metrics_tensor[2].item() / batch_count_total,
            "train_rel_loss": metrics_tensor[3].item() / batch_count_total,
            "train_attn_loss": metrics_tensor[4].item() / batch_count_total,
            "train_spectral_loss": metrics_tensor[5].item() / batch_count_total,
            "train_acc": 100.0 * metrics_tensor[6].item() / metrics_tensor[7].item(),
            "total_samples": int(metrics_tensor[7].item()),
        }
        for key in loss_dict:
            if key.startswith("weight_"):
                result[key] = loss_dict[key]
        return result

    def train_ddp(
        self,
        train_loader: torch.utils.data.DataLoader,
        train_sampler: torch.utils.data.distributed.DistributedSampler,
        val_loader: torch.utils.data.DataLoader,
        start_epoch: int = 0,
    ) -> dict[str, list[Any]]:
        """Full BASD training loop."""
        num_epochs = self.config.training.num_epochs

        if self.is_main_process:
            print(num_epochs, self.world_size, self.config.data.batch_size * self.world_size)

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)

            val_metrics = evaluate_model(
                self.ddp_model, val_loader, self.device,
                criterion=self.criterion,
                rank=self.rank, world_size=self.world_size,
            )

            self.step_scheduler(epoch)

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            dist.barrier()

            if self.is_main_process:
                throughput = train_metrics["total_samples"] / epoch_time
                print(
                    epoch + 1,
                    num_epochs,
                    train_metrics["train_loss"],
                    train_metrics["train_ce_loss"],
                    train_metrics["train_rsd_loss"],
                    train_metrics["train_rel_loss"],
                    train_metrics["train_attn_loss"],
                    train_metrics["train_spectral_loss"],
                    train_metrics["train_acc"],
                    val_metrics["val_acc"],
                    current_lr,
                    epoch_time,
                    throughput,
                )

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                if val_metrics["val_acc"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["val_acc"]
                    self.save_checkpoint("best_model.pth", epoch, val_metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", epoch, val_metrics)

                if self.should_early_stop(val_metrics["loss"]):
                    print(epoch + 1, val_metrics["loss"])
                    break

        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)
        if self.is_main_process:
            print(self.best_val_acc)

        return dict(self.metrics_history)
