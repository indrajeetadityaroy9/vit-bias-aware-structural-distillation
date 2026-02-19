from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, SWALR

from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

from vit_inductive_bias_distillation.config import Config
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.losses.combined import BASDLoss
from vit_inductive_bias_distillation.models.teacher import TeacherModel, extract_intermediates
from vit_inductive_bias_distillation.runtime_log import log_event

__all__ = ["BASDTrainer", "seed_everything"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class BASDTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        teacher: TeacherModel,
        config: Config,
        device: torch.device,
        steps_per_epoch: int,
    ):
        self.model = torch.compile(student_model, mode="max-autotune")
        self.teacher_model = teacher.model
        self.config = config
        self.device = device
        total_steps = steps_per_epoch * config.training.num_epochs

        self.token_layers = list(config.basd.token_layers)
        self._teacher_num_layers = teacher.num_layers
        self._teacher_loader = teacher.loader
        self._teacher_feature_format = teacher.feature_format
        student_embed_dim = self.model.embed_dim
        student_num_tokens = self.model.patch_embed.num_patches
        cross_attn_num_heads = max(1, teacher.embed_dim // 64)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
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

        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.training.swa_lr)
        self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = config.training.early_stopping_min_delta
        self._early_stop_counter = 0
        self._early_stop_best_score = -float("inf")

        _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        self._autocast_dtype = _DTYPE_MAP[config.training.autocast_dtype]
        self._grad_clip_norm = config.training.grad_clip_norm
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.metrics_history: dict[str, list[Any]] = defaultdict(list)

        self.basd_loss = BASDLoss(
            base_criterion=self.criterion,
            student_dim=student_embed_dim,
            teacher_dim=teacher.embed_dim,
            num_student_tokens=student_num_tokens,
            cross_attn_num_heads=cross_attn_num_heads,
            config=config.basd,
            total_steps=total_steps,
            disable_components=config.basd.disable_components,
            student_num_heads=config.model.vit.num_heads,
            teacher_num_heads=teacher.num_heads,
        ).to(device)

        self.optimizer.add_param_group({
            "params": list(self.basd_loss.parameters()),
            "lr": config.training.learning_rate,
        })

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=0.8, num_classes=config.model.num_classes),
            CutMix(alpha=1.0, num_classes=config.model.num_classes),
        ])

        log_event(
            "trainer_init",
            teacher_model=config.basd.teacher_model_name,
            teacher_dim=teacher.embed_dim,
            student_dim=student_embed_dim,
            token_layers=self.token_layers,
        )

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, Any]) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "swa_model_state_dict": self.swa_model.state_dict(),
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

        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        log_event("checkpoint_saved", path=str(checkpoint_path), epoch=epoch + 1)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.swa_model.load_state_dict(ckpt["swa_model_state_dict"])
        self.basd_loss.load_state_dict(ckpt["basd_loss_state_dict"])
        self.best_val_acc = ckpt["best_val_acc"]
        self.metrics_history = defaultdict(list, ckpt["metrics_history"])
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_state"]["torch_cuda"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        random.setstate(ckpt["rng_state"]["python"])
        return ckpt["epoch"] + 1

    def step_scheduler(self, epoch: int) -> None:
        if epoch >= self.swa_start_epoch:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()

    def should_early_stop(self, val_loss: float) -> bool:
        score = -val_loss
        if score < self._early_stop_best_score + self.early_stopping_min_delta:
            self._early_stop_counter += 1
        else:
            self._early_stop_best_score = score
            self._early_stop_counter = 0
        return self._early_stop_counter >= self.early_stopping_patience

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        self.model.train()
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

            student_imgs, mixed_targets = self.mixup_cutmix(student_imgs, targets)

            with autocast(device_type="cuda", dtype=self._autocast_dtype):
                student_results = self.model(
                    student_imgs,
                    layer_indices=self.token_layers,
                )

                teacher_results = extract_intermediates(
                    self.teacher_model, clean_imgs, self._teacher_num_layers,
                    loader=self._teacher_loader,
                    feature_format=self._teacher_feature_format,
                )

                loss, loss_dict = self.basd_loss(
                    student_results.output,
                    mixed_targets,
                    student_results.intermediates,
                    student_results.attention_weights,
                    teacher_results.all_tokens,
                    teacher_results.all_attentions,
                    self.global_step,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.basd_loss.parameters()),
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

            predicted = student_results.output.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

            self.global_step += 1

        result = {
            "train_loss": total_loss / batch_count,
            "train_ce_loss": total_ce_loss / batch_count,
            "train_rsd_loss": total_rsd_loss / batch_count,
            "train_rel_loss": total_rel_loss / batch_count,
            "train_attn_loss": total_attn_loss / batch_count,
            "train_spectral_loss": total_spectral_loss / batch_count,
            "train_acc": 100.0 * correct / total,
            "total_samples": total,
        }
        for key in loss_dict:
            if key.startswith("weight_"):
                result[key] = loss_dict[key]
        return result

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        start_epoch: int = 0,
    ) -> dict[str, list[Any]]:
        num_epochs = self.config.training.num_epochs

        log_event(
            "train_config",
            epochs=num_epochs,
            batch_size=self.config.data.batch_size,
        )

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch(train_loader)

            val_metrics = evaluate_model(
                self.model, val_loader, self.device,
                criterion=self.criterion,
            )

            self.step_scheduler(epoch)

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            throughput = train_metrics["total_samples"] / epoch_time

            log_event(
                "epoch_summary",
                epoch=epoch + 1,
                epochs=num_epochs,
                train_loss=train_metrics["train_loss"],
                train_ce_loss=train_metrics["train_ce_loss"],
                train_rsd_loss=train_metrics["train_rsd_loss"],
                train_rel_loss=train_metrics["train_rel_loss"],
                train_attn_loss=train_metrics["train_attn_loss"],
                train_spectral_loss=train_metrics["train_spectral_loss"],
                train_acc=train_metrics["train_acc"],
                val_acc=val_metrics["val_acc"],
                lr=current_lr,
                epoch_time_sec=epoch_time,
                throughput_img_per_sec=throughput,
            )

            for key, value in {**train_metrics, **val_metrics}.items():
                self.metrics_history[key].append(value)
            self.metrics_history["epoch_time"].append(epoch_time)
            self.metrics_history["throughput"].append(throughput)

            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.save_checkpoint("best_model.pth", epoch, val_metrics)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", epoch, val_metrics)

            if self.should_early_stop(val_metrics["loss"]):
                log_event("early_stop", epoch=epoch + 1, val_loss=val_metrics["loss"])
                break

        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)
        log_event("train_complete", best_val_acc=self.best_val_acc)

        return dict(self.metrics_history)
