from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.optim.swa_utils import AveragedModel, SWALR

from torchvision.transforms import MixUp, CutMix, RandomChoice

from vit_inductive_bias_distillation.config import Config
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.losses.combined import BASDLoss
from vit_inductive_bias_distillation.models.teacher import TeacherModel, extract_intermediates

__all__ = ["BASDTrainer"]


class BASDTrainer:
    def __init__(
        self,
        student_model: nn.Module,
        teacher: TeacherModel,
        config: Config,
        accelerator: Accelerator,
        steps_per_epoch: int,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.config = config
        self.teacher_model = teacher.model
        total_steps = steps_per_epoch * config.training.num_epochs

        self.token_layers = list(config.basd.token_layers)
        self._teacher_num_layers = teacher.num_layers
        self._teacher_loader = teacher.loader
        self._teacher_feature_format = teacher.feature_format
        student_embed_dim = student_model.embed_dim
        student_num_tokens = student_model.patch_embed.num_patches
        cross_attn_num_heads = max(1, teacher.embed_dim // 64)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.optimizer = optim.AdamW(
            student_model.parameters(),
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

        self.basd_loss = BASDLoss(
            base_criterion=self.criterion,
            student_dim=student_embed_dim,
            teacher_dim=teacher.embed_dim,
            num_student_tokens=student_num_tokens,
            cross_attn_num_heads=cross_attn_num_heads,
            config=config.basd,
            total_steps=total_steps,
            student_num_heads=config.model.vit.num_heads,
            teacher_num_heads=teacher.num_heads,
        ).to(self.device)

        self.optimizer.add_param_group({
            "params": list(self.basd_loss.parameters()),
            "lr": config.training.learning_rate,
        })

        self.model, self.optimizer, self.scheduler = accelerator.prepare(
            student_model, self.optimizer, self.scheduler
        )

        accelerator.register_for_checkpointing(self.basd_loss)

        # Compile after Accelerate wrapping so graph capture includes distributed wrappers.
        self.model = torch.compile(self.model, mode="max-autotune")

        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.training.swa_lr)
        self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)
        accelerator.register_for_checkpointing(self.swa_model)

        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = config.training.early_stopping_min_delta
        self._early_stop_counter = 0
        self._early_stop_best_score = -float("inf")

        self._grad_clip_norm = config.training.grad_clip_norm
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.metrics_history: dict[str, list[Any]] = defaultdict(list)

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=0.8, num_classes=config.model.num_classes),
            CutMix(alpha=1.0, num_classes=config.model.num_classes),
        ])

        print(
            f"event=trainer_init teacher={config.basd.teacher_model_name} "
            f"teacher_dim={teacher.embed_dim} student_dim={student_embed_dim} "
            f"token_layers={self.token_layers}"
        )

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints" / name
        self.accelerator.save_state(str(checkpoint_dir))

        custom_state = {
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "metrics_history": dict(self.metrics_history),
        }
        torch.save(custom_state, checkpoint_dir / "custom_state.pth")
        print(f"event=checkpoint_saved path={checkpoint_dir} epoch={epoch + 1}")

    def save_model_for_eval(self, filename: str, epoch: int) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(
            {"epoch": epoch, "model_state_dict": unwrapped.state_dict()},
            checkpoint_dir / filename,
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        self.accelerator.load_state(checkpoint_path)

        custom = torch.load(
            Path(checkpoint_path) / "custom_state.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.best_val_acc = custom["best_val_acc"]
        self.metrics_history = defaultdict(list, custom["metrics_history"])
        return custom["epoch"] + 1

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

            with self.accelerator.autocast():
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

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.basd_loss.parameters()),
                self._grad_clip_norm,
            )
            self.optimizer.step()
            self.basd_loss.layer_selector.project_to_stiefel()
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

        print(f"event=train_config epochs={num_epochs} batch_size={self.config.data.batch_size}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch(train_loader)

            val_metrics = evaluate_model(
                self.model, val_loader, self.device,
                criterion=self.criterion,
                num_classes=self.config.model.num_classes,
            )

            self.step_scheduler(epoch)

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]
            throughput = train_metrics["total_samples"] / epoch_time

            print(
                f"event=epoch_summary epoch={epoch + 1} num_epochs={num_epochs} "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"ce={train_metrics['train_ce_loss']:.4f} "
                f"rsd={train_metrics['train_rsd_loss']:.4f} "
                f"rel={train_metrics['train_rel_loss']:.4f} "
                f"attn={train_metrics['train_attn_loss']:.4f} "
                f"spectral={train_metrics['train_spectral_loss']:.4f} "
                f"train_acc={train_metrics['train_acc']:.2f} "
                f"val_acc={val_metrics['val_acc']:.2f} "
                f"lr={current_lr:.2e} epoch_time_s={epoch_time:.1f} "
                f"throughput_img_per_sec={throughput:.0f}"
            )

            for key, value in {**train_metrics, **val_metrics}.items():
                self.metrics_history[key].append(value)
            self.metrics_history["epoch_time"].append(epoch_time)
            self.metrics_history["throughput"].append(throughput)

            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.save_checkpoint("best_model", epoch, val_metrics)
                self.save_model_for_eval("best_model.pth", epoch)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}", epoch, val_metrics)

            if self.should_early_stop(val_metrics["loss"]):
                print(f"event=early_stop epoch={epoch + 1} val_loss={val_metrics['loss']:.4f}")
                break

        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)
        print(f"event=train_complete best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)
