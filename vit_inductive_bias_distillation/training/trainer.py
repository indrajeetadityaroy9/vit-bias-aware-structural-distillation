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

from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

from vit_inductive_bias_distillation.config import Config
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.losses.combined import BASDLoss
from vit_inductive_bias_distillation.models.teacher import TeacherModel, extract_intermediates


class Trainer:
    def __init__(
        self,
        student_model: nn.Module,
        config: Config,
        accelerator: Accelerator,
        teacher: TeacherModel | None = None,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.config = config
        self.distill = teacher is not None

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

        if self.distill:
            self._teacher = teacher
            student_embed_dim = student_model.embed_dim
            student_num_tokens = student_model.patch_embed.num_patches
            cross_attn_num_heads = max(1, teacher.embed_dim // 64)

            self.basd_loss = BASDLoss(
                base_criterion=self.criterion,
                student_dim=student_embed_dim,
                teacher_dim=teacher.embed_dim,
                student_depth=student_model.depth,
                num_student_tokens=student_num_tokens,
                cross_attn_num_heads=cross_attn_num_heads,
                config=config.basd,
                student_num_heads=config.model.vit.num_heads,
                teacher_num_heads=teacher.num_heads,
            ).to(self.device)

            self.token_layers = list(self.basd_loss.token_layers)

            self.optimizer.add_param_group({
                "params": list(self.basd_loss.parameters()),
                "lr": config.training.learning_rate,
            })

        self.model, self.optimizer, self.scheduler = accelerator.prepare(
            student_model, self.optimizer, self.scheduler
        )

        if self.distill:
            accelerator.register_for_checkpointing(self.basd_loss)

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
        self.best_val_acc = 0.0
        self.metrics_history: dict[str, list[Any]] = defaultdict(list)

        self.mixup_cutmix = RandomChoice([
            MixUp(alpha=0.8, num_classes=config.model.num_classes),
            CutMix(alpha=1.0, num_classes=config.model.num_classes),
        ])

    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints" / name
        self.accelerator.save_state(str(checkpoint_dir))

        custom_state = {
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "metrics_history": dict(self.metrics_history),
        }
        if self.distill:
            custom_state["schedule_state"] = self.basd_loss.schedule.state_dict()
            custom_state["lagrangian_state"] = self.basd_loss.lagrangian_reg.state_dict()
        torch.save(custom_state, checkpoint_dir / "custom_state.pth")
        print(f"event=checkpoint_saved path={checkpoint_dir} epoch={epoch + 1} name={name}")

    def save_model_for_eval(self, filename: str, epoch: int) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        torch.save(
            {"epoch": epoch, "model_state_dict": unwrapped.state_dict()},
            checkpoint_dir / filename,
        )

    def save_swa_for_eval(self, filename: str, epoch: int) -> None:
        checkpoint_dir = Path(self.config.run.output_dir) / self.config.run.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_swa = self.accelerator.unwrap_model(self.swa_model.module)
        torch.save(
            {"epoch": epoch, "model_state_dict": unwrapped_swa.state_dict()},
            checkpoint_dir / filename,
        )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        self.accelerator.load_state(checkpoint_path)

        custom = torch.load(
            Path(checkpoint_path) / "custom_state.pth",
            map_location=self.device,
        )
        self.best_val_acc = custom["best_val_acc"]
        self.metrics_history = defaultdict(list, custom["metrics_history"])
        if self.distill:
            self.basd_loss.schedule.load_state_dict(custom["schedule_state"])
            self.basd_loss.lagrangian_reg.load_state_dict(custom["lagrangian_state"])
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

        if self.distill:
            return self._train_epoch_distill(train_loader)
        return self._train_epoch_baseline(train_loader)

    def _train_epoch_baseline(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch in train_loader:
            inputs = batch["pixel_values"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            inputs, mixed_targets = self.mixup_cutmix(inputs, targets)

            with self.accelerator.autocast():
                student_results = self.model(inputs)
                loss = self.criterion(student_results.output, mixed_targets)

            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self._grad_clip_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            predicted = student_results.output.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

        return {
            "train_total_loss": total_loss / batch_count,
            "train_acc": 100.0 * correct / total,
            "total_samples": total,
        }

    def _train_epoch_distill(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> dict[str, Any]:
        self._teacher.model.eval()

        _LOSS_KEYS = ("ce_loss", "rsd_loss", "rel_loss", "attn_loss", "spectral_loss")
        loss_accum: dict[str, float] = defaultdict(float)
        correct = 0
        total = 0
        batch_count = 0

        for batch in train_loader:
            clean_imgs = batch["clean"].to(self.device, non_blocking=True)
            student_imgs = batch["augmented"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            student_imgs, mixed_targets = self.mixup_cutmix(student_imgs, targets)

            with self.accelerator.autocast():
                student_results = self.model(
                    student_imgs,
                    layer_indices=self.token_layers,
                )

                teacher_results = extract_intermediates(
                    self._teacher.model, clean_imgs,
                    num_layers=self._teacher.num_layers,
                    loader=self._teacher.loader,
                    feature_format=self._teacher.feature_format,
                )

                loss, loss_dict = self.basd_loss(
                    student_results.output,
                    mixed_targets,
                    student_results.intermediates,
                    student_results.attention_weights,
                    teacher_results.all_tokens,
                    teacher_results.all_attentions,
                )

            self.accelerator.backward(loss)
            self.basd_loss.post_step_update(loss_dict)

            self.accelerator.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.basd_loss.parameters()),
                self._grad_clip_norm,
            )
            self.optimizer.step()
            self.basd_loss.project_to_stiefel()
            self.optimizer.zero_grad(set_to_none=True)

            loss_accum["total_loss"] += loss.item()
            for key in _LOSS_KEYS:
                loss_accum[key] += loss_dict[key]

            predicted = student_results.output.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

        result = {f"train_{k}": v / batch_count for k, v in loss_accum.items()}
        result["train_acc"] = 100.0 * correct / total
        result["total_samples"] = total
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
        epoch = start_epoch

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

            if self.distill:
                print(
                    f"event=epoch_summary mode=distill epoch={epoch + 1} num_epochs={num_epochs} "
                    f"train_loss={train_metrics['train_total_loss']:.6f} "
                    f"ce={train_metrics['train_ce_loss']:.6f} "
                    f"rsd={train_metrics['train_rsd_loss']:.6f} "
                    f"rel={train_metrics['train_rel_loss']:.6f} "
                    f"attn={train_metrics['train_attn_loss']:.6f} "
                    f"spectral={train_metrics['train_spectral_loss']:.6f} "
                    f"train_acc={train_metrics['train_acc']:.4f} "
                    f"val_acc={val_metrics['val_acc']:.4f} "
                    f"lr={current_lr:.8f} "
                    f"epoch_time_s={epoch_time:.3f} "
                    f"throughput_img_per_sec={throughput:.2f}"
                )
            else:
                print(
                    f"event=epoch_summary mode=baseline epoch={epoch + 1} num_epochs={num_epochs} "
                    f"train_loss={train_metrics['train_total_loss']:.6f} "
                    f"train_acc={train_metrics['train_acc']:.4f} "
                    f"val_acc={val_metrics['val_acc']:.4f} "
                    f"lr={current_lr:.8f} "
                    f"epoch_time_s={epoch_time:.3f} "
                    f"throughput_img_per_sec={throughput:.2f}"
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
                print(f"event=early_stop epoch={epoch + 1} val_loss={val_metrics['loss']:.6f}")
                break

        if self.distill:
            def _bn_loader():
                for batch in train_loader:
                    yield batch["clean"], batch["label"]
            torch.optim.swa_utils.update_bn(_bn_loader(), self.swa_model, self.device)
        else:
            def _bn_loader():
                for batch in train_loader:
                    yield batch["pixel_values"], batch["label"]
            torch.optim.swa_utils.update_bn(_bn_loader(), self.swa_model, self.device)
        self.save_swa_for_eval("swa_model.pth", epoch)
        print(f"event=train_complete best_val_acc={self.best_val_acc:.4f}")

        return dict(self.metrics_history)
