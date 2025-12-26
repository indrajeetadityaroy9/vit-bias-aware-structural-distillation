"""
Knowledge Distillation for DeiT (Data-efficient Image Transformer).

Implements distillation training where a Vision Transformer (student) learns from
a pre-trained CNN (teacher) using a distillation token.

Based on: "Training data-efficient image transformers & distillation through attention"
(Touvron et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import logging
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict

from src.training import DDPTrainer, LabelSmoothingCrossEntropy
from src.config import (
    Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig,
    ViTConfig, DistillationConfig
)

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Combined loss for DeiT distillation training.

    Supports two distillation modes:
    1. Hard distillation (default): Uses argmax of teacher predictions
       loss = (1-alpha)*CE(cls_out, targets) + alpha*CE(dist_out, argmax(teacher))

    2. Soft distillation: Uses temperature-scaled KL divergence
       loss = (1-alpha)*CE(cls_out, targets) + alpha*tau^2*KL(dist_out/tau, teacher/tau)

    Args:
        base_criterion: Loss function for ground truth (e.g., LabelSmoothingCrossEntropy)
        distillation_type: 'hard' or 'soft'
        alpha: Weight for distillation loss (0 = no distillation, 1 = only distillation)
        tau: Temperature for soft distillation
    """

    def __init__(self, base_criterion, distillation_type='hard', alpha=0.5, tau=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.distillation_type = distillation_type

        # Validate alpha is in [0, 1]
        if not 0 <= alpha <= 1:
            raise ValueError(f"Distillation alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

        # Validate tau is positive (for soft distillation)
        if tau <= 0:
            raise ValueError(f"Distillation tau must be positive, got {tau}")
        self.tau = tau

    def forward(self, student_cls_output, student_dist_output, targets, teacher_output):
        """
        Compute distillation loss.

        Args:
            student_cls_output: Student [CLS] token predictions (B, num_classes)
            student_dist_output: Student [DIST] token predictions (B, num_classes)
            targets: Ground truth labels (B,) or (B, num_classes) for soft labels
            teacher_output: Teacher model predictions (B, num_classes)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Ground truth loss on [CLS] token
        cls_loss = self.base_criterion(student_cls_output, targets)

        if self.distillation_type == 'hard':
            # Hard labels from teacher (argmax)
            teacher_labels = teacher_output.argmax(dim=1)
            dist_loss = F.cross_entropy(student_dist_output, teacher_labels)
        else:
            # Soft distillation with temperature scaling
            soft_teacher = F.softmax(teacher_output / self.tau, dim=1)
            soft_student = F.log_softmax(student_dist_output / self.tau, dim=1)
            dist_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            # Scale by tau^2 as per Hinton et al.
            dist_loss = dist_loss * (self.tau ** 2)

        # Combined loss
        total_loss = (1 - self.alpha) * cls_loss + self.alpha * dist_loss

        loss_dict = {
            'cls_loss': cls_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


class DistillationTrainer(DDPTrainer):
    """
    DDP Trainer for knowledge distillation with DeiT.

    Extends DDPTrainer with distillation-specific functionality:
    - Manages both student (DeiT) and teacher (CNN) models
    - Uses DistillationLoss for combined training
    - Tracks additional metrics: cls_loss, dist_loss, agreement_rate
    - Supports distillation warmup (no distillation for first N epochs)

    Args:
        student_model: DeiT model wrapped with DDP
        teacher_model: Pre-trained CNN teacher model (frozen)
        config: Training configuration
        device: CUDA device
        rank: DDP process rank
        world_size: Number of DDP processes
    """

    def __init__(self, student_model, teacher_model, config, device, rank, world_size):
        # Initialize parent with student model
        super().__init__(student_model, config, device, rank, world_size)

        # Store teacher model (frozen, eval mode)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Get distillation config
        distill_config = config.distillation
        self.distillation_type = distill_config.distillation_type
        self.distillation_alpha = distill_config.alpha
        self.distillation_tau = distill_config.tau
        self.distillation_warmup_epochs = distill_config.distillation_warmup_epochs

        # Alpha scheduling parameters
        self.alpha_schedule = getattr(distill_config, 'alpha_schedule', 'constant')
        self.alpha_start = getattr(distill_config, 'alpha_start', 0.0)
        self.alpha_end = getattr(distill_config, 'alpha_end', distill_config.alpha)
        self.num_epochs = config.training.num_epochs

        # Create distillation loss (alpha will be updated per epoch for scheduling)
        self.distillation_criterion = DistillationLoss(
            base_criterion=self.criterion,
            distillation_type=self.distillation_type,
            alpha=self.distillation_alpha,
            tau=self.distillation_tau
        )

        # Additional metrics tracking
        self.distillation_metrics = defaultdict(list)

        if self.is_main_process:
            logger.info(f"Distillation Trainer initialized:")
            logger.info(f"  - Type: {self.distillation_type}")
            logger.info(f"  - Alpha: {self.distillation_alpha}")
            logger.info(f"  - Alpha schedule: {self.alpha_schedule}")
            if self.alpha_schedule != 'constant':
                logger.info(f"  - Alpha range: {self.alpha_start} -> {self.alpha_end}")
            logger.info(f"  - Tau: {self.distillation_tau}")
            logger.info(f"  - Warmup epochs: {self.distillation_warmup_epochs}")

    def get_scheduled_alpha(self, epoch):
        """
        Get the scheduled alpha value for the current epoch.

        Supports 'constant', 'linear', and 'cosine' schedules.
        Alpha scheduling starts after warmup epochs.
        """
        if self.alpha_schedule == 'constant':
            return self.distillation_alpha

        # Calculate progress excluding warmup epochs
        effective_epoch = max(0, epoch - self.distillation_warmup_epochs)
        effective_total = max(1, self.num_epochs - self.distillation_warmup_epochs)
        progress = min(1.0, effective_epoch / effective_total)

        if self.alpha_schedule == 'linear':
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        elif self.alpha_schedule == 'cosine':
            # Cosine annealing from alpha_start to alpha_end
            import math
            return self.alpha_start + (self.alpha_end - self.alpha_start) * (1 - math.cos(progress * math.pi)) / 2
        else:
            return self.distillation_alpha

    def train_epoch_ddp(self, train_loader, train_sampler):
        """
        Train one epoch with distillation.

        Handles dual outputs from DeiT (cls_logits, dist_logits) and
        computes distillation loss from frozen teacher.
        """
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()  # Ensure teacher stays in eval mode

        total_loss = 0
        total_cls_loss = 0
        total_dist_loss = 0
        correct = 0
        total = 0
        agreement_total = 0
        batch_count = 0

        # Check if we're in warmup phase (no distillation)
        in_warmup = self.current_epoch < self.distillation_warmup_epochs

        # Update alpha based on schedule (after warmup)
        if not in_warmup:
            current_alpha = self.get_scheduled_alpha(self.current_epoch)
            self.distillation_criterion.alpha = current_alpha
        else:
            current_alpha = 0.0

        if self.is_main_process:
            desc = f"Epoch {self.current_epoch + 1}"
            if in_warmup:
                desc += " [Warmup - No Distillation]"
            elif self.alpha_schedule != 'constant':
                desc += f" [α={current_alpha:.3f}]"
            pbar = tqdm(train_loader, desc=desc)
        else:
            pbar = train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    # Student forward pass - returns (cls_logits, dist_logits) during training
                    student_output = self.ddp_model(inputs)

                    if isinstance(student_output, tuple):
                        cls_output, dist_output = student_output
                    else:
                        # Fallback if model returns single output
                        cls_output = student_output
                        dist_output = student_output

                    # Teacher forward pass (no grad)
                    with torch.no_grad():
                        teacher_output = self.teacher_model(inputs)

                    if in_warmup:
                        # Warmup: only use classification loss
                        loss = self.criterion(cls_output, targets)
                        cls_loss_val = loss.item()
                        dist_loss_val = 0.0
                    else:
                        # Full distillation loss
                        loss, loss_dict = self.distillation_criterion(
                            cls_output, dist_output, targets, teacher_output
                        )
                        cls_loss_val = loss_dict['cls_loss']
                        dist_loss_val = loss_dict['dist_loss']

                    loss = loss / self.grad_accum_steps

                if self.scaler is not None:
                    # FP16 with GradScaler
                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.ddp_model.parameters(),
                                self.config.training.gradient_clip_val
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # BF16 without GradScaler
                    loss.backward()

                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.ddp_model.parameters(),
                                self.config.training.gradient_clip_val
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                # Non-AMP path
                student_output = self.ddp_model(inputs)

                if isinstance(student_output, tuple):
                    cls_output, dist_output = student_output
                else:
                    cls_output = student_output
                    dist_output = student_output

                with torch.no_grad():
                    teacher_output = self.teacher_model(inputs)

                if in_warmup:
                    loss = self.criterion(cls_output, targets)
                    cls_loss_val = loss.item()
                    dist_loss_val = 0.0
                else:
                    loss, loss_dict = self.distillation_criterion(
                        cls_output, dist_output, targets, teacher_output
                    )
                    cls_loss_val = loss_dict['cls_loss']
                    dist_loss_val = loss_dict['dist_loss']

                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.ddp_model.parameters(),
                            self.config.training.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            total_cls_loss += cls_loss_val
            total_dist_loss += dist_loss_val

            # Accuracy based on cls head
            _, predicted = cls_output.max(1)

            # Handle one-hot encoded targets
            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)

            # Agreement rate: % where student distillation head agrees with teacher
            if not in_warmup:
                teacher_preds = teacher_output.argmax(dim=1)
                student_dist_preds = dist_output.argmax(dim=1)
                agreement_total += (student_dist_preds == teacher_preds).sum().item()

            batch_count += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                postfix = {
                    'Loss': f'{total_loss/batch_count:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                if not in_warmup:
                    postfix['Agree'] = f'{100.*agreement_total/total:.1f}%'
                pbar.set_postfix(postfix)

            self.global_step += 1

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        cls_loss_tensor = torch.tensor([total_cls_loss], device=self.device)
        dist_loss_tensor = torch.tensor([total_dist_loss], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)
        agreement_tensor = torch.tensor([agreement_total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(cls_loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(dist_loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(agreement_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / (batch_count * self.world_size)
        avg_cls_loss = cls_loss_tensor.item() / (batch_count * self.world_size)
        avg_dist_loss = dist_loss_tensor.item() / (batch_count * self.world_size)
        avg_acc = 100. * correct_tensor.item() / total_tensor.item()
        avg_agreement = 100. * agreement_tensor.item() / total_tensor.item() if not in_warmup else 0.0

        metrics = {
            'train_loss': avg_loss,
            'train_cls_loss': avg_cls_loss,
            'train_dist_loss': avg_dist_loss,
            'train_acc': avg_acc,
            'train_agreement': avg_agreement
        }

        return metrics

    @torch.no_grad()
    def validate_ddp(self, val_loader):
        """
        Validate with detailed metrics for distillation.

        Reports:
        - Overall accuracy (averaged cls + dist outputs)
        - cls_only_acc: accuracy using only cls head
        - dist_only_acc: accuracy using only dist head
        """
        self.ddp_model.eval()
        total_loss = 0
        correct_avg = 0
        correct_cls = 0
        correct_dist = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Get separate outputs using the model's method
            model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model

            if hasattr(model, 'get_classifier_outputs'):
                cls_out, dist_out, avg_out = model.get_classifier_outputs(inputs)
            else:
                # Fallback
                outputs = self.ddp_model(inputs)
                if isinstance(outputs, tuple):
                    cls_out, dist_out = outputs
                    avg_out = (cls_out + dist_out) / 2
                else:
                    cls_out = outputs
                    dist_out = outputs
                    avg_out = outputs

            loss = self.criterion(avg_out, targets)
            total_loss += loss.item() * inputs.size(0)

            # Handle one-hot encoded targets
            if len(targets.shape) > 1:
                targets = targets.argmax(1)

            # Track accuracy for each output
            correct_avg += avg_out.argmax(1).eq(targets).sum().item()
            correct_cls += cls_out.argmax(1).eq(targets).sum().item()
            if dist_out is not None:
                correct_dist += dist_out.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        correct_avg_tensor = torch.tensor([correct_avg], device=self.device)
        correct_cls_tensor = torch.tensor([correct_cls], device=self.device)
        correct_dist_tensor = torch.tensor([correct_dist], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_avg_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_cls_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_dist_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / total_tensor.item()
        avg_acc = 100. * correct_avg_tensor.item() / total_tensor.item()
        cls_acc = 100. * correct_cls_tensor.item() / total_tensor.item()
        dist_acc = 100. * correct_dist_tensor.item() / total_tensor.item()

        metrics = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'val_cls_acc': cls_acc,
            'val_dist_acc': dist_acc
        }

        return metrics

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """
        Full distillation training loop.
        """
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            logger.info(f"Starting DeiT distillation training for {num_epochs} epochs")
            logger.info(f"World Size: {self.world_size}")
            logger.info(f"Effective Batch Size: {self.config.data.batch_size * self.world_size}")
            logger.info(f"Distillation warmup: {self.distillation_warmup_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # LR Warmup phase (separate from distillation warmup)
            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training
            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)

            # Validation
            val_metrics = self.validate_ddp(val_loader)

            # Update scheduler
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Only rank 0 logs and saves
            if self.is_main_process:
                in_warmup = epoch < self.distillation_warmup_epochs
                warmup_indicator = " [Warmup]" if in_warmup else ""

                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}]{warmup_indicator} "
                    f"Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}% "
                    f"(cls: {val_metrics['val_cls_acc']:.2f}%, dist: {val_metrics['val_dist_acc']:.2f}%), "
                    f"Agreement: {train_metrics['train_agreement']:.1f}%, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Save metrics history
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best (based on averaged accuracy)
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                # Periodic checkpoint
                if (epoch + 1) % self.config.logging.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Synchronize all processes after each epoch
            self.dist.barrier()

        # SWA finalization
        if self.use_swa and self.is_main_process:
            logger.info("Updating batch normalization statistics for SWA model")
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            logger.info(f"Distillation training completed. Best Val Acc: {self.best_val_acc:.2f}%")

        return dict(self.metrics_history)

    def save_checkpoint(self, filename, epoch, metrics):
        """Save checkpoint with distillation-specific information."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (unwrap DDP)
        model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'metrics_history': dict(self.metrics_history),
            # Distillation-specific info
            'distillation_config': {
                'type': self.distillation_type,
                'alpha': self.distillation_alpha,
                'tau': self.distillation_tau,
                'warmup_epochs': self.distillation_warmup_epochs,
                'teacher_checkpoint': self.config.distillation.teacher_checkpoint
            }
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.use_swa:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()

        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
