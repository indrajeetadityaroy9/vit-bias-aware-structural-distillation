"""
Distillation training engines.

Provides:
- DistillationTrainer: DeiT-style CNN→ViT distillation
- SelfSupervisedDistillationTrainer: CST-style DINO→ViT distillation
"""

import math
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from src.training import DDPTrainer, build_checkpoint_dict

from .losses import DistillationLoss, SelfSupervisedDistillationLoss


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
            alpha_str = f" alpha_range={self.alpha_start}->{self.alpha_end}" if self.alpha_schedule != 'constant' else ""
            print(f"distill type={self.distillation_type} alpha={self.distillation_alpha} schedule={self.alpha_schedule}{alpha_str} tau={self.distillation_tau} warmup={self.distillation_warmup_epochs}")

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

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

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
                        self.optimizer.zero_grad(set_to_none=True)
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
                        self.optimizer.zero_grad(set_to_none=True)
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
                    self.optimizer.zero_grad(set_to_none=True)

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
        """Full distillation training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            print(f"start=distill epochs={num_epochs} gpus={self.world_size} batch_size={self.config.data.batch_size * self.world_size} warmup={self.distillation_warmup_epochs}")

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

            if self.is_main_process:
                warmup_str = " warmup" if epoch < self.distillation_warmup_epochs else ""
                print(f"epoch={epoch + 1}/{num_epochs}{warmup_str} loss={train_metrics['train_loss']:.4f} "
                      f"train_acc={train_metrics['train_acc']:.2f} val_acc={val_metrics['val_acc']:.2f} "
                      f"cls_acc={val_metrics['val_cls_acc']:.2f} dist_acc={val_metrics['val_dist_acc']:.2f} "
                      f"agreement={train_metrics['train_agreement']:.1f} lr={current_lr:.6f} time={epoch_time:.1f}s")

                # Save metrics history
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best (based on averaged accuracy)
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                # Periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"early_stop epoch={epoch + 1}")
                        break

            self.dist.barrier()

        if self.use_swa and self.is_main_process:
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            print(f"done best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)

    def save_checkpoint(self, filename, epoch, metrics):
        """Save checkpoint with distillation-specific information using shared utility."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model (unwrap DDP)
        model = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
        swa_model = self.swa_model if self.use_swa else None
        scaler = self.scaler if self.use_amp else None

        # Distillation-specific metadata
        extra_metadata = {
            'distillation_config': {
                'type': self.distillation_type,
                'alpha': self.distillation_alpha,
                'tau': self.distillation_tau,
                'warmup_epochs': self.distillation_warmup_epochs,
                'teacher_checkpoint': self.config.distillation.teacher_checkpoint
            }
        }

        checkpoint = build_checkpoint_dict(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=scaler,
            swa_model=swa_model,
            epoch=epoch,
            metrics=metrics,
            config=self.config,
            best_val_acc=self.best_val_acc,
            metrics_history=self.metrics_history,
            extra_metadata=extra_metadata
        )

        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"checkpoint={save_path}")


class SelfSupervisedDistillationTrainer(DDPTrainer):
    """
    Trainer for CST-style self-supervised token correlation distillation.

    Uses a pretrained DINO/DINOv2 teacher instead of a weaker CNN.
    Implements two-stage training:
    - Stage A: L_ce + L_tok (representation matching)
    - Stage B: L_ce + L_tok + L_rel (add correlation matching)
    """

    def __init__(self, student_model, teacher_model, teacher_embed_dim,
                 config, device, rank, world_size):
        # Initialize base trainer
        super().__init__(student_model, config, device, rank, world_size)

        # Store teacher
        self.teacher_model = teacher_model
        self.teacher_embed_dim = teacher_embed_dim

        # Get configs
        self.ss_config = config.ss_distillation
        self.token_layers = self.ss_config.token_layers

        # Get student embedding dim
        student_module = student_model.module if hasattr(student_model, 'module') else student_model
        self.student_embed_dim = student_module.embed_dim

        # Compute student token count for interpolation
        # For CIFAR-10 with patch_size=4: 32/4 = 8, so 8x8 = 64 tokens
        self.student_num_tokens = student_module.patch_embed.num_patches

        # Create distillation loss
        self.distillation_criterion = SelfSupervisedDistillationLoss(
            base_criterion=self.criterion,
            student_dim=self.student_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            config=self.ss_config
        ).to(device)

        # CRITICAL: Add projection head params to optimizer
        self.optimizer.add_param_group({
            'params': self.distillation_criterion.token_rep_loss.parameters(),
            'lr': config.training.learning_rate
        })

        # Add structural loss parameters if CKA/Gram enabled
        if self.distillation_criterion.structural_loss is not None:
            self.optimizer.add_param_group({
                'params': self.distillation_criterion.structural_loss.parameters(),
                'lr': config.training.learning_rate
            })
        if self.distillation_criterion.gram_only_loss is not None:
            self.optimizer.add_param_group({
                'params': self.distillation_criterion.gram_only_loss.parameters(),
                'lr': config.training.learning_rate
            })

        # Track structural loss usage
        self.use_cka = getattr(self.ss_config, 'use_cka_loss', False)
        self.use_gram = getattr(self.ss_config, 'use_gram_loss', False)

        # Track dual-augment mode
        self.use_dual_augment = getattr(self.ss_config, 'use_dual_augment', False)

        # CLS-only mode for CKA (cleaner global semantic alignment)
        self.use_cls_only = getattr(self.ss_config, 'use_cls_only', False)

        # Create MixUp function for training loop (applied only to student images)
        self.mixup_fn = None
        self.num_classes = config.model.num_classes
        aug_config = config.data.augmentation
        if aug_config.get('mixup') or aug_config.get('cutmix'):
            self.mixup_alpha = aug_config.get('mixup_alpha', 1.0) if aug_config.get('mixup') else 0.0
            self.cutmix_alpha = aug_config.get('cutmix_alpha', 1.0) if aug_config.get('cutmix') else 0.0

        if self.is_main_process:
            cka_str = f" cka=lambda={getattr(self.ss_config, 'lambda_cka', 0.5)}" if self.use_cka else ""
            gram_str = f" gram=lambda={getattr(self.ss_config, 'lambda_gram', 0.5)}" if self.use_gram else ""
            cls_str = " cls_only" if self.use_cls_only else ""
            print(f"ss_distill teacher={self.ss_config.teacher_model_name} teacher_dim={self.teacher_embed_dim} "
                  f"student_dim={self.student_embed_dim} layers={self.token_layers} "
                  f"lambda_tok={self.ss_config.lambda_tok} lambda_rel={self.ss_config.lambda_rel} "
                  f"rel_warmup={self.ss_config.rel_warmup_epochs} dual_aug={self.use_dual_augment}{cka_str}{gram_str}{cls_str}")

    def apply_mixup(self, images, targets):
        """
        Apply MixUp or CutMix augmentation to images and targets.

        Args:
            images: Batch of images (B, C, H, W)
            targets: Integer class labels (B,)

        Returns:
            mixed_images: Augmented images
            mixed_targets: Soft labels (B, num_classes)
        """
        B = images.shape[0]
        device = images.device

        # Randomly choose between MixUp and CutMix
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha == 0 or torch.rand(1).item() > 0.5)

        if use_cutmix:
            alpha = self.cutmix_alpha
        else:
            alpha = self.mixup_alpha

        if alpha <= 0:
            # No mixing, just convert to one-hot
            targets_onehot = torch.zeros(B, self.num_classes, device=device)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
            return images, targets_onehot

        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)

        # Get random permutation for mixing
        rand_idx = torch.randperm(B, device=device)

        if use_cutmix:
            # CutMix: cut and paste patches
            _, _, H, W = images.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_idx, :, bby1:bby2, bbx1:bbx2]

            # Adjust lambda based on actual cut size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        else:
            # MixUp: linear interpolation
            mixed_images = lam * images + (1 - lam) * images[rand_idx]

        # Create soft labels
        targets_onehot = torch.zeros(B, self.num_classes, device=device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        targets_shuffled = torch.zeros(B, self.num_classes, device=device)
        targets_shuffled.scatter_(1, targets[rand_idx].unsqueeze(1), 1.0)

        mixed_targets = lam * targets_onehot + (1 - lam) * targets_shuffled

        return mixed_images, mixed_targets

    def get_teacher_intermediates(self, x, layer_indices):
        """
        Extract intermediate tokens from teacher (DINO/DINOv2).

        Handles resolution mismatch:
        1. Upsample input to 224x224
        2. Get teacher tokens (196 for 14x14 patches) OR CLS token only
        3. Interpolate teacher tokens to match student count (if using patches)

        Args:
            x: Input tensor (B, C, H, W) - original resolution
            layer_indices: List of layer indices to extract

        Returns:
            intermediates: Dict[layer_idx] -> (B, N_student, teacher_dim) or (B, 1, teacher_dim) for CLS
            patch_tokens: Final teacher patch/CLS tokens
        """
        with torch.no_grad():
            B = x.shape[0]

            # Upsample input to 224x224 for teacher (bicubic for sharper edges)
            if x.shape[-1] < 224:
                teacher_input = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
            else:
                teacher_input = x

            intermediates = {}

            # DINOv2 supports get_intermediate_layers
            if hasattr(self.teacher_model, 'get_intermediate_layers'):
                if self.use_cls_only:
                    # CLS-only mode: Extract CLS tokens for global semantic alignment
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=True
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        if isinstance(outputs[i], tuple):
                            cls_token = outputs[i][1]  # (B, D)
                        else:
                            cls_token = outputs[i][:, 0, :]  # (B, D)
                        intermediates[layer_idx] = cls_token.unsqueeze(1)

                    # Get final CLS token
                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[max(layer_indices)], return_class_token=True
                    )[0]
                    if isinstance(final_output, tuple):
                        patch_tokens = final_output[1].unsqueeze(1)
                    else:
                        patch_tokens = final_output[:, 0, :].unsqueeze(1)
                else:
                    # Patch token mode: Extract and interpolate patch tokens
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=False
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        teacher_tokens = outputs[i]
                        teacher_tokens = self._interpolate_tokens(teacher_tokens, self.student_num_tokens)
                        intermediates[layer_idx] = teacher_tokens

                    # Get final patch tokens
                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[11], return_class_token=False
                    )[0]
                    patch_tokens = self._interpolate_tokens(final_output, self.student_num_tokens)
            else:
                # Fallback: manual extraction via forward hooks
                hooks = []
                captured = {}

                def make_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            if self.use_cls_only:
                                captured[idx] = output[0][:, 0:1, :]
                            else:
                                captured[idx] = output[0][:, 1:, :]
                        else:
                            if self.use_cls_only:
                                captured[idx] = output[:, 0:1, :]
                            else:
                                captured[idx] = output[:, 1:, :]
                    return hook

                # Register hooks
                for idx in layer_indices:
                    if hasattr(self.teacher_model, 'blocks'):
                        hook = self.teacher_model.blocks[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)

                # Forward pass
                _ = self.teacher_model(teacher_input)

                # Remove hooks
                for hook in hooks:
                    hook.remove()

                # Process captured tokens
                for idx in layer_indices:
                    if idx in captured:
                        if self.use_cls_only:
                            intermediates[idx] = captured[idx]
                        else:
                            intermediates[idx] = self._interpolate_tokens(captured[idx], self.student_num_tokens)

                # Get final tokens from last layer
                last_idx = max(layer_indices)
                patch_tokens = intermediates[last_idx]

        return intermediates, patch_tokens

    def _interpolate_tokens(self, tokens, target_num_tokens):
        """
        Interpolate teacher tokens to match student token count.

        Teacher tokens: (B, N_teacher, D) where N_teacher = 196 (14x14)
        Target: (B, N_student, D) where N_student = 64 (8x8) for CIFAR
        """
        B, N, D = tokens.shape
        if N == target_num_tokens:
            return tokens

        # Compute grid sizes
        H_t = int(N ** 0.5)
        H_s = int(target_num_tokens ** 0.5)

        # Reshape to spatial grid: (B, N, D) -> (B, D, H, W)
        tokens = tokens.transpose(1, 2).reshape(B, D, H_t, H_t)

        # Interpolate
        tokens = F.interpolate(tokens, size=(H_s, H_s), mode='bilinear', align_corners=False)

        # Reshape back: (B, D, H, W) -> (B, N, D)
        tokens = tokens.reshape(B, D, -1).transpose(1, 2)

        return tokens

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch with self-supervised distillation."""
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()

        # Training state
        total_loss = 0
        total_ce_loss = 0
        total_tok_loss = 0
        total_rel_loss = 0
        total_cka_loss = 0
        total_gram_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # Stage indicator
        in_stage_a = self.current_epoch < self.ss_config.rel_warmup_epochs

        # Optional: freeze projectors for first N epochs
        if self.current_epoch < self.ss_config.projector_warmup_epochs:
            for p in self.distillation_criterion.token_rep_loss.parameters():
                p.requires_grad = False
        else:
            for p in self.distillation_criterion.token_rep_loss.parameters():
                p.requires_grad = True

        if self.is_main_process:
            stage = "A (L_tok only)" if in_stage_a else "B (L_tok + L_rel)"
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [{stage}]")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # Handle dual-augment batch format: (clean_imgs, student_imgs, targets)
            if self.use_dual_augment:
                clean_imgs, student_imgs, targets = batch
                clean_imgs = clean_imgs.to(self.device)
                student_imgs = student_imgs.to(self.device)
                targets = targets.to(self.device)

                # Apply MixUp/CutMix ONLY to student images
                if hasattr(self, 'mixup_alpha') and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
                    student_imgs, targets = self.apply_mixup(student_imgs, targets)

                teacher_input = clean_imgs
                student_input = student_imgs
            else:
                # Standard batch format: (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                teacher_input = inputs
                student_input = inputs

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    # Student forward with intermediates
                    student_module = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
                    student_results = student_module.forward_with_intermediates(
                        student_input, layer_indices=self.token_layers,
                        return_cls_only=self.use_cls_only
                    )
                    student_output = student_results['output']
                    student_intermediates = student_results['intermediates']
                    student_patch_tokens = student_results['patch_tokens']

                    # Teacher forward with intermediates
                    teacher_intermediates, teacher_patch_tokens = self.get_teacher_intermediates(
                        teacher_input, self.token_layers
                    )

                    # Compute combined loss
                    loss, loss_dict = self.distillation_criterion(
                        student_output, targets,
                        student_intermediates, teacher_intermediates,
                        student_patch_tokens, teacher_patch_tokens,
                        self.current_epoch
                    )

                    loss = loss / self.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.ddp_model.parameters()) +
                                list(self.distillation_criterion.token_rep_loss.parameters()),
                                self.config.training.gradient_clip_val
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.config.training.gradient_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.ddp_model.parameters()) +
                                list(self.distillation_criterion.token_rep_loss.parameters()),
                                self.config.training.gradient_clip_val
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
            else:
                # Non-AMP path
                student_module = self.ddp_model.module if hasattr(self.ddp_model, 'module') else self.ddp_model
                student_results = student_module.forward_with_intermediates(
                    student_input, layer_indices=self.token_layers,
                    return_cls_only=self.use_cls_only
                )

                teacher_intermediates, teacher_patch_tokens = self.get_teacher_intermediates(
                    teacher_input, self.token_layers
                )

                loss, loss_dict = self.distillation_criterion(
                    student_results['output'], targets,
                    student_results['intermediates'], teacher_intermediates,
                    student_results['patch_tokens'], teacher_patch_tokens,
                    self.current_epoch
                )

                loss = loss / self.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.ddp_model.parameters()) +
                            list(self.distillation_criterion.token_rep_loss.parameters()),
                            self.config.training.gradient_clip_val
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
            total_ce_loss += loss_dict['ce_loss']
            total_tok_loss += loss_dict['tok_loss']
            total_rel_loss += loss_dict['rel_loss']
            total_cka_loss += loss_dict.get('cka_loss', 0.0)
            total_gram_loss += loss_dict.get('gram_loss', 0.0)

            # Accuracy
            if isinstance(student_output, tuple):
                cls_output = student_output[0]
            else:
                cls_output = student_output
            _, predicted = cls_output.max(1)

            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)
            batch_count += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                postfix = {
                    'Loss': f'{total_loss/batch_count:.4f}',
                    'CE': f'{total_ce_loss/batch_count:.4f}',
                    'Tok': f'{total_tok_loss/batch_count:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                }
                if total_rel_loss > 0:
                    postfix['Rel'] = f'{total_rel_loss/batch_count:.4f}'
                if self.use_cka and total_cka_loss > 0:
                    postfix['CKA'] = f'{total_cka_loss/batch_count:.4f}'
                if self.use_gram and total_gram_loss > 0:
                    postfix['Gram'] = f'{total_gram_loss/batch_count:.4f}'
                pbar.set_postfix(postfix)

            self.global_step += 1

        # Aggregate metrics across GPUs
        metrics_tensor = torch.tensor([
            total_loss, total_ce_loss, total_tok_loss, total_rel_loss,
            total_cka_loss, total_gram_loss,
            correct, total, batch_count
        ], device=self.device)

        self.dist.all_reduce(metrics_tensor, op=self.dist.ReduceOp.SUM)

        batch_count_total = int(metrics_tensor[8].item())
        avg_loss = metrics_tensor[0].item() / batch_count_total
        avg_ce_loss = metrics_tensor[1].item() / batch_count_total
        avg_tok_loss = metrics_tensor[2].item() / batch_count_total
        avg_rel_loss = metrics_tensor[3].item() / batch_count_total
        avg_cka_loss = metrics_tensor[4].item() / batch_count_total
        avg_gram_loss = metrics_tensor[5].item() / batch_count_total
        avg_acc = 100. * metrics_tensor[6].item() / metrics_tensor[7].item()

        metrics = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_tok_loss': avg_tok_loss,
            'train_rel_loss': avg_rel_loss,
            'train_cka_loss': avg_cka_loss,
            'train_gram_loss': avg_gram_loss,
            'train_acc': avg_acc,
            'effective_lambda_rel': loss_dict['effective_lambda_rel'],
            'effective_lambda_cka': loss_dict.get('effective_lambda_cka', 0.0)
        }

        return metrics

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """Full self-supervised distillation training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            print(f"start=ss_distill epochs={num_epochs} gpus={self.world_size} batch_size={self.config.data.batch_size * self.world_size} "
                  f"stage_a=0-{self.ss_config.rel_warmup_epochs-1} stage_b={self.ss_config.rel_warmup_epochs}+")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # LR Warmup phase
            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Training
            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)

            # Validation (use base class method)
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

            if self.is_main_process:
                stage = "A" if epoch < self.ss_config.rel_warmup_epochs else "B"
                rel_str = f" rel={train_metrics['train_rel_loss']:.4f}" if train_metrics['train_rel_loss'] > 0 else ""
                cka_str = f" cka={train_metrics.get('train_cka_loss', 0):.4f}" if self.use_cka and train_metrics.get('train_cka_loss', 0) > 0 else ""
                gram_str = f" gram={train_metrics.get('train_gram_loss', 0):.4f}" if self.use_gram and train_metrics.get('train_gram_loss', 0) > 0 else ""

                print(f"epoch={epoch + 1}/{num_epochs} stage={stage} loss={train_metrics['train_loss']:.4f} "
                      f"ce={train_metrics['train_ce_loss']:.4f} tok={train_metrics['train_tok_loss']:.4f}{rel_str}{cka_str}{gram_str} "
                      f"train_acc={train_metrics['train_acc']:.2f} val_acc={val_metrics['val_acc']:.2f} "
                      f"lr={current_lr:.6f} time={epoch_time:.1f}s")

                # Save metrics history
                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best
                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                # Periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"early_stop epoch={epoch + 1}")
                        break

            self.dist.barrier()

        if self.use_swa and self.is_main_process:
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            print(f"done best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)


__all__ = ['DistillationTrainer', 'SelfSupervisedDistillationTrainer']
