"""
Distillation training engines.

Provides:
- DistillationTrainer: DeiT-style CNN->ViT distillation
- SelfSupervisedDistillationTrainer: CST-style DINO->ViT distillation
"""

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data import apply_batch_mixing
from src.training.losses import DistillationLoss, SelfSupervisedDistillationLoss
from src.models.teachers import DINO_NUM_LAYERS
from src.training.trainer import DDPTrainer
from src.training.checkpointing import build_checkpoint_dict


class DistillationTrainer(DDPTrainer):
    """DDP Trainer for knowledge distillation with DeiT."""

    def __init__(self, student_model, teacher_model, config, device, rank, world_size):
        super().__init__(student_model, config, device, rank, world_size)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        distill_config = config.distillation
        self.distillation_type = distill_config.distillation_type
        self.distillation_alpha = distill_config.alpha
        self.distillation_tau = distill_config.tau
        self.distillation_warmup_epochs = distill_config.distillation_warmup_epochs

        self.alpha_schedule = distill_config.alpha_schedule
        self.alpha_start = distill_config.alpha_start
        self.alpha_end = distill_config.alpha_end
        self.num_epochs = config.training.num_epochs

        self.distillation_criterion = DistillationLoss(
            base_criterion=self.criterion,
            distillation_type=self.distillation_type,
            alpha=self.distillation_alpha,
            tau=self.distillation_tau
        )

        if self.is_main_process:
            alpha_str = f" alpha_range={self.alpha_start}->{self.alpha_end}" if self.alpha_schedule != 'constant' else ""
            print(f"distill type={self.distillation_type} alpha={self.distillation_alpha} schedule={self.alpha_schedule}{alpha_str} tau={self.distillation_tau} warmup={self.distillation_warmup_epochs}")

    def get_scheduled_alpha(self, epoch):
        """Get the scheduled alpha value for the current epoch."""
        if self.alpha_schedule == 'constant':
            return self.distillation_alpha

        effective_epoch = max(0, epoch - self.distillation_warmup_epochs)
        effective_total = max(1, self.num_epochs - self.distillation_warmup_epochs)
        progress = min(1.0, effective_epoch / effective_total)

        if self.alpha_schedule == 'linear':
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        elif self.alpha_schedule == 'cosine':
            return self.alpha_start + (self.alpha_end - self.alpha_start) * (1 - math.cos(progress * math.pi)) / 2
        else:
            return self.distillation_alpha

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch with distillation."""
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()

        total_loss = 0
        total_cls_loss = 0
        total_dist_loss = 0
        correct = 0
        total = 0
        agreement_total = 0
        batch_count = 0

        in_warmup = self.current_epoch < self.distillation_warmup_epochs

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
                desc += f" [\u03b1={current_alpha:.3f}]"
            pbar = tqdm(train_loader, desc=desc)
        else:
            pbar = train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.mixup_alpha > 0 or self.cutmix_alpha > 0:
                inputs, targets = apply_batch_mixing(
                    inputs, targets, self.num_classes,
                    self.mixup_alpha, self.cutmix_alpha,
                )

            with autocast(device_type='cuda', dtype=self.autocast_dtype):
                cls_output, dist_output = self.ddp_model(inputs)

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

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

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

            _, predicted = cls_output.max(1)
            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)

            if not in_warmup:
                teacher_preds = teacher_output.argmax(dim=1)
                student_dist_preds = dist_output.argmax(dim=1)
                agreement_total += (student_dist_preds == teacher_preds).sum().item()

            batch_count += 1

            if self.is_main_process:
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

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(dist_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(agreement_tensor, op=dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / (batch_count * self.world_size)
        avg_cls_loss = cls_loss_tensor.item() / (batch_count * self.world_size)
        avg_dist_loss = dist_loss_tensor.item() / (batch_count * self.world_size)
        avg_acc = 100. * correct_tensor.item() / total_tensor.item()
        avg_agreement = 100. * agreement_tensor.item() / total_tensor.item() if not in_warmup else 0.0

        return {
            'train_loss': avg_loss,
            'train_cls_loss': avg_cls_loss,
            'train_dist_loss': avg_dist_loss,
            'train_acc': avg_acc,
            'train_agreement': avg_agreement
        }

    @torch.no_grad()
    def validate_ddp(self, val_loader):
        """Validate with detailed metrics for distillation."""
        self.ddp_model.eval()
        total_loss = 0
        correct_avg = 0
        correct_cls = 0
        correct_dist = 0
        total = 0

        model = self.ddp_model.module

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Use forward_features to get separate cls/dist outputs in eval mode
            features = model.forward_features(inputs)
            cls_out = model.head(features[:, 0])
            dist_out = model.head_dist(features[:, 1])
            avg_out = (cls_out + dist_out) / 2

            loss = self.criterion(avg_out, targets)
            total_loss += loss.item() * inputs.size(0)

            if len(targets.shape) > 1:
                targets = targets.argmax(1)

            correct_avg += avg_out.argmax(1).eq(targets).sum().item()
            correct_cls += cls_out.argmax(1).eq(targets).sum().item()
            correct_dist += dist_out.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        loss_tensor = torch.tensor([total_loss], device=self.device)
        correct_avg_tensor = torch.tensor([correct_avg], device=self.device)
        correct_cls_tensor = torch.tensor([correct_cls], device=self.device)
        correct_dist_tensor = torch.tensor([correct_dist], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_avg_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_cls_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_dist_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / total_tensor.item()
        avg_acc = 100. * correct_avg_tensor.item() / total_tensor.item()
        cls_acc = 100. * correct_cls_tensor.item() / total_tensor.item()
        dist_acc = 100. * correct_dist_tensor.item() / total_tensor.item()

        return {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'val_cls_acc': cls_acc,
            'val_dist_acc': dist_acc
        }

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """Full distillation training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            print(f"start=distill epochs={num_epochs} gpus={self.world_size} batch_size={self.config.data.batch_size * self.world_size} warmup={self.distillation_warmup_epochs}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)
            val_metrics = self.validate_ddp(val_loader)

            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
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

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"early_stop epoch={epoch + 1}")
                        break

            dist.barrier()

        if self.use_swa and self.is_main_process:
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            print(f"done best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)

    def save_checkpoint(self, filename, epoch, metrics):
        """Save checkpoint with distillation-specific information."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            model=self.ddp_model.module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            swa_model=self.swa_model if self.use_swa else None,
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
        super().__init__(student_model, config, device, rank, world_size)

        self.teacher_model = teacher_model
        self.teacher_embed_dim = teacher_embed_dim

        self.ss_config = config.ss_distillation
        self.token_layers = self.ss_config.token_layers

        student_module = student_model.module
        self.student_embed_dim = student_module.embed_dim
        self.student_num_tokens = student_module.patch_embed.num_patches

        self.distillation_criterion = SelfSupervisedDistillationLoss(
            base_criterion=self.criterion,
            student_dim=self.student_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            config=self.ss_config
        ).to(device)

        # Add projection head params to optimizer
        self.optimizer.add_param_group({
            'params': self.distillation_criterion.token_rep_loss.parameters(),
            'lr': config.training.learning_rate
        })

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

        self.use_cka = self.ss_config.use_cka_loss
        self.use_gram = self.ss_config.use_gram_loss
        self.use_dual_augment = self.ss_config.use_dual_augment
        self.use_cls_only = self.ss_config.use_cls_only

        # Dynamic last layer index for teacher (fixes hard-coded [11])
        teacher_model_name = self.ss_config.teacher_model_name
        self.teacher_last_layer = DINO_NUM_LAYERS[teacher_model_name] - 1

        if self.is_main_process:
            cka_str = f" cka=lambda={self.ss_config.lambda_cka}" if self.use_cka else ""
            gram_str = f" gram=lambda={self.ss_config.lambda_gram}" if self.use_gram else ""
            cls_str = " cls_only" if self.use_cls_only else ""
            print(f"ss_distill teacher={self.ss_config.teacher_model_name} teacher_dim={self.teacher_embed_dim} "
                  f"student_dim={self.student_embed_dim} layers={self.token_layers} "
                  f"lambda_tok={self.ss_config.lambda_tok} lambda_rel={self.ss_config.lambda_rel} "
                  f"rel_warmup={self.ss_config.rel_warmup_epochs} dual_aug={self.use_dual_augment}{cka_str}{gram_str}{cls_str}")

    def get_teacher_intermediates(self, x, layer_indices):
        """
        Extract intermediate tokens from teacher (DINO/DINOv2).

        Handles resolution mismatch by upsampling input to 224x224 for teacher,
        then interpolating teacher tokens to match student token count.
        """
        with torch.no_grad():
            B = x.shape[0]

            if x.shape[-1] < 224:
                teacher_input = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
            else:
                teacher_input = x

            intermediates = {}

            if self.ss_config.teacher_type == 'dinov2':
                if self.use_cls_only:
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=True
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        cls_token = outputs[i][1]  # (B, D) from (patch_tokens, cls_token) tuple
                        intermediates[layer_idx] = cls_token.unsqueeze(1)

                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[max(layer_indices)], return_class_token=True
                    )[0]
                    patch_tokens = final_output[1].unsqueeze(1)
                else:
                    outputs = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=layer_indices, return_class_token=False
                    )
                    for i, layer_idx in enumerate(layer_indices):
                        teacher_tokens = outputs[i]
                        teacher_tokens = self._interpolate_tokens(teacher_tokens, self.student_num_tokens)
                        intermediates[layer_idx] = teacher_tokens

                    final_output = self.teacher_model.get_intermediate_layers(
                        teacher_input, n=[self.teacher_last_layer], return_class_token=False
                    )[0]
                    patch_tokens = self._interpolate_tokens(final_output, self.student_num_tokens)
            else:
                # DINO v1: manual extraction via forward hooks
                hooks = []
                captured = {}

                def make_hook(idx):
                    def hook(module, input, output):
                        if self.use_cls_only:
                            captured[idx] = output[:, 0:1, :]
                        else:
                            captured[idx] = output[:, 1:, :]
                    return hook

                for idx in layer_indices:
                    hook = self.teacher_model.blocks[idx].register_forward_hook(make_hook(idx))
                    hooks.append(hook)

                _ = self.teacher_model(teacher_input)

                for hook in hooks:
                    hook.remove()

                for idx in layer_indices:
                    if self.use_cls_only:
                        intermediates[idx] = captured[idx]
                    else:
                        intermediates[idx] = self._interpolate_tokens(captured[idx], self.student_num_tokens)

                last_idx = max(layer_indices)
                patch_tokens = intermediates[last_idx]

        return intermediates, patch_tokens

    def _interpolate_tokens(self, tokens, target_num_tokens):
        """Interpolate teacher tokens to match student token count."""
        B, N, D = tokens.shape
        if N == target_num_tokens:
            return tokens

        H_t = int(N ** 0.5)
        H_s = int(target_num_tokens ** 0.5)

        tokens = tokens.transpose(1, 2).reshape(B, D, H_t, H_t)
        tokens = F.interpolate(tokens, size=(H_s, H_s), mode='bilinear', align_corners=False)
        tokens = tokens.reshape(B, D, -1).transpose(1, 2)

        return tokens

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch with self-supervised distillation."""
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        self.teacher_model.eval()

        total_loss = 0
        total_ce_loss = 0
        total_tok_loss = 0
        total_rel_loss = 0
        total_cka_loss = 0
        total_gram_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        in_stage_a = self.current_epoch < self.ss_config.rel_warmup_epochs

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
            if self.use_dual_augment:
                clean_imgs, student_imgs, targets = batch
                clean_imgs = clean_imgs.to(self.device)
                student_imgs = student_imgs.to(self.device)
                targets = targets.to(self.device)

                if self.mixup_alpha > 0 or self.cutmix_alpha > 0:
                    student_imgs, targets = apply_batch_mixing(
                        student_imgs, targets, self.num_classes,
                        self.mixup_alpha, self.cutmix_alpha,
                    )

                teacher_input = clean_imgs
                student_input = student_imgs
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                teacher_input = inputs
                student_input = inputs

            with autocast(device_type='cuda', dtype=self.autocast_dtype):
                student_module = self.ddp_model.module
                student_results = student_module.forward_with_intermediates(
                    student_input, layer_indices=self.token_layers
                )
                student_output = student_results['output']
                student_intermediates = student_results['intermediates']
                student_patch_tokens = student_results['patch_tokens']

                teacher_intermediates, teacher_patch_tokens = self.get_teacher_intermediates(
                    teacher_input, self.token_layers
                )

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
            total_cka_loss += loss_dict['cka_loss']
            total_gram_loss += loss_dict['gram_loss']

            # Accuracy
            cls_output = student_output[0]
            _, predicted = cls_output.max(1)

            if len(targets.shape) > 1:
                targets_idx = targets.argmax(1)
            else:
                targets_idx = targets

            correct += predicted.eq(targets_idx).sum().item()
            total += targets_idx.size(0)
            batch_count += 1

            if self.is_main_process:
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

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        batch_count_total = int(metrics_tensor[8].item())
        avg_loss = metrics_tensor[0].item() / batch_count_total
        avg_ce_loss = metrics_tensor[1].item() / batch_count_total
        avg_tok_loss = metrics_tensor[2].item() / batch_count_total
        avg_rel_loss = metrics_tensor[3].item() / batch_count_total
        avg_cka_loss = metrics_tensor[4].item() / batch_count_total
        avg_gram_loss = metrics_tensor[5].item() / batch_count_total
        avg_acc = 100. * metrics_tensor[6].item() / metrics_tensor[7].item()

        return {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_tok_loss': avg_tok_loss,
            'train_rel_loss': avg_rel_loss,
            'train_cka_loss': avg_cka_loss,
            'train_gram_loss': avg_gram_loss,
            'train_acc': avg_acc,
            'effective_lambda_rel': loss_dict['effective_lambda_rel'],
            'effective_lambda_cka': loss_dict['effective_lambda_cka']
        }

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """Full self-supervised distillation training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            print(f"start=ss_distill epochs={num_epochs} gpus={self.world_size} batch_size={self.config.data.batch_size * self.world_size} "
                  f"stage_a=0-{self.ss_config.rel_warmup_epochs-1} stage_b={self.ss_config.rel_warmup_epochs}+")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)
            val_metrics = self.validate_ddp(val_loader)

            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.is_main_process:
                stage = "A" if epoch < self.ss_config.rel_warmup_epochs else "B"
                rel_str = f" rel={train_metrics['train_rel_loss']:.4f}" if train_metrics['train_rel_loss'] > 0 else ""
                cka_str = f" cka={train_metrics['train_cka_loss']:.4f}" if self.use_cka and train_metrics['train_cka_loss'] > 0 else ""
                gram_str = f" gram={train_metrics['train_gram_loss']:.4f}" if self.use_gram and train_metrics['train_gram_loss'] > 0 else ""

                print(f"epoch={epoch + 1}/{num_epochs} stage={stage} loss={train_metrics['train_loss']:.4f} "
                      f"ce={train_metrics['train_ce_loss']:.4f} tok={train_metrics['train_tok_loss']:.4f}{rel_str}{cka_str}{gram_str} "
                      f"train_acc={train_metrics['train_acc']:.2f} val_acc={val_metrics['val_acc']:.2f} "
                      f"lr={current_lr:.6f} time={epoch_time:.1f}s")

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"early_stop epoch={epoch + 1}")
                        break

            dist.barrier()

        if self.use_swa and self.is_main_process:
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        if self.is_main_process:
            print(f"done best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)
