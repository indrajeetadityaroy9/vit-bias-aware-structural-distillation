"""
BASD-v2 training engine.
"""

import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from src.models.projectors import CrossAttentionProjector
from src.training.losses.combined import BASDv2Loss


class BASDTrainer:
    """Trainer for BASD distillation."""

    def __init__(self, student_model, teacher_model, teacher_embed_dim, config, device, rank, world_size):
        self.ddp_model = student_model
        self.teacher_model = teacher_model
        self.teacher_embed_dim = teacher_embed_dim
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0

        self.token_layers = config.basd.token_layers
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
                    eta_min=1e-6,
                ),
            ],
            milestones=[self.warmup_epochs],
        )

        self.swa_model = AveragedModel(self.ddp_model.module)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.training.swa_lr)
        self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        self.early_stopping_patience = config.training.early_stopping_patience
        self.early_stopping_min_delta = 0.001
        self._early_stop_counter = 0
        self._early_stop_best_score = -float('inf')

        self.autocast_dtype = torch.bfloat16
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.metrics_history = defaultdict(list)

        self._loss_tensor = torch.zeros(1, device=device)
        self._correct_tensor = torch.zeros(1, device=device, dtype=torch.long)
        self._total_tensor = torch.zeros(1, device=device, dtype=torch.long)

        self.cross_attn_projectors = torch.nn.ModuleList([
            CrossAttentionProjector(
                num_student_tokens=self.student_num_tokens,
                teacher_dim=self.teacher_embed_dim,
                student_dim=self.teacher_embed_dim,
                num_heads=config.basd.cross_attn_num_heads,
            )
            for _ in self.token_layers
        ]).to(device)

        self.basd_loss = BASDv2Loss(
            base_criterion=self.criterion,
            student_dim=self.student_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            config=config.basd,
        ).to(device)

        self.optimizer.add_param_group({
            'params': list(self.cross_attn_projectors.parameters()) + list(self.basd_loss.parameters()),
            'lr': config.training.learning_rate,
        })

        self.attn_layer_pairs = [tuple(p) for p in config.basd.attn_layer_pairs]
        self.teacher_attn_layers = sorted({p[1] for p in self.attn_layer_pairs})

        if self.is_main_process:
            print(
                f"basd teacher={config.basd.teacher_model_name} "
                f"teacher_dim={self.teacher_embed_dim} student_dim={self.student_embed_dim} "
                f"layers={self.token_layers}"
            )

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / self.config.experiment_name / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.ddp_model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'swa_model_state_dict': self.swa_model.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_acc': self.best_val_acc,
            'metrics_history': dict(self.metrics_history),
            'rng_state': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all(),
                'numpy': np.random.get_state(),
                'python': random.getstate(),
            },
        }

        torch.save(checkpoint, checkpoint_dir / filename)
        if self.is_main_process:
            print(f"checkpoint={filename}")

    @torch.no_grad()
    def validate_ddp(self, val_loader) -> dict:
        """Validate the current model."""
        self.ddp_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.ddp_model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        self._loss_tensor.fill_(total_loss)
        self._correct_tensor.fill_(correct)
        self._total_tensor.fill_(total)

        dist.all_reduce(self._loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._total_tensor, op=dist.ReduceOp.SUM)

        return {
            'val_loss': self._loss_tensor.item() / self._total_tensor.item(),
            'val_acc': 100.0 * self._correct_tensor.item() / self._total_tensor.item(),
        }

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

    def get_teacher_intermediates(self, x):
        """Extract intermediate teacher tokens and attention maps in one pass."""
        with torch.no_grad():
            hooks = []
            captured_tokens = {}
            captured_attns = {}
            all_token_layers = set(self.token_layers)
            all_attn_layers = set(self.teacher_attn_layers)
            all_layers = all_token_layers | all_attn_layers

            for layer_idx in all_layers:
                block = self.teacher_model.blocks[layer_idx]

                if layer_idx in all_token_layers:
                    def make_token_hook(idx):
                        def hook(module, input, output):
                            captured_tokens[idx] = output[:, 1:, :]
                        return hook
                    hooks.append(block.register_forward_hook(make_token_hook(layer_idx)))

                if layer_idx in all_attn_layers:
                    def make_attn_hook(idx):
                        def hook(module, input, output):
                            batch_size, num_tokens, channels = input[0].shape
                            num_heads = module.num_heads
                            head_dim = channels // num_heads
                            qkv = module.qkv(input[0]).reshape(
                                batch_size, num_tokens, 3, num_heads, head_dim
                            ).permute(2, 0, 3, 1, 4)
                            q, k, _ = qkv.unbind(0)
                            attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                            captured_attns[idx] = attn.softmax(dim=-1).detach()
                        return hook
                    hooks.append(block.attn.register_forward_hook(make_attn_hook(layer_idx)))

            self.teacher_model(x)

            for hook in hooks:
                hook.remove()

            intermediates = {}
            raw_intermediates = {}
            for i, layer_idx in enumerate(self.token_layers):
                raw_intermediates[layer_idx] = captured_tokens[layer_idx]
                intermediates[layer_idx] = self.cross_attn_projectors[i](captured_tokens[layer_idx])

        return intermediates, raw_intermediates, captured_attns

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch."""
        train_sampler.set_epoch(self.current_epoch)
        self.ddp_model.train()
        self.teacher_model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_rsd_loss = 0.0
        total_gram_loss = 0.0
        total_attn_loss = 0.0
        total_spectral_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}") if self.is_main_process else train_loader

        for clean_imgs, student_imgs, targets in pbar:
            clean_imgs = clean_imgs.to(self.device, non_blocking=True)
            student_imgs = student_imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=self.autocast_dtype):
                student_results = self.ddp_model.module.forward_with_intermediates(
                    student_imgs,
                    layer_indices=self.token_layers,
                )
                student_output = student_results['output']
                student_intermediates = student_results['intermediates']
                student_attns = student_results['attention_weights']

                teacher_intermediates, raw_teacher_intermediates, teacher_attns = self.get_teacher_intermediates(clean_imgs)
                loss, loss_dict = self.basd_loss(
                    student_output,
                    targets,
                    student_intermediates,
                    teacher_intermediates,
                    raw_teacher_intermediates,
                    student_attns,
                    teacher_attns,
                    self.global_step,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.ddp_model.parameters()) +
                list(self.basd_loss.parameters()) +
                list(self.cross_attn_projectors.parameters()),
                1.0,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_ce_loss += loss_dict['ce_loss']
            total_rsd_loss += loss_dict['rsd_loss']
            total_gram_loss += loss_dict['gram_loss']
            total_attn_loss += loss_dict['attn_loss']
            total_spectral_loss += loss_dict['spectral_loss']

            predicted = student_output.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{total_loss / batch_count:.4f}',
                    'ce': f'{total_ce_loss / batch_count:.4f}',
                    'rsd': f'{total_rsd_loss / batch_count:.4f}',
                    'gram': f'{total_gram_loss / batch_count:.4f}',
                    'attn': f'{total_attn_loss / batch_count:.4f}',
                    'spec': f'{total_spectral_loss / batch_count:.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%',
                })

            self.global_step += 1

        metrics_tensor = torch.tensor([
            total_loss,
            total_ce_loss,
            total_rsd_loss,
            total_gram_loss,
            total_attn_loss,
            total_spectral_loss,
            correct,
            total,
            batch_count,
        ], device=self.device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        batch_count_total = int(metrics_tensor[8].item())
        return {
            'train_loss': metrics_tensor[0].item() / batch_count_total,
            'train_ce_loss': metrics_tensor[1].item() / batch_count_total,
            'train_rsd_loss': metrics_tensor[2].item() / batch_count_total,
            'train_gram_loss': metrics_tensor[3].item() / batch_count_total,
            'train_attn_loss': metrics_tensor[4].item() / batch_count_total,
            'train_spectral_loss': metrics_tensor[5].item() / batch_count_total,
            'train_acc': 100.0 * metrics_tensor[6].item() / metrics_tensor[7].item(),
            'total_samples': int(metrics_tensor[7].item()),
            'sigma_rsd': loss_dict['sigma_rsd'],
            'sigma_gram': loss_dict['sigma_gram'],
            'sigma_attn': loss_dict['sigma_attn'],
            'sigma_spectral': loss_dict['sigma_spectral'],
            'weight_rsd': loss_dict['weight_rsd'],
            'weight_gram': loss_dict['weight_gram'],
            'weight_attn': loss_dict['weight_attn'],
            'weight_spectral': loss_dict['weight_spectral'],
        }

    def train_ddp(self, train_loader, train_sampler, val_loader):
        """Full BASD training loop."""
        num_epochs = self.config.training.num_epochs
        self.basd_loss.set_total_steps(len(train_loader) * num_epochs)

        if self.is_main_process:
            print(
                f"start=basd epochs={num_epochs} gpus={self.world_size} "
                f"batch_size={self.config.data.batch_size * self.world_size}"
            )

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)
            val_metrics = self.validate_ddp(val_loader)
            self.step_scheduler(epoch)

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.is_main_process:
                throughput = train_metrics['total_samples'] / epoch_time
                print(
                    f"epoch={epoch + 1}/{num_epochs} "
                    f"loss={train_metrics['train_loss']:.4f} "
                    f"ce={train_metrics['train_ce_loss']:.4f} "
                    f"rsd={train_metrics['train_rsd_loss']:.4f} "
                    f"gram={train_metrics['train_gram_loss']:.4f} "
                    f"attn={train_metrics['train_attn_loss']:.4f} "
                    f"spec={train_metrics['train_spectral_loss']:.4f} "
                    f"train_acc={train_metrics['train_acc']:.2f} "
                    f"val_acc={val_metrics['val_acc']:.2f} "
                    f"lr={current_lr:.6f} "
                    f"time={epoch_time:.1f}s "
                    f"img/s={throughput:.0f}"
                )

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.should_early_stop(val_metrics['val_loss']):
                    print(f"early_stop epoch={epoch + 1}")
                    break

        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)
        if self.is_main_process:
            print(f"done best_val_acc={self.best_val_acc:.2f}")

        return dict(self.metrics_history)
