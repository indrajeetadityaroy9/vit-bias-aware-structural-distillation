"""
Training engines for H100 GPUs.

Uses BF16 mixed precision, fused optimizers, and TF32 by default.
"""

import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from src.core import Config

from .components import EarlyStopping, LabelSmoothingCrossEntropy
from .optimizers import build_optimizer, build_scheduler
from .checkpointing import build_checkpoint_dict


class DDPTrainer:
    """
    DDP trainer optimized for H100 GPUs.

    Uses BF16 mixed precision (no GradScaler needed).
    """

    def __init__(self, model: nn.Module, config: Config, device: torch.device, rank: int, world_size: int):
        # Get base model (handle DDP wrapping)
        base_model = model.module if hasattr(model, 'module') else model

        self.model = base_model
        self.ddp_model = model
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Loss function
        if config.training.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(config.training.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler (fused by default)
        self.optimizer = build_optimizer(base_model, config, device)
        self.scheduler = build_scheduler(self.optimizer, config)

        # BF16 autocast (H100 native, no scaler needed)
        self.autocast_dtype = torch.bfloat16

        # SWA
        self.use_swa = config.training.use_swa
        if self.use_swa:
            self.swa_model = AveragedModel(base_model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.training.swa_lr)
            self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        # Early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta
            )
        else:
            self.early_stopping = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.metrics_history = defaultdict(list)

        # Pre-allocated tensors for metric reduction
        self._loss_tensor = torch.zeros(1, device=device)
        self._correct_tensor = torch.zeros(1, device=device, dtype=torch.long)
        self._total_tensor = torch.zeros(1, device=device, dtype=torch.long)

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        """Save checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = build_checkpoint_dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=None,
            swa_model=self.swa_model if self.use_swa else None,
            epoch=epoch,
            metrics=metrics,
            config=self.config,
            best_val_acc=self.best_val_acc,
            metrics_history=self.metrics_history
        )

        torch.save(checkpoint, checkpoint_dir / filename)
        if self.is_main_process:
            print(f"checkpoint={filename}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_swa and 'swa_model_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))

    def train_epoch_ddp(self, train_loader, train_sampler) -> dict:
        """Train one epoch."""
        train_sampler.set_epoch(self.current_epoch)
        self.ddp_model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}") if self.is_main_process else train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=self.autocast_dtype):
                outputs = self.ddp_model(inputs)
                loss = self.criterion(outputs, targets) / self.grad_accum_steps

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

            total_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            if len(targets.shape) > 1:
                targets = targets.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            batch_count += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'Loss': f'{total_loss/batch_count:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

            self.global_step += 1

        # All-reduce metrics
        self._loss_tensor.fill_(total_loss)
        self._correct_tensor.fill_(correct)
        self._total_tensor.fill_(total)

        dist.all_reduce(self._loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._total_tensor, op=dist.ReduceOp.SUM)

        return {
            'train_loss': self._loss_tensor.item() / (batch_count * self.world_size),
            'train_acc': 100.0 * self._correct_tensor.item() / self._total_tensor.item()
        }

    @torch.no_grad()
    def validate_ddp(self, val_loader) -> dict:
        """Validate."""
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
            _, predicted = outputs.max(1)
            if len(targets.shape) > 1:
                targets = targets.argmax(1)
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
            'val_acc': 100.0 * self._correct_tensor.item() / self._total_tensor.item()
        }

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs: int = None) -> dict:
        """Full training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            print(f"start=train epochs={num_epochs} gpus={self.world_size} batch_size={self.config.data.batch_size * self.world_size}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Warmup
            if epoch < self.config.training.warmup_epochs:
                warmup_lr = self.config.training.learning_rate * (epoch + 1) / self.config.training.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            train_metrics = self.train_epoch_ddp(train_loader, train_sampler)
            val_metrics = self.validate_ddp(val_loader)

            # Scheduler
            if self.use_swa and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.ddp_model.module)
                self.swa_scheduler.step()
            elif self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            if self.is_main_process:
                print(f"epoch={epoch + 1}/{num_epochs} train_loss={train_metrics['train_loss']:.4f} "
                      f"train_acc={train_metrics['train_acc']:.2f} val_loss={val_metrics['val_loss']:.4f} "
                      f"val_acc={val_metrics['val_acc']:.2f} lr={self.optimizer.param_groups[0]['lr']:.6f} "
                      f"time={time.time() - epoch_start:.1f}s")

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                if val_metrics['val_acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint('best_model.pth', epoch, val_metrics)

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_metrics)

                if self.early_stopping and self.early_stopping(val_metrics['val_loss']):
                    print(f"early_stop epoch={epoch + 1}")
                    break

            dist.barrier()

        if self.use_swa and self.is_main_process:
            torch.optim.swa_utils.update_bn(train_loader, self.swa_model, self.device)

        return dict(self.metrics_history)


__all__ = ['DDPTrainer']
