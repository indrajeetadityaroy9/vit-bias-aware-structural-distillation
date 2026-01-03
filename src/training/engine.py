"""
Training engines for standard and distributed training.

Provides:
- Trainer: Single-GPU training loop
- DDPTrainer: Multi-GPU DDP training loop
"""

import time
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from src.config import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig

from .components import EarlyStopping, LabelSmoothingCrossEntropy
from .optimizers import build_optimizer, build_scheduler, HAS_PYTORCH_2, HAS_PYTORCH_2_1
from .checkpointing import build_checkpoint_dict, restore_rng_state

logger = logging.getLogger(__name__)


class Trainer:
    """Single-GPU trainer with H100 optimizations."""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        # Enable TF32 for H100 (and other Ampere+ GPUs)
        # This provides ~3x speedup on matrix multiplications
        if config.training.use_tf32 and device.type == 'cuda':
            torch.set_float32_matmul_precision('high')  # 'high' for speed, 'highest' for precision
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled: matmul precision='high', cuDNN TF32=True")

        if config.training.label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(config.training.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Use module-level utilities for optimizer and scheduler
        self.optimizer = build_optimizer(model, config, device)
        self.scheduler = build_scheduler(self.optimizer, config)

        self.use_amp = config.training.use_amp and device.type == 'cuda'
        self.autocast_kwargs = {}
        self.scaler = None
        if self.use_amp:
            # Use BF16 for better numerical stability on H100
            if config.training.use_bf16:
                self.autocast_kwargs = {'device_type': 'cuda', 'dtype': torch.bfloat16}
                logger.info("Using BF16 mixed precision (H100 optimized) - GradScaler disabled")
            else:
                self.scaler = GradScaler()
                self.autocast_kwargs = {'device_type': 'cuda'}
                logger.info("Using FP16 mixed precision with GradScaler")

        self.use_swa = config.training.use_swa
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=config.training.swa_lr
            )
            self.swa_start_epoch = int(config.training.swa_start_epoch * config.training.num_epochs)

        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta
            )
        else:
            self.early_stopping = None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        self.grad_accum_steps = config.training.gradient_accumulation_steps

        self.metrics_history = defaultdict(list)

        # torch.compile support
        self.use_compile = config.training.use_compile
        self.compile_mode = config.training.compile_mode

    def compile_model(self):
        """Apply torch.compile optimization to the model."""
        if self.use_compile:
            if not HAS_PYTORCH_2:
                logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
                return False
            if self.compile_mode == 'max-autotune' and not HAS_PYTORCH_2_1:
                logger.warning("compile_mode='max-autotune' requires PyTorch 2.1+. Using 'default' mode.")
                self.compile_mode = 'default'
            logger.info(f"Compiling model with mode='{self.compile_mode}' (H100 optimized)")
            self.model = torch.compile(self.model, mode=self.compile_mode)
            return True
        return False

    def save_checkpoint(self, filename, epoch, metrics):
        """Save checkpoint using shared utility."""
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        swa_model = self.swa_model if self.use_swa else None
        scaler = self.scaler if self.use_amp else None

        checkpoint = build_checkpoint_dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=scaler,
            swa_model=swa_model,
            epoch=epoch,
            metrics=metrics,
            config=self.config,
            best_val_acc=self.best_val_acc,
            metrics_history=self.metrics_history
        )

        save_path = checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([
                Config,
                DataConfig,
                ModelConfig,
                TrainingConfig,
                LoggingConfig
            ])
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.use_swa and 'swa_model_state_dict' in checkpoint:
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))

        logger.info(f"Checkpoint loaded from {checkpoint_path}")


class DDPTrainer(Trainer):
    """Multi-GPU DDP trainer with H100 optimizations."""

    def __init__(self, model, config, device, rank, world_size):
        super().__init__(model.module if hasattr(model, 'module') else model, config, device)

        # Store DDP-wrapped model for training
        self.ddp_model = model
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        import torch.distributed as dist
        self.dist = dist

    def compile_model(self):
        """Apply torch.compile optimization to the DDP model."""
        if self.use_compile:
            if not HAS_PYTORCH_2:
                if self.is_main_process:
                    logger.warning("torch.compile requires PyTorch 2.0+. Skipping compilation.")
                return False
            if self.compile_mode == 'max-autotune' and not HAS_PYTORCH_2_1:
                if self.is_main_process:
                    logger.warning("compile_mode='max-autotune' requires PyTorch 2.1+. Using 'default' mode.")
                self.compile_mode = 'default'
            if self.is_main_process:
                logger.info(f"Compiling DDP model with mode='{self.compile_mode}' (H100 optimized)")
            self.ddp_model = torch.compile(self.ddp_model, mode=self.compile_mode)
            return True
        return False

    def train_epoch_ddp(self, train_loader, train_sampler):
        """Train one epoch with DDP."""
        train_sampler.set_epoch(self.current_epoch)

        self.ddp_model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        if self.is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = train_loader

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast(**self.autocast_kwargs):
                    outputs = self.ddp_model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.scaler is not None:
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
                outputs = self.ddp_model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
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
                pbar.set_postfix({
                    'Loss': f'{total_loss/batch_count:.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

            self.global_step += 1

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / (batch_count * self.world_size)
        avg_acc = 100. * correct_tensor.item() / total_tensor.item()

        metrics = {
            'train_loss': avg_loss,
            'train_acc': avg_acc
        }

        return metrics

    @torch.no_grad()
    def validate_ddp(self, val_loader):
        """Validate with DDP."""
        self.ddp_model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.ddp_model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)

            if len(targets.shape) > 1:
                targets = targets.argmax(1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        # Aggregate metrics across all GPUs
        loss_tensor = torch.tensor([total_loss], device=self.device)
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        self.dist.all_reduce(loss_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(correct_tensor, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(total_tensor, op=self.dist.ReduceOp.SUM)

        avg_loss = loss_tensor.item() / total_tensor.item()
        avg_acc = 100. * correct_tensor.item() / total_tensor.item()

        metrics = {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }

        return metrics

    def train_ddp(self, train_loader, train_sampler, val_loader, num_epochs=None):
        """Full DDP training loop."""
        num_epochs = num_epochs or self.config.training.num_epochs

        if self.is_main_process:
            logger.info(f"Starting DDP training for {num_epochs} epochs")
            logger.info(f"World Size: {self.world_size}")
            logger.info(f"Effective Batch Size: {self.config.data.batch_size * self.world_size}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Warmup phase
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
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            # Only rank 0 logs and saves
            if self.is_main_process:
                logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_acc']:.2f}%, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_acc']:.2f}%, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                for key, value in {**train_metrics, **val_metrics}.items():
                    self.metrics_history[key].append(value)

                # Save checkpoint if best
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
            logger.info("DDP training completed")

        return dict(self.metrics_history)


__all__ = ['Trainer', 'DDPTrainer']
