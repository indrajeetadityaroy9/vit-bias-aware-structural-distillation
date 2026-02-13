"""
CLI entry point for BASD training.

Usage:
    python -m src config.yaml
    torchrun --nproc_per_node=N -m src config.yaml
"""

import argparse
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.schema import load_config, save_config
from src.data import get_dataset_info, create_dataloaders
from src.evaluation.metrics import evaluate_model
from src.models.deit import DeiT
from src.models.teachers import load_dino_teacher
from src.training.distillation import BASDTrainer


def main():
    parser = argparse.ArgumentParser(description='BASD Training CLI')
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )

    config = load_config(args.config)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = Path(config.output_dir) / config.experiment_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"mode=basd config={args.config} "
            f"gpus={world_size}x{torch.cuda.get_device_name(0)}"
        )

    dist.barrier()

    dataset_info = get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']

    student = DeiT({**config.model.__dict__, **config.vit.__dict__}).to(device)
    student = DDP(student, device_ids=[local_rank])
    student = torch.compile(student, mode='max-autotune')

    teacher, teacher_dim = load_dino_teacher(config.basd.teacher_model_name, device)
    train_loader, train_sampler, val_loader, test_loader = create_dataloaders(
        config, world_size, rank, True
    )

    trainer = BASDTrainer(student, teacher, teacher_dim, config, device, rank, world_size)
    trainer.train_ddp(train_loader, train_sampler, val_loader)

    if rank == 0:
        results = evaluate_model(trainer.ddp_model.module, test_loader, trainer.device)
        metrics = results['metrics']
        print(
            f"eval accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision_macro']:.4f} "
            f"recall={metrics['recall_macro']:.4f} "
            f"f1={metrics['f1_macro']:.4f} "
            f"loss={metrics['loss']:.4f}"
        )

        save_config(config, output_dir / 'config.yaml')
        with open(output_dir / 'results.txt', 'w') as f:
            f.write(f"BASD\n{'='*60}\n")
            f.write(f"GPUs: {world_size}, Batch: {config.data.batch_size * world_size}\n")
            f.write(f"Teacher: {config.basd.teacher_model_name}\n")
            f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 (macro): {metrics['f1_macro']:.4f}\n")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
