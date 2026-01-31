import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.core import (
    load_config, save_config, Config,
    is_torchrun, init_distributed, cleanup_distributed, set_seed,
)
from src.modeling import create_model
from src.datasets import get_dataset, get_dataset_info, create_data_loaders, get_dual_augment_dataset
from src.evaluation import evaluate_model as run_eval, print_evaluation_summary
from src.training import DDPTrainer
from src.distillation import DistillationTrainer, SelfSupervisedDistillationTrainer, load_dino_teacher
from src.analytics import run_analytics


def create_dataloaders(config, world_size, rank, use_dual_augment=False):
    """Create distributed dataloaders."""
    train_dataset = get_dual_augment_dataset(config) if use_dual_augment else get_dataset(config, is_train=True)
    val_dataset = get_dataset(config, is_train=False)
    test_dataset = get_dataset(config, is_train=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    def worker_init(worker_id):
        np.random.seed(config.seed + rank * 1000 + worker_id)
        random.seed(config.seed + rank * 1000 + worker_id)

    kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=4 if config.data.num_workers > 0 else None,
        worker_init_fn=worker_init if config.data.num_workers > 0 else None,
    )

    return (
        DataLoader(train_dataset, sampler=train_sampler, **kwargs),
        train_sampler,
        DataLoader(val_dataset, sampler=val_sampler, **kwargs),
        DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers, pin_memory=True),
    )


def evaluate_and_save(trainer, test_loader, dataset_info, config, output_dir, world_size, mode_name, extra_info=None):
    """Evaluate and save results."""
    model = trainer.ddp_model.module if hasattr(trainer.ddp_model, 'module') else trainer.ddp_model
    results = run_eval(model, test_loader, trainer.device, class_names=dataset_info.get('classes'))
    print_evaluation_summary(results, class_names=dataset_info.get('classes'))

    save_config(config, output_dir / 'config.yaml')

    metrics = results['metrics']
    with open(output_dir / 'results.txt', 'w') as f:
        f.write(f"{mode_name}\n{'='*60}\n")
        f.write(f"GPUs: {world_size}, Batch: {config.data.batch_size * world_size}\n")
        if extra_info:
            for k, v in extra_info.items():
                f.write(f"{k}: {v}\n")
        f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 (macro): {metrics['f1_macro']:.4f}\n")

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    return {'accuracy': metrics['accuracy'], 'output_dir': str(output_dir)}


def train_worker(rank, world_size, config_path, mode):
    """Distributed training worker."""
    rank, world_size, device = init_distributed(rank, world_size)
    is_main = rank == 0

    try:
        config = load_config(config_path)
        set_seed(config.seed + rank)

        # Setup directories
        output_dir = Path(config.output_dir) / config.experiment_name
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'checkpoints').mkdir(exist_ok=True)
            print(f"mode={mode} config={config_path} gpus={world_size}x{torch.cuda.get_device_name(0)}")

        dist.barrier()

        # Dataset info
        dataset_info = get_dataset_info(config)
        config.model.in_channels = dataset_info['in_channels']
        config.model.num_classes = dataset_info['num_classes']

        # Model config
        model_config = config.model.__dict__.copy()
        if hasattr(config, 'vit') and config.vit:
            model_config.update(config.vit.__dict__ if hasattr(config.vit, '__dict__') else config.vit)

        if mode == 'standard':
            model = create_model(config.model.model_type, model_config).to(device)
            model = DDP(model, device_ids=[rank])
            model = torch.compile(model, mode='max-autotune')

            train_loader, train_sampler, val_loader, test_loader = create_dataloaders(config, world_size, rank)
            trainer = DDPTrainer(model, config, device, rank, world_size)
            trainer.train_ddp(train_loader, train_sampler, val_loader)

            if is_main:
                evaluate_and_save(trainer, test_loader, dataset_info, config, output_dir, world_size, 'Standard Training')

        elif mode == 'distill':
            student = create_model('deit', model_config).to(device)

            # Load teacher
            teacher_path = Path(config.distillation.teacher_checkpoint)
            teacher_config = config.model.__dict__.copy()
            teacher = create_model(config.distillation.teacher_model_type, teacher_config).to(device)
            ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
            teacher.load_state_dict(ckpt['model_state_dict'])
            teacher.eval()

            student = DDP(student, device_ids=[rank], find_unused_parameters=True)
            student = torch.compile(student, mode='max-autotune')

            train_loader, train_sampler, val_loader, test_loader = create_dataloaders(config, world_size, rank)
            trainer = DistillationTrainer(student, teacher, config, device, rank, world_size)
            trainer.train_ddp(train_loader, train_sampler, val_loader)

            if is_main:
                evaluate_and_save(trainer, test_loader, dataset_info, config, output_dir, world_size, 'Distillation',
                                  {'Teacher': config.distillation.teacher_model_type})

        elif mode == 'ss_distill':
            student = create_model('deit', model_config).to(device)
            teacher, teacher_dim = load_dino_teacher(
                config.ss_distillation.teacher_type,
                config.ss_distillation.teacher_model_name,
                device
            )

            student = DDP(student, device_ids=[rank], find_unused_parameters=True)
            student = torch.compile(student, mode='max-autotune')

            use_dual = getattr(config.ss_distillation, 'use_dual_augment', False)
            train_loader, train_sampler, val_loader, test_loader = create_dataloaders(config, world_size, rank, use_dual)
            trainer = SelfSupervisedDistillationTrainer(student, teacher, teacher_dim, config, device, rank, world_size)
            trainer.train_ddp(train_loader, train_sampler, val_loader)

            if is_main:
                evaluate_and_save(trainer, test_loader, dataset_info, config, output_dir, world_size, 'SS Distillation',
                                  {'Teacher': config.ss_distillation.teacher_model_name})

    finally:
        cleanup_distributed()


def train(config_path, mode='standard'):
    """Entry point for training."""
    if is_torchrun():
        train_worker(None, None, config_path, mode)
    else:
        # Single GPU fallback
        train_worker(0, 1, config_path, mode)


def evaluate(config_path, checkpoint_path):
    """Evaluate a model."""
    device = torch.device('cuda')
    config = load_config(config_path)
    dataset_info = get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']

    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit:
        model_config.update(config.vit.__dict__ if hasattr(config.vit, '__dict__') else config.vit)

    model = create_model(config.model.model_type, model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, _, test_loader = create_data_loaders(config)
    results = run_eval(model, test_loader, device, class_names=dataset_info.get('classes'))
    print_evaluation_summary(results, class_names=dataset_info.get('classes'))


def analyze(config_path, checkpoint_path, metrics='all', output_dir=None):
    """Run analytics."""
    device = torch.device('cuda')
    config = load_config(config_path)
    dataset_info = get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']

    model_config = config.model.__dict__.copy()
    if hasattr(config, 'vit') and config.vit:
        model_config.update(config.vit.__dict__ if hasattr(config.vit, '__dict__') else config.vit)

    model = create_model(config.model.model_type, model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_loader = DataLoader(get_dataset(config, is_train=False), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    out_path = Path(output_dir or config.output_dir) / 'analytics'
    out_path.mkdir(parents=True, exist_ok=True)

    metric_list = ['hessian', 'attention', 'cka'] if metrics == 'all' else [m.strip() for m in metrics.split(',')]

    class Cfg:
        hessian_samples = 1024
        cka_kernel = 'linear'
        vit = getattr(config, 'vit', None)

    results = run_analytics(model, Cfg(), device, test_loader, metric_list, out_path / 'results.json')
    print(f"analytics saved={out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='H100 Training CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Train
    train_p = subparsers.add_parser('train')
    train_p.add_argument('config', type=str)

    # Distill
    distill_p = subparsers.add_parser('train-distill')
    distill_p.add_argument('config', type=str)

    # SS Distill
    ss_p = subparsers.add_parser('train-ss-distill')
    ss_p.add_argument('config', type=str)

    # Evaluate
    eval_p = subparsers.add_parser('evaluate')
    eval_p.add_argument('config', type=str)
    eval_p.add_argument('checkpoint', type=str)

    # Analyze
    analyze_p = subparsers.add_parser('analyze')
    analyze_p.add_argument('config', type=str)
    analyze_p.add_argument('checkpoint', type=str)
    analyze_p.add_argument('--metrics', type=str, default='all')
    analyze_p.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'train':
        train(args.config, 'standard')
    elif args.command == 'train-distill':
        train(args.config, 'distill')
    elif args.command == 'train-ss-distill':
        train(args.config, 'ss_distill')
    elif args.command == 'evaluate':
        evaluate(args.config, args.checkpoint)
    elif args.command == 'analyze':
        analyze(args.config, args.checkpoint, args.metrics, args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
