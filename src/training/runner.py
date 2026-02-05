"""
Training orchestration: worker setup, training loops, evaluation, and analysis.

Provides:
- build_model_config: Merge model + vit configs
- evaluate_and_save: Post-training evaluation
- train_worker: DDP worker entry point
- train: Top-level training dispatcher
- evaluate: Standalone evaluation
- analyze: Standalone analytics
"""

from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.config import load_config, save_config
from src.utils import set_seed, is_torchrun, init_distributed, cleanup_distributed
from src.models import create_model
from src.models.teachers import load_dino_teacher
from src.data import get_dataset, get_dataset_info, create_dataloaders
from src.evaluation import evaluate_model as run_eval, print_evaluation_summary, run_analytics
from src.training.trainer import DDPTrainer
from src.training.distillation import DistillationTrainer, SelfSupervisedDistillationTrainer


def build_model_config(config):
    """Build model config dict by merging model and vit configs."""
    model_config = config.model.__dict__.copy()
    if config.vit:
        model_config.update(config.vit.__dict__)
    return model_config


def evaluate_and_save(trainer, test_loader, dataset_info, config, output_dir, world_size, mode_name, extra_info=None):
    """Evaluate and save results."""
    model = trainer.ddp_model.module
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
        model_config = build_model_config(config)

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

            train_loader, train_sampler, val_loader, test_loader = create_dataloaders(config, world_size, rank, config.ss_distillation.use_dual_augment)
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

    model = create_model(config.model.model_type, build_model_config(config)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_dataset = get_dataset(config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    results = run_eval(model, test_loader, device, class_names=dataset_info.get('classes'))
    print_evaluation_summary(results, class_names=dataset_info.get('classes'))


def analyze(config_path, checkpoint_path, metrics='all', output_dir=None):
    """Run analytics."""
    device = torch.device('cuda')
    config = load_config(config_path)
    dataset_info = get_dataset_info(config)
    config.model.in_channels = dataset_info['in_channels']
    config.model.num_classes = dataset_info['num_classes']

    model = create_model(config.model.model_type, build_model_config(config)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_loader = DataLoader(get_dataset(config, is_train=False), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    out_path = Path(output_dir or config.output_dir) / 'analytics'
    out_path.mkdir(parents=True, exist_ok=True)

    metric_list = ['hessian', 'attention', 'cka'] if metrics == 'all' else [m.strip() for m in metrics.split(',')]

    results = run_analytics(model, config, device, test_loader, metric_list, save_path=out_path / 'results.json')
    print(f"analytics saved={out_path}")
    return results
