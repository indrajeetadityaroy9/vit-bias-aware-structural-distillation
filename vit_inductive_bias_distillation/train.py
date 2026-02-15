"""BASD training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from vit_inductive_bias_distillation.config import load_config, save_config
from vit_inductive_bias_distillation.data import create_dataloaders
from vit_inductive_bias_distillation.evaluation.metrics import evaluate_model
from vit_inductive_bias_distillation.models import build_student_model, load_teacher
from vit_inductive_bias_distillation.training import BASDTrainer, init_distributed, seed_everything

def main() -> None:
    parser = argparse.ArgumentParser(description="BASD Training")
    parser.add_argument("config", type=str, help="Path to experiment config YAML")
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    config = load_config(args.config)
    seed_everything(config.seed, rank)

    output_dir = Path(config.output_dir) / config.experiment_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(args.config, world_size, torch.cuda.get_device_name(0))

    dist.barrier()

    student = build_student_model(config, device)
    student = DDP(student, device_ids=[device.index])
    student = torch.compile(student, mode="max-autotune")

    teacher = load_teacher(config.basd.teacher_model_name, device)

    train_loader, train_sampler, val_loader, test_loader = create_dataloaders(
        config, world_size, rank, True
    )

    total_steps = len(train_loader) * config.training.num_epochs
    trainer = BASDTrainer(
        student, teacher.model, teacher.embed_dim, config,
        device, rank, world_size, total_steps,
    )

    start_epoch = 0
    if config.resume_from:
        start_epoch = trainer.load_checkpoint(config.resume_from)
        if rank == 0:
            print(config.resume_from, start_epoch)

    trainer.train_ddp(train_loader, train_sampler, val_loader, start_epoch=start_epoch)

    if rank == 0:
        results = evaluate_model(trainer.ddp_model.module, test_loader, device, trainer.criterion)
        metrics = results["metrics"]
        print(
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["loss"],
        )

        save_config(config, output_dir / "config.yaml")
        with open(output_dir / "results.txt", "w") as f:
            f.write(f"BASD\n{'=' * 60}\n")
            f.write(f"GPUs: {world_size}, Batch: {config.data.batch_size * world_size}\n")
            f.write(f"Teacher: {config.basd.teacher_model_name}\n")
            f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 (macro): {metrics['f1_macro']:.4f}\n")
        print(str(output_dir))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
