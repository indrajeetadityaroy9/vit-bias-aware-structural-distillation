from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vit_inductive_bias_distillation.config import load_config
from vit_inductive_bias_distillation.data import create_dataloaders
from vit_inductive_bias_distillation.evaluation.suite import run_eval_suite, save_metrics
from vit_inductive_bias_distillation.models.deit import DeiT
from vit_inductive_bias_distillation.models.teacher import load_teacher
from vit_inductive_bias_distillation.runtime_log import log_event
from vit_inductive_bias_distillation.training import BASDTrainer, seed_everything
from vit_inductive_bias_distillation.training.setup import setup_device


def main() -> None:
    parser = argparse.ArgumentParser(description="BASD Training")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    device = setup_device()
    config = load_config(args.config)
    seed_everything(config.run.seed)

    output_dir = Path(config.run.output_dir) / config.run.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_event(
        "train_start",
        config_path=args.config,
        device=str(device),
        cuda_name=torch.cuda.get_device_name(0),
    )

    student = DeiT(config.model.vit, config.model).to(device)
    teacher = load_teacher(config.basd.teacher_model_name, device)
    train_loader, val_loader = create_dataloaders(config)

    trainer = BASDTrainer(
        student, teacher, config,
        device, steps_per_epoch=len(train_loader),
    )

    start_epoch = 0
    if config.checkpoint.resume_from:
        start_epoch = trainer.load_checkpoint(config.checkpoint.resume_from)
        log_event(
            "checkpoint_resumed",
            path=config.checkpoint.resume_from,
            start_epoch=start_epoch,
        )

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    results = run_eval_suite(trainer.model, config, device, config_path=args.config)
    metrics_path = save_metrics(results, output_dir, config)
    log_event("metrics_saved", path=str(metrics_path))


if __name__ == "__main__":
    main()
