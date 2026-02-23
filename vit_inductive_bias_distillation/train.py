from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator

from vit_inductive_bias_distillation.config import load_config, setup_torch_backends
from vit_inductive_bias_distillation.data.datasets import create_dataloaders
from vit_inductive_bias_distillation.evaluation.metrics import run_eval_suite, save_metrics
from vit_inductive_bias_distillation.models.deit import DeiT
from vit_inductive_bias_distillation.models.teacher import load_teacher
from vit_inductive_bias_distillation.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="MCSD Training")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    setup_torch_backends()
    config = load_config(args.config)
    torch.manual_seed(config.run.seed)

    accelerator = Accelerator(mixed_precision={"bfloat16": "bf16", "float16": "fp16"}[config.training.autocast_dtype])

    output_dir = Path(config.run.output_dir) / config.run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = config.training.mode
    print(f"event=train_start config_path={args.config} device={accelerator.device} mode={mode}")

    student = DeiT(config.model.vit, config.model).to(accelerator.device)

    if mode == "baseline":
        train_loader, val_loader = create_dataloaders(config, view_mode="single")
        trainer = Trainer(student, config, accelerator=accelerator)
    else:
        teacher = load_teacher(config.basd.teacher_model_name, accelerator.device, config.basd.teacher_loader)
        train_loader, val_loader = create_dataloaders(config, view_mode="dual")
        trainer = Trainer(student, config, accelerator=accelerator, teacher=teacher)

    start_epoch = 0
    if config.checkpoint.resume_from:
        start_epoch = trainer.load_checkpoint(config.checkpoint.resume_from)
        print(f"event=checkpoint_resumed path={config.checkpoint.resume_from} start_epoch={start_epoch}")

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)

    swa_model = accelerator.unwrap_model(trainer.swa_model.module)
    results = run_eval_suite(
        swa_model, config, accelerator.device,
        config_path=args.config,
    )

    metrics_path = save_metrics(results, output_dir, config)
    print(f"event=metrics_saved path={metrics_path}")


if __name__ == "__main__":
    main()
