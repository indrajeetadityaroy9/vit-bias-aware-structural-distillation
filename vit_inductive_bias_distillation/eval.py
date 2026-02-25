from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator

from vit_inductive_bias_distillation.config import load_config, setup_torch_backends
from vit_inductive_bias_distillation.evaluation.metrics import run_eval_suite, save_metrics
from vit_inductive_bias_distillation.models.deit import DeiT


def main() -> None:
    parser = argparse.ArgumentParser(description="MCSD Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    setup_torch_backends()
    config = load_config(args.config)

    accelerator = Accelerator()

    model = DeiT(config.model.vit, config.model).to(accelerator.device)

    ckpt = torch.load(config.checkpoint.path, map_location=accelerator.device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"event=checkpoint_loaded path={config.checkpoint.path} epoch={ckpt['epoch']}")

    results = run_eval_suite(
        model, config, accelerator.device,
        config_path=args.config,
    )

    output_dir = Path(config.run.output_dir) / config.run.name
    metrics_path = save_metrics(results, output_dir, config)
    print(f"event=metrics_saved path={metrics_path}")


if __name__ == "__main__":
    main()
