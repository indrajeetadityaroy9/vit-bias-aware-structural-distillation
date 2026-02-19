from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vit_inductive_bias_distillation.config import load_config
from vit_inductive_bias_distillation.evaluation.suite import run_eval_suite, save_metrics
from vit_inductive_bias_distillation.models.deit import DeiT
from vit_inductive_bias_distillation.runtime_log import log_event
from vit_inductive_bias_distillation.training.setup import setup_device


def main() -> None:
    parser = argparse.ArgumentParser(description="BASD Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    device = setup_device()
    config = load_config(args.config)

    model = DeiT(config.model.vit, config.model).to(device)

    ckpt = torch.load(config.checkpoint.path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log_event("checkpoint_loaded", path=config.checkpoint.path, epoch=ckpt["epoch"])

    results = run_eval_suite(model, config, device, config_path=args.config)
    output_dir = Path(config.run.output_dir) / config.run.name
    metrics_path = save_metrics(results, output_dir, config)
    log_event("metrics_saved", path=str(metrics_path))


if __name__ == "__main__":
    main()
