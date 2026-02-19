from __future__ import annotations

from typing import Any

BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "imagenet1k": {
        "DINOv2-ViT-S/14 (teacher)": {"top1": 81.1, "params_m": 22.0},
        "DeiT-Tiny (no distill)": {"top1": 72.2, "params_m": 5.0},
        "DeiT-Small (no distill)": {"top1": 79.8, "params_m": 22.0},
        "DeiT-Tiny (hard distill, RegNet teacher)": {"top1": 74.5, "params_m": 5.0},
        "DeiT-Small (hard distill, RegNet teacher)": {"top1": 81.2, "params_m": 22.0},
        "DKD (R34→R18)": {"top1": 71.70},
        "ReviewKD (R34→R18)": {"top1": 71.61},
    },
    "cifar100": {
        "DKD (WRN-40-2→WRN-16-2)": {"top1": 76.24},
        "DKD (R32x4→R8x4)": {"top1": 76.32},
        "ReviewKD (WRN-40-2→WRN-16-2)": {"top1": 76.12},
        "ReviewKD (R32x4→R8x4)": {"top1": 75.63},
    },
}


def format_baseline_table(dataset_name: str, our_results: dict[str, Any]) -> str:
    baselines = BASELINES[dataset_name]

    lines = [
        f"\n{'=' * 60}",
        f"Baseline Comparison — {dataset_name}",
        f"{'=' * 60}",
        f"{'Method':<45} {'Top-1':>7} {'Params':>8}",
        f"{'-' * 60}",
    ]

    for name, vals in baselines.items():
        top1 = f"{vals['top1']:.2f}"
        params = f"{vals['params_m']:.1f}M" if "params_m" in vals else "—"
        lines.append(f"{name:<45} {top1:>7} {params:>8}")

    lines.append(f"{'-' * 60}")
    lines.append(f"{'BASD (ours)':<45} {our_results['val_acc']:>7.2f} {our_results['param_count_m']:.1f}M")
    lines.append(f"{'=' * 60}\n")

    return "\n".join(lines)
