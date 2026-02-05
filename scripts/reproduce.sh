#!/usr/bin/env bash
# Reproduce all paper results.
# Usage: bash scripts/reproduce.sh [NUM_GPUS]
set -euo pipefail

NGPUS="${1:-1}"
RUN="torchrun --nproc_per_node=$NGPUS -m src"

echo "=== Baselines ==="
$RUN train experiments/baselines/adaptive_cnn_cifar.yaml
$RUN train experiments/baselines/resnet18_cifar.yaml
$RUN train experiments/baselines/convnext_v2_cifar.yaml
$RUN train experiments/baselines/adaptive_cnn_mnist.yaml

echo "=== CNN → ViT Distillation ==="
$RUN train-distill experiments/baselines/deit_distill_cifar.yaml

echo "=== Main: CST-CKA Self-Supervised Distillation ==="
$RUN train-ss-distill experiments/main/cst_cka_cifar.yaml

echo "=== Ablations ==="
$RUN train-ss-distill experiments/ablations/gram_matrix.yaml
$RUN train-ss-distill experiments/ablations/ce_only.yaml

echo "=== All experiments complete ==="
