#!/bin/bash
# reproduce_table1.sh - Train all baselines for accuracy table
#
# This script reproduces Table 1: Comparison of Knowledge Distillation Methods
# Expected results on CIFAR-10:
#   - DeiT Baseline (no distill): ~86%
#   - EXP-1 (ResNet-18 teacher): ~90%
#   - EXP-3 (DINOv2 + CKA): ~89%
#
# Prerequisites:
#   - Trained ResNet-18 teacher at outputs/resnet18_cifar/checkpoints/best_model.pth
#
# Usage:
#   ./scripts/reproduce_table1.sh [--num-gpus N]

set -e

NUM_GPUS=${1:-2}
if [[ "$1" == "--num-gpus" ]]; then
    NUM_GPUS=$2
fi

echo "============================================================"
echo "Reproducing Table 1: Knowledge Distillation Methods Comparison"
echo "Using $NUM_GPUS GPU(s)"
echo "============================================================"

# Step 1: Train ResNet-18 teacher (if not already trained)
RESNET_CKPT="outputs/resnet18_cifar/checkpoints/best_model.pth"
if [ ! -f "$RESNET_CKPT" ]; then
    echo ""
    echo "[Step 1/4] Training ResNet-18 teacher..."
    python main.py train configs/cifar10/baselines/resnet18.yaml --num-gpus $NUM_GPUS
else
    echo ""
    echo "[Step 1/4] ResNet-18 teacher already exists, skipping..."
fi

# Step 2: Train DeiT baseline (no distillation)
echo ""
echo "[Step 2/4] Training DeiT baseline (no distillation)..."
python main.py train configs/cifar10/baselines/deit_tiny.yaml --num-gpus $NUM_GPUS

# Step 3: Train DeiT with ResNet-18 distillation (EXP-1)
echo ""
echo "[Step 3/4] Training DeiT with ResNet-18 distillation (EXP-1)..."
python main.py train-distill configs/cifar10/distillation/exp1_resnet_teacher.yaml --num-gpus $NUM_GPUS

# Step 4: Train DeiT with DINOv2 CKA distillation (EXP-3)
echo ""
echo "[Step 4/4] Training DeiT with DINOv2 CKA distillation (EXP-3)..."
python main.py train-ss-distill configs/cifar10/distillation/exp3_dino_teacher.yaml --num-gpus $NUM_GPUS

echo ""
echo "============================================================"
echo "Table 1 reproduction complete!"
echo "Check outputs/ for results"
echo "============================================================"
