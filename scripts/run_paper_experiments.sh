#!/bin/bash
# =============================================================================
# Locality Curse Paper Experiments - Full Reproducibility Script
# =============================================================================
#
# This script reproduces all experiments for the paper:
# "The Locality Curse: CNN Teachers Damage ViT Students Through Inductive Bias Mismatch"
#
# Usage:
#   ./scripts/run_paper_experiments.sh [--gpus N] [--seed S]
#
# Requirements:
#   - Python 3.8+ with dependencies installed
#   - CUDA-capable GPU (tested on H100)
#   - ~50GB disk space for checkpoints
#
# Estimated runtime: ~8-12 hours on single H100
# =============================================================================

set -e  # Exit on error

# Configuration
GPUS=${1:-1}
SEED=${2:-42}
OUTPUT_DIR="./outputs/paper_experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Locality Curse Paper Experiments"
echo "=============================================="
echo "GPUs: $GPUS"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=============================================="

mkdir -p "$OUTPUT_DIR/logs"

# =============================================================================
# PHASE 0: Prerequisites - Train Teacher Models
# =============================================================================

echo ""
echo "[PHASE 0] Training Teacher Models..."
echo "=============================================="

# 0.1 Train ResNet-18 Teacher (Classic CNN)
echo "[0.1] Training ResNet-18 Teacher..."
python main.py train configs/resnet18_cifar_config.yaml \
    --num-gpus $GPUS \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/logs/phase0_resnet18_teacher.log"

RESNET_CKPT="./outputs/resnet18_cifar/best_model.pth"
if [ ! -f "$RESNET_CKPT" ]; then
    echo "ERROR: ResNet-18 checkpoint not found at $RESNET_CKPT"
    exit 1
fi
echo "  ResNet-18 Teacher trained: $RESNET_CKPT"

# =============================================================================
# PHASE 1: Train Student Models (Three-Way Comparison)
# =============================================================================

echo ""
echo "[PHASE 1] Training Student Models..."
echo "=============================================="

# 1.1 Baseline DeiT (No Distillation)
echo "[1.1] Training Baseline DeiT (no distillation)..."
python main.py train configs/deit_cifar_config.yaml \
    --num-gpus $GPUS \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/logs/phase1_baseline_deit.log"

BASELINE_CKPT="./outputs/deit_cifar/best_model.pth"
echo "  Baseline DeiT trained: $BASELINE_CKPT"

# 1.2 CNN-Distilled DeiT (ResNet-18 Teacher) - THE CURSE
echo "[1.2] Training CNN-Distilled DeiT (ResNet-18 teacher)..."
python main.py train-distill configs/deit_resnet18_distill_config.yaml \
    --num-gpus $GPUS \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/logs/phase1_cnn_distilled.log"

CNN_DISTILLED_CKPT="./outputs/deit_resnet18_distill/best_model.pth"
echo "  CNN-Distilled DeiT trained: $CNN_DISTILLED_CKPT"

# 1.3 DINO-Distilled DeiT (DINOv2 Teacher) - THE CURE
echo "[1.3] Training DINO-Distilled DeiT (DINOv2 teacher)..."
python main.py train-ss-distill configs/deit_ss_distill_cka_cifar_config.yaml \
    --num-gpus $GPUS \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/logs/phase1_dino_distilled.log"

DINO_DISTILLED_CKPT="./outputs/deit_dinov2_cka/best_model.pth"
echo "  DINO-Distilled DeiT trained: $DINO_DISTILLED_CKPT"

# =============================================================================
# PHASE 2: Locality Curse Forensics
# =============================================================================

echo ""
echo "[PHASE 2] Running Locality Curse Forensics..."
echo "=============================================="

FORENSICS_DIR="$OUTPUT_DIR/forensics_$TIMESTAMP"
mkdir -p "$FORENSICS_DIR"

python main.py analyze-locality configs/deit_cifar_config.yaml \
    "$CNN_DISTILLED_CKPT" \
    "$BASELINE_CKPT" \
    "$DINO_DISTILLED_CKPT" \
    -n "CNN-Distilled" \
    -n "Baseline" \
    -n "DINO-Distilled" \
    -o "$FORENSICS_DIR" \
    --num-samples 1000 \
    2>&1 | tee "$OUTPUT_DIR/logs/phase2_forensics.log"

echo "  Forensics complete: $FORENSICS_DIR"

# =============================================================================
# PHASE 3: Generate Paper Figures
# =============================================================================

echo ""
echo "[PHASE 3] Generating Paper Figures..."
echo "=============================================="

# Copy key figures to paper directory
FIGURES_DIR="$OUTPUT_DIR/figures"
mkdir -p "$FIGURES_DIR"

cp "$FORENSICS_DIR/locality_spectrum.pdf" "$FIGURES_DIR/fig1_locality_spectrum.pdf" 2>/dev/null || true
cp "$FORENSICS_DIR/layer_progression.pdf" "$FIGURES_DIR/fig2_layer_progression.pdf" 2>/dev/null || true
cp "$FORENSICS_DIR/forensics_summary.pdf" "$FIGURES_DIR/fig3_forensics_summary.pdf" 2>/dev/null || true

echo "  Figures saved to: $FIGURES_DIR"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Model Checkpoints:"
echo "  - ResNet-18 Teacher: $RESNET_CKPT"
echo "  - Baseline DeiT: $BASELINE_CKPT"
echo "  - CNN-Distilled DeiT: $CNN_DISTILLED_CKPT"
echo "  - DINO-Distilled DeiT: $DINO_DISTILLED_CKPT"
echo ""
echo "Forensics Results:"
echo "  - JSON: $FORENSICS_DIR/*_forensics.json"
echo "  - Plots: $FORENSICS_DIR/*.pdf"
echo ""
echo "Paper Figures:"
echo "  - $FIGURES_DIR/"
echo ""
echo "Expected Results (Three-Way Comparison):"
echo "  Metric                | CNN-Distilled | Baseline | DINO-Distilled"
echo "  ---------------------|---------------|----------|---------------"
echo "  Test Accuracy        | ~88%          | ~85%     | >89%"
echo "  Hessian Trace        | High (sharp)  | Medium   | Low (flat)"
echo "  Avg Attention Dist   | Low (local)   | Medium   | High (global)"
echo "  Collapsed Heads %    | High          | Medium   | Low"
echo ""
echo "To verify the Locality Curse hypothesis:"
echo "  1. Check: CNN-Distilled has LOWER MAD than Baseline (active harm)"
echo "  2. Check: CNN-Distilled has HIGHER Hessian trace (sharper minima)"
echo "  3. Check: DINO-Distilled has HIGHER MAD and accuracy"
echo "=============================================="
