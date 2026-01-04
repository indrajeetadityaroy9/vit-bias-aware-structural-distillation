#!/bin/bash
# reproduce_figure3.sh - Run Locality Curse forensics for Figure 3
#
# This script reproduces Figure 3: Locality Curse Visualization
# Shows three-way comparison of attention patterns:
#   - CNN-Distilled (local attention, "Locality Curse")
#   - Baseline (medium range)
#   - DINO-Distilled (global attention)
#
# Prerequisites:
#   - All models from reproduce_table1.sh should be trained
#
# Usage:
#   ./scripts/reproduce_figure3.sh

set -e

echo "============================================================"
echo "Reproducing Figure 3: Locality Curse Forensics"
echo "============================================================"

# Define checkpoint paths
BASELINE_CKPT="outputs/deit_baseline_cifar/checkpoints/best_model.pth"
CNN_DISTILL_CKPT="outputs/deit_resnet18_distill/checkpoints/best_model.pth"
DINO_DISTILL_CKPT="outputs/deit_dinov2_cka/checkpoints/best_model.pth"

# Check that all checkpoints exist
missing=0
for ckpt in "$BASELINE_CKPT" "$CNN_DISTILL_CKPT" "$DINO_DISTILL_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Missing checkpoint: $ckpt"
        missing=1
    fi
done

if [ $missing -eq 1 ]; then
    echo ""
    echo "Please run ./scripts/reproduce_table1.sh first to train all models."
    exit 1
fi

echo ""
echo "Running Locality Curse forensics analysis..."
echo ""

python main.py analyze-locality configs/cifar10/baselines/deit_tiny.yaml \
    "$CNN_DISTILL_CKPT" "$DINO_DISTILL_CKPT" "$BASELINE_CKPT" \
    --names "CNN-Distilled" "DINO-Distilled" "Baseline" \
    --output-dir outputs/figure3_forensics \
    --num-samples 512

echo ""
echo "============================================================"
echo "Figure 3 reproduction complete!"
echo "Output files:"
echo "  - outputs/figure3_forensics/locality_spectrum.png"
echo "  - outputs/figure3_forensics/layer_progression.png"
echo "  - outputs/figure3_forensics/forensics_summary.png"
echo "  - outputs/figure3_forensics/*_forensics.json"
echo "============================================================"
