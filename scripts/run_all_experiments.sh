#!/bin/bash
# run_all_experiments.sh - Master script for full experiment reproduction
#
# This script runs all experiments for the ViT Inductive Bias Distillation paper.
#
# Order:
#   1. Train all models (Table 1)
#   2. Run analytics (CKA, attention distance)
#   3. Run Locality Curse forensics (Figure 3)
#   4. Generate comparison plots
#
# Usage:
#   ./scripts/run_all_experiments.sh [--num-gpus N]
#
# Estimated time:
#   - 2x H100: ~8-12 hours
#   - 1x A100: ~16-24 hours

set -e

NUM_GPUS=${1:-2}
if [[ "$1" == "--num-gpus" ]]; then
    NUM_GPUS=$2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "ViT Inductive Bias Distillation - Full Experiment Suite"
echo "============================================================"
echo "Project directory: $PROJECT_DIR"
echo "GPUs: $NUM_GPUS"
echo "============================================================"
echo ""

# Step 1: Reproduce Table 1 (train all models)
echo "[1/4] Training all models (Table 1)..."
./scripts/reproduce_table1.sh --num-gpus $NUM_GPUS

# Step 2: Run analytics on each model
echo ""
echo "[2/4] Running analytics on trained models..."

for config in outputs/*/config.yaml; do
    if [ -f "$config" ]; then
        dir=$(dirname "$config")
        ckpt="$dir/checkpoints/best_model.pth"
        exp_name=$(basename "$dir")

        if [ -f "$ckpt" ]; then
            echo "  Analyzing: $exp_name"
            python main.py analyze "$config" "$ckpt" \
                --metrics cka,attention \
                --output-dir "outputs/analytics/$exp_name" \
                --num-samples 1024
        fi
    fi
done

# Step 3: Reproduce Figure 3 (Locality Curse forensics)
echo ""
echo "[3/4] Running Locality Curse forensics (Figure 3)..."
./scripts/reproduce_figure3.sh

# Step 4: Generate comparison plots
echo ""
echo "[4/4] Generating comparison plots..."
if [ -f "scripts/generate_comparison_plots.py" ]; then
    python scripts/generate_comparison_plots.py
else
    echo "  (generate_comparison_plots.py not found, skipping)"
fi

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results summary:"
echo "  - Model checkpoints: outputs/*/checkpoints/"
echo "  - Training curves: outputs/*/training_history.png"
echo "  - Analytics: outputs/analytics/"
echo "  - Forensics: outputs/figure3_forensics/"
echo ""
echo "Key files for paper:"
echo "  - Table 1: outputs/*/test_metrics.txt"
echo "  - Figure 3: outputs/figure3_forensics/forensics_summary.png"
echo "  - CKA heatmaps: outputs/analytics/*/cka_heatmap.png"
echo "============================================================"
