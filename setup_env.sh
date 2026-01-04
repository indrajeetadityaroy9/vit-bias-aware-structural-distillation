#!/bin/bash
# setup_env.sh - H100 Optimized Environment Setup
#
# This script creates a virtual environment with all dependencies
# configured for optimal performance on H100 (SXM5) GPUs with CUDA 12.1+.
#
# Requirements:
#   - Python 3.10+ (3.11 recommended)
#   - CUDA 12.1+ drivers installed
#   - ~20GB disk space for environment
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# Estimated time: 10-15 minutes (Flash Attention compilation takes longest)

set -e

echo "============================================================"
echo "H100 Environment Setup for ViT Inductive Bias Distillation"
echo "============================================================"

# 1. Create and activate virtual environment
echo ""
echo "[Step 1/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and build tools (Critical for building FlashAttn)
echo ""
echo "[Step 2/6] Upgrading pip and build tools..."
pip install --upgrade pip wheel setuptools ninja packaging

# 3. Install PyTorch (H100 requires CUDA 12.1+)
# We install this FIRST to ensure subsequent packages link against the correct torch version
echo ""
echo "[Step 3/6] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Core Dependencies
echo ""
echo "[Step 4/6] Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Install PyHessian (Not on PyPI, must install from source)
# Used for: Hessian Trace Analysis in Locality Curse forensics
echo ""
echo "[Step 5/6] Installing PyHessian from source..."
pip install git+https://github.com/amirgholami/PyHessian.git

# 6. Install Flash Attention 2 (Optional but recommended for H100 speed)
# This takes a few minutes to compile
echo ""
echo "[Step 6/6] Installing Flash Attention 2 (this may take a few minutes)..."
pip install flash-attn --no-build-isolation || {
    echo "WARNING: Flash Attention installation failed."
    echo "This is optional - PyTorch SDPA will be used as fallback."
    echo "To retry manually: pip install flash-attn --no-build-isolation"
}

# Verification
echo ""
echo "============================================================"
echo "Environment setup complete!"
echo "============================================================"
echo ""
echo "Verification:"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU count: {torch.cuda.device_count()}')

# Check for Flash Attention
try:
    import flash_attn
    print(f'  Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('  Flash Attention: Not installed (SDPA fallback will be used)')

# Check for timm
import timm
print(f'  timm version: {timm.__version__}')

# Check for pyhessian
try:
    import pyhessian
    print('  PyHessian: Installed')
except ImportError:
    print('  PyHessian: Not installed')
"

echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To run training:"
echo "  python main.py train configs/cifar10/baselines/deit_tiny.yaml --num-gpus 2"
echo "============================================================"
