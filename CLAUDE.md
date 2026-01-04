# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for studying inductive bias mismatch in heterogeneous knowledge distillation, specifically CNN-to-ViT transfer. The core hypothesis: negative transfer from CNN→ViT distillation stems from conflicting inductive biases, not weak teachers.

## Commands

### Training

```bash
# Standard training (AdaptiveCNN or ViT without distillation)
python main.py train configs/<config>.yaml --num-gpus 2

# CNN→DeiT knowledge distillation (requires pre-trained teacher checkpoint)
python main.py train-distill configs/deit_cifar_config.yaml --num-gpus 2

# Self-supervised distillation (DINOv2→DeiT, CKA structural alignment)
python main.py train-ss-distill configs/deit_ss_distill_cka_cifar_config.yaml --num-gpus 2
```

### Evaluation & Analysis

```bash
# Evaluate model on test set
python main.py evaluate configs/<config>.yaml ./outputs/checkpoints/best_model.pth

# Single image inference with optional TTA
python main.py test configs/<config>.yaml ./checkpoint.pth image.jpg --tta

# Research analytics (CKA heatmaps, attention distance, Hessian)
python main.py analyze configs/<config>.yaml ./checkpoint.pth --metrics cka,attention --output-dir outputs/analytics

# Locality Curse Forensics (compare CNN-distilled vs DINO-distilled vs baseline)
python main.py analyze-locality configs/deit_cifar_config.yaml \
    outputs/cnn_distilled.pth outputs/dino_distilled.pth outputs/baseline.pth \
    -n "CNN-Distilled" -n "DINO-Distilled" -n "Baseline" \
    -o outputs/forensics
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_critical_bugs.py::TestTokenCorrelationLoss -v
```

## Architecture

### Core Components

- **`main.py`**: Unified CLI entry point with `unified_ddp_worker()` handling all training modes (standard, distill, ss_distill)
- **`src/models.py`**: `ModelFactory` creates models; `AdaptiveCNN` with SE blocks for MNIST/CIFAR
- **`src/vit.py`**: `DeiT` implementation with distillation token, hybrid patch embedding, `forward_with_intermediates()` for extracting layer representations
- **`src/distillation/`**: Knowledge distillation module
  - `engine.py`: `DistillationTrainer` (CNN→DeiT soft/hard KL), `SelfSupervisedDistillationTrainer` (DINOv2→DeiT)
  - `teachers.py`: DINOv2 teacher loading (`load_dino_teacher()`)
  - `losses/`: `standard.py` (KL), `token.py` (TokenRepresentationLoss, TokenCorrelationLoss), `structural.py` (CKALoss, GramMatrixLoss), `combined.py`
- **`src/training/`**: Training infrastructure
  - `engine.py`: `Trainer` and `DDPTrainer` with DDP, AMP (BF16), SWA, gradient clipping
  - `checkpointing.py`: `build_checkpoint_dict()`, `restore_rng_state()` for reproducibility
  - `optimizers.py`: Fused optimizers and schedulers
  - `components.py`: `EarlyStopping`, `LabelSmoothing`

### Distillation Loss Hierarchy

```
SelfSupervisedDistillationLoss
├── L_ce: Cross-entropy on CLS head only
├── L_tok: TokenRepresentationLoss (cosine/MSE on projected intermediate tokens)
├── L_rel: TokenCorrelationLoss (KL divergence on correlation matrices, staged warmup)
├── L_cka: CKALoss (Centered Kernel Alignment for structural similarity)
└── L_gram: GramMatrixLoss (Frobenius norm, ablation baseline)
```

### Key Design Patterns

1. **Staged training**: `rel_warmup_epochs` delays correlation loss activation for stability
2. **Dual-augment mode**: Clean images for teacher, augmented for student (`use_dual_augment=True`)
3. **Token interpolation**: Teacher tokens (196 for 224×224) are bilinearly interpolated to match student count (64 for CIFAR)
4. **CLS-only mode**: `use_cls_only=True` uses batch-wise CKA on CLS tokens instead of noisy spatial interpolation

### Configuration System

YAML configs in `configs/` with nested dataclasses in `src/config.py`:
- `DataConfig`, `ModelConfig`, `TrainingConfig`, `LoggingConfig`
- `ViTConfig`: DeiT-specific (patch_size, embed_dim, depth, distillation token)
- `DistillationConfig`: CNN teacher settings
- `SelfSupervisedDistillationConfig`: DINOv2 teacher, CKA/token loss weights, warmup epochs

### Model Types

Registered in `ModelFactory`:
- `adaptive_cnn`: SE-ResNet for MNIST (709K params) / CIFAR (17.6M params)
- `deit`: Data-efficient Image Transformer with distillation token
- `resnet18_cifar`: ResNet-18 adapted for CIFAR (classic CNN control)
- `convnext_v2_tiny`: Modern CNN bridge (requires timm)

## Key Implementation Details

- **Checkpoints**: Use `build_checkpoint_dict()` and `restore_rng_state()` from `src/training/checkpointing.py` for reproducibility
- **H100 optimizations**: BF16 (`use_bf16`), `torch.compile` (`use_compile`), TF32 matmul (`use_tf32`), fused optimizers (`use_fused_optimizer`)
- **DDP**: All training uses `DistributedDataParallel`; single GPU is DDP with world_size=1
- **Analytics**: `AnalyticsRunner` in `src/analytics/engine.py` computes CKA matrices, attention distances, Hessian trace
- **Locality Curse Forensics**: `LocalityCurseForensics` class in `src/analytics/engine.py` for proving CNN→ViT inductive bias mismatch

### Locality Curse Forensics Kit

Diagnostic toolkit in `src/analytics/engine.py` to prove the "Locality Curse" hypothesis:

**Core Hypothesis**: CNN teachers damage ViT students by forcing local attention patterns.

**Key Metrics** (per model):
- `avg_attention_distance`: Mean attention distance in patch units (lower = more local)
- `collapsed_heads_ratio`: Fraction of heads with MAD < 1.5 (local heads)
- `avg_cls_dispersion`: Spatial variance of CLS token attention (lower = focused)
- `avg_cls_self_attn`: CLS self-attention score (> 0.9 indicates feature collapse)
- `hessian_trace`: Loss landscape curvature (higher = sharper minima, worse generalization)

**Expected Ordering** (Three-Way Comparison):
```
CNN-Distilled (local) < Baseline (medium) < DINO-Distilled (global)
```

**Output Files**:
- `{model_name}_forensics.json`: Per-model metrics and per-layer statistics
- `locality_spectrum.png`: Sorted head distances (convex = local, concave = global)
- `layer_progression.png`: Attention distance by layer depth
- `forensics_summary.png`: 4-panel publication figure

**Safety Check**: If `cls_collapse_ratio > 0.5`, model has feature collapse (different pathology than locality curse)

### Analytics Module Structure

```
src/analytics/
├── engine.py           # AnalyticsRunner, LocalityCurseForensics
├── metrics/
│   ├── attention.py    # AttentionDistanceAnalyzer
│   ├── representation.py  # CKA computation
│   └── geometry.py     # Hessian analysis
└── visualization/
    └── plotting.py     # Publication-quality plots
```

### Important Training Flags

In `TrainingConfig`:
- `use_bf16`: BF16 instead of FP16 (native H100 support, no GradScaler needed)
- `use_compile`: Enable `torch.compile` for kernel fusion
- `compile_mode`: 'default', 'reduce-overhead', or 'max-autotune'
- `use_fused_optimizer`: Use fused AdamW/SGD kernels
- `use_tf32`: Enable TF32 for matrix operations

In `SelfSupervisedDistillationConfig`:
- `rel_warmup_epochs`: Delay correlation loss for training stability
- `cka_warmup_epochs`: Delay CKA loss activation
- `use_dual_augment`: Clean images for teacher, augmented for student
- `use_cls_only`: Use CLS tokens only for CKA (avoids spatial interpolation noise)
