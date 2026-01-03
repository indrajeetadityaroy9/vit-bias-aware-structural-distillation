# Technical Overview: Dataset-Adaptive CNN Framework

## Project Summary

A research framework for studying **inductive bias mismatch** in heterogeneous knowledge distillation, specifically CNN-to-ViT transfer. The core hypothesis: negative transfer from CNN→ViT distillation stems from conflicting inductive biases (locality vs. global attention), not weak teachers.

**Codebase Statistics**: ~8,400 lines of Python across 10 core modules, 56 classes, 18 configuration presets.

---

## Project Tree Structure

```
dataset-adaptive-CNN/
├── main.py                          # Unified CLI entry point (1,050 lines)
│   ├── train                        # Standard training (AdaptiveCNN, DeiT)
│   ├── train-distill                # CNN→DeiT knowledge distillation
│   ├── train-ss-distill             # DINOv2→DeiT self-supervised distillation
│   ├── evaluate                     # Test set evaluation with metrics
│   ├── test                         # Single image inference with TTA
│   ├── analyze                      # CKA, attention, Hessian analytics
│   └── analyze-locality             # Locality Curse Forensics comparison
│
├── src/                             # Core library modules
│   ├── config.py                    # Type-safe configuration (440 lines)
│   │   ├── DataConfig               # Dataset, augmentation, normalization
│   │   ├── ModelConfig              # Architecture specification
│   │   ├── TrainingConfig           # Optimizer, scheduler, H100 flags
│   │   ├── ViTConfig                # DeiT-specific parameters
│   │   ├── DistillationConfig       # CNN teacher settings
│   │   ├── SelfSupervisedDistillationConfig  # DINOv2 teacher settings
│   │   └── ConfigManager            # YAML/JSON loading with validation
│   │
│   ├── models.py                    # CNN architectures (254 lines)
│   │   ├── SEBlock                  # Squeeze-and-Excitation attention
│   │   ├── ResidualBlock            # Residual block with optional SE
│   │   ├── AdaptiveCNN              # Dataset-adaptive SE-ResNet
│   │   │   ├── MNIST variant        # 709K params, 6 blocks
│   │   │   └── CIFAR variant        # 17.6M params, 11 blocks
│   │   └── ModelFactory             # Registry pattern for model creation
│   │
│   ├── vit.py                       # Vision Transformer (716 lines)
│   │   ├── DropPath                 # Stochastic depth regularization
│   │   ├── PatchEmbed               # Conv2d patch tokenization
│   │   ├── HybridPatchEmbed         # 2-layer conv stem option
│   │   ├── MultiHeadSelfAttention   # MHSA with Flash Attention v2
│   │   ├── MLP                      # Transformer FFN with GELU
│   │   ├── TransformerBlock         # Pre-LN architecture
│   │   └── DeiT                     # Complete DeiT-Tiny implementation
│   │       ├── Distillation token   # Separate head for KD
│   │       ├── Position interpolation # Multi-resolution support
│   │       └── forward_with_intermediates() # Layer extraction
│   │
│   ├── teachers.py                  # Teacher architectures (310 lines)
│   │   ├── BasicBlock               # ResNet residual block
│   │   ├── ResNet18CIFAR            # Classic CNN (11.2M params)
│   │   └── ConvNeXtV2Tiny           # Modern hybrid CNN (28.6M params)
│   │
│   ├── training.py                  # Training infrastructure (657 lines)
│   │   ├── build_optimizer()        # H100-optimized (fused kernels)
│   │   ├── build_scheduler()        # Step, Cosine, Plateau, Cyclic
│   │   ├── build_checkpoint_dict()  # Complete state serialization
│   │   ├── restore_rng_state()      # Reproducible resume
│   │   ├── EarlyStopping            # Patience-based stopping
│   │   ├── LabelSmoothingCrossEntropy # Soft label support
│   │   ├── Trainer                  # Single-GPU with AMP, SWA
│   │   └── DDPTrainer               # Distributed Data Parallel
│   │
│   ├── distillation.py              # Knowledge distillation (1,973 lines)
│   │   ├── DistillationLoss         # Hard/soft KL divergence
│   │   ├── DistillationTrainer      # CNN→DeiT trainer
│   │   ├── ProjectionHead           # Dimension alignment MLP
│   │   ├── TokenRepresentationLoss  # L_tok: cosine/MSE on tokens
│   │   ├── TokenCorrelationLoss     # L_rel: correlation matrix KL
│   │   ├── CKALoss                  # Centered Kernel Alignment
│   │   ├── GramMatrixLoss           # Frobenius norm (ablation)
│   │   ├── LayerWiseStructuralLoss  # Multi-layer structural matching
│   │   ├── SelfSupervisedDistillationLoss # Combined loss scheduler
│   │   └── SelfSupervisedDistillationTrainer # DINOv2→DeiT trainer
│   │
│   ├── analytics.py                 # Research analytics (1,674 lines)
│   │   ├── HessianAnalyzer          # Loss landscape curvature
│   │   ├── AttentionDistanceAnalyzer # Locality metrics
│   │   │   └── compute_head_statistics() # Per-head MAD, entropy
│   │   ├── CKAAnalyzer              # Representational similarity
│   │   ├── AnalyticsRunner          # Unified analysis orchestrator
│   │   ├── AnalyticsVisualizer      # Publication-quality plots
│   │   └── LocalityCurseForensics   # Comparative diagnostics
│   │
│   ├── datasets.py                  # Data loading (598 lines)
│   │   ├── Cutout                   # Occlusion augmentation
│   │   ├── MixingDataset            # MixUp/CutMix wrapper
│   │   ├── DualAugmentDataset       # Teacher/student split paths
│   │   └── DatasetManager           # Static loader utilities
│   │
│   ├── evaluation.py                # Model evaluation (327 lines)
│   │   ├── ModelEvaluator           # Metrics, confusion matrix, ROC
│   │   └── TestTimeAugmentation     # Multi-view averaging
│   │
│   └── visualization.py             # Visualization utilities (306 lines)
│       ├── FeatureMapVisualizer     # Layer activation grids
│       ├── GradCAM                  # Gradient-weighted activation maps
│       └── TrainingVisualizer       # Loss/accuracy curves
│
├── configs/                         # YAML configuration presets (18 files)
│   ├── mnist_config.yaml            # MNIST baseline
│   ├── cifar_config.yaml            # CIFAR-10 baseline
│   ├── deit_cifar_config.yaml       # DeiT standalone training
│   ├── deit_resnet18_distill_config.yaml    # CNN→DeiT distillation
│   ├── deit_ss_distill_cka_cifar_config.yaml # DINOv2→DeiT with CKA
│   ├── resnet18_cifar_config.yaml   # ResNet-18 teacher training
│   └── ...                          # Additional ablation configs
│
├── tests/                           # Test suite
│   ├── test_critical_bugs.py        # Core functionality tests
│   └── test_forensics_dry_run.py    # Forensics pipeline validation
│
├── scripts/
│   └── generate_comparison_plots.py # Results visualization
│
├── pyproject.toml                   # Package configuration
├── CLAUDE.md                        # AI assistant guidance
└── README.md                        # Project documentation
```

---

## Theoretical Foundations

### 1. The Locality Curse Hypothesis

**Core Claim**: CNN teachers damage ViT students by forcing local attention patterns, leading to:
- Reduced effective receptive field
- Sharper loss landscape (poor generalization)
- Lower test accuracy despite strong training signals

**Inductive Bias Conflict**:
| Architecture | Inductive Bias | Attention Pattern |
|--------------|----------------|-------------------|
| CNN (ResNet) | Strong locality | Fixed 3×3 receptive field |
| ViT (DeiT)   | Global context  | Learnable full-image attention |
| CNN (ConvNeXt)| Hybrid         | Depthwise + global aggregation |

### 2. Knowledge Distillation Framework

**Standard KD Loss** (Hinton et al., 2015):
```
L_KD = α·L_CE(y, p_s) + (1-α)·τ²·KL(softmax(z_t/τ) || softmax(z_s/τ))
```
where:
- `α`: Hard label weight (0.5 default)
- `τ`: Temperature (4.0 default, softens distributions)
- `z_t, z_s`: Teacher/student logits

**DeiT Distillation** (Touvron et al., 2021):
- Separate distillation token learns from teacher
- Hard distillation: `L_dist = CE(argmax(z_t), p_dist)`
- Soft distillation: `L_dist = KL(z_t/τ || z_dist/τ)`

### 3. Self-Supervised Structural Distillation

**Token Representation Loss (L_tok)**:
```
L_tok = Σ_l λ_l · (1 - cos(proj(s_l), interp(t_l)))
```
- Matches intermediate token representations
- Bilinear interpolation for resolution mismatch (196→64 tokens)

**Token Correlation Loss (L_rel)** (CSKD-inspired):
```
L_rel = KL(corr(S_l) || corr(T_l))
```
- Matches self-correlation structure, not absolute values
- Staged warmup: disabled for first N epochs

**Centered Kernel Alignment (CKA)** (Kornblith et al., 2019):
```
CKA(X, Y) = HSIC(X, Y) / √(HSIC(X, X) · HSIC(Y, Y))
```
- Scale-invariant structural similarity
- Linear kernel: `K = XX^T`
- RBF kernel: `K_ij = exp(-||x_i - x_j||² / 2σ²)`

### 4. Loss Landscape Analysis

**Hessian Trace** (Spaced KD, Theorem 1):
```
Tr(H) = Σ_i λ_i  (sum of eigenvalues)
```
- Estimated via Hutchinson's method (stochastic trace)
- Lower trace → flatter landscape → better generalization
- **Hypothesis**: CNN-distilled models have higher trace

**Attention Distance Metric**:
```
d^{l,h} = Σ_{i,j} A^{l,h}_{i,j} · ||pos_i - pos_j||_2
```
- Weighted average distance in patch coordinates
- Lower distance → more local attention → "cursed"

**CLS Spatial Dispersion**:
```
σ²_CLS = Var_spatial(A_{CLS→patches})
```
- Variance of CLS attention across spatial grid
- Lower dispersion → focused attention → local bias

---

## Key Mechanisms

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Loading                      │
│  YAML → ConfigManager → DataConfig + ModelConfig + ...       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Preparation                          │
│  DatasetManager.create_data_loaders()                        │
│  ├── MixingDataset (MixUp/CutMix)                           │
│  └── DualAugmentDataset (clean teacher / aug student)       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Creation                            │
│  ModelFactory.create_model()                                 │
│  ├── Student: DeiT (trainable)                              │
│  └── Teacher: ResNet18/ConvNeXt/DINOv2 (frozen)             │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training Loop (DDP)                         │
│  DDPTrainer / DistillationTrainer / SSDistillTrainer         │
│  ├── Forward: student(x) → cls_logits, dist_logits          │
│  ├── Forward: teacher(x) → teacher_out (no_grad)            │
│  ├── Loss: L_CE + L_dist + L_tok + L_rel + L_CKA            │
│  ├── Backward: gradient accumulation + clipping              │
│  ├── Optimizer step (fused AdamW)                           │
│  └── Checkpoint: build_checkpoint_dict()                     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Evaluation                               │
│  ModelEvaluator.evaluate()                                   │
│  ├── Accuracy, Precision, Recall, F1                        │
│  ├── Confusion matrix, ROC curves                           │
│  └── TestTimeAugmentation (optional)                         │
└─────────────────────────────────────────────────────────────┘
```

### Analytics Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Trained DeiT Models                         │
│  CNN-Distilled | DINO-Distilled | Baseline (no distillation) │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LocalityCurseForensics                       │
│  ├── HessianAnalyzer.compute_trace()                        │
│  │   └── Hutchinson's estimator → loss curvature            │
│  ├── AttentionDistanceAnalyzer                              │
│  │   └── compute_head_statistics()                          │
│  │       ├── mean_distance (patch-to-patch MAD)             │
│  │       ├── entropy (attention concentration)               │
│  │       ├── cls_dispersion (spatial variance)              │
│  │       └── cls_self_attn (collapse indicator)             │
│  └── CKAAnalyzer.compute_cka_matrix()                       │
│      └── Layer-to-layer alignment heatmap                   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  AnalyticsVisualizer                          │
│  ├── plot_locality_spectrum() → sorted head distances        │
│  ├── plot_layer_progression() → distance by layer           │
│  ├── plot_forensics_summary() → 4-panel figure              │
│  └── plot_cka_heatmap() → representational similarity       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Expected Results                          │
│  CNN-Distilled < Baseline < DINO-Distilled (for MAD)         │
│  CNN-Distilled > Baseline > DINO-Distilled (for Hessian)     │
└─────────────────────────────────────────────────────────────┘
```

### H100 Optimization Stack

```
┌─────────────────────────────────────────────────────────────┐
│                   TrainingConfig Flags                        │
│  use_bf16=True, use_compile=True, use_fused_optimizer=True   │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Hardware Acceleration Layers                     │
├─────────────────────────────────────────────────────────────┤
│  BF16 Mixed Precision                                        │
│  └── torch.autocast('cuda', dtype=torch.bfloat16)           │
│  └── No GradScaler needed (BF16 has same exponent as FP32)  │
├─────────────────────────────────────────────────────────────┤
│  Flash Attention v2 (SDPA)                                   │
│  └── F.scaled_dot_product_attention()                       │
│  └── 25-40% speedup, O(N) memory vs O(N²)                   │
├─────────────────────────────────────────────────────────────┤
│  Fused Optimizers                                            │
│  └── torch.optim.AdamW(fused=True)                          │
│  └── Single kernel for param update                         │
├─────────────────────────────────────────────────────────────┤
│  TF32 Matmul                                                 │
│  └── torch.backends.cuda.matmul.allow_tf32 = True           │
│  └── 3.5x faster on H100 Tensor Cores                       │
├─────────────────────────────────────────────────────────────┤
│  torch.compile                                               │
│  └── Graph fusion, kernel optimization                       │
│  └── Modes: default, reduce-overhead, max-autotune          │
└─────────────────────────────────────────────────────────────┘
```

---

## Distillation Loss Hierarchy

```
SelfSupervisedDistillationLoss
│
├── L_CE: Cross-entropy on CLS head
│   └── LabelSmoothingCrossEntropy (supports soft labels)
│
├── L_tok: TokenRepresentationLoss
│   ├── Cosine similarity on intermediate tokens
│   ├── Projection head for dimension alignment
│   └── Bilinear interpolation for token count mismatch
│
├── L_rel: TokenCorrelationLoss
│   ├── KL divergence on correlation matrices
│   ├── Staged warmup (disabled for first N epochs)
│   └── Temperature scaling for soft matching
│
├── L_cka: CKALoss
│   ├── Centered Kernel Alignment
│   ├── Linear or RBF kernel options
│   └── CLS-only mode for global semantics
│
└── L_gram: GramMatrixLoss (ablation baseline)
    └── Frobenius norm of gram matrices
```

---

## Configuration System

### Key Config Relationships

```yaml
# Model determines input processing
model.model_type: deit → requires vit config
model.model_type: adaptive_cnn → uses model.architecture

# Training mode determines loss
train → L_CE only
train-distill → L_CE + L_dist (DistillationConfig required)
train-ss-distill → L_CE + L_tok + L_rel + L_cka (SSDistillConfig required)

# DINOv2 teacher requires specific settings
ss_distillation.teacher_type: dinov2
ss_distillation.teacher_model_name: dinov2_vits14
ss_distillation.teacher_embed_dim: 384
ss_distillation.use_dual_augment: true  # Critical for DINOv2
```

### Preset Configurations

| Config | Dataset | Model | Training Mode | Key Settings |
|--------|---------|-------|---------------|--------------|
| `cifar_config.yaml` | CIFAR-10 | AdaptiveCNN | Standard | 200 epochs, AutoAugment |
| `deit_cifar_config.yaml` | CIFAR-10 | DeiT-Tiny | Standard | 300 epochs, drop_path=0.1 |
| `deit_resnet18_distill_config.yaml` | CIFAR-10 | DeiT-Tiny | CNN→ViT | α=0.5, τ=4.0 |
| `deit_ss_distill_cka_cifar_config.yaml` | CIFAR-10 | DeiT-Tiny | DINOv2→ViT | CKA + L_tok |

---

## Expected Experimental Results

### Three-Way Model Comparison

| Metric | CNN-Distilled | Baseline | DINO-Distilled |
|--------|---------------|----------|----------------|
| Test Accuracy | ~88% | ~85% | >89% |
| Hessian Trace | High (sharp) | Medium | Low (flat) |
| Avg Attention Distance | Low (local) | Medium | High (global) |
| Collapsed Heads Ratio | High | Medium | Low |
| CLS Dispersion | Low (focused) | Medium | High (broad) |

### Interpretation

- **CNN-Distilled < Baseline**: Negative transfer confirmed
- **DINO-Distilled > Baseline**: Self-supervised teachers help
- **High Hessian for CNN**: Sharper minima = worse generalization
- **Low MAD for CNN**: Locality curse = attention forced local

---

## Usage Examples

### Training a Baseline DeiT
```bash
python main.py train configs/deit_cifar_config.yaml --num-gpus 2
```

### CNN→DeiT Knowledge Distillation
```bash
# First train the teacher
python main.py train configs/resnet18_cifar_config.yaml --num-gpus 2

# Then distill to student
python main.py train-distill configs/deit_resnet18_distill_config.yaml --num-gpus 2
```

### DINOv2→DeiT Self-Supervised Distillation
```bash
python main.py train-ss-distill configs/deit_ss_distill_cka_cifar_config.yaml --num-gpus 2
```

### Locality Curse Forensics
```bash
python main.py analyze-locality configs/deit_cifar_config.yaml \
    outputs/cnn_distilled/best.pth \
    outputs/baseline/best.pth \
    outputs/dino_distilled/best.pth \
    -n "CNN-Distilled" -n "Baseline" -n "DINO-Distilled" \
    -o outputs/forensics
```

---

## References

1. **DeiT**: Touvron et al., "Training data-efficient image transformers & distillation through attention" (ICML 2021)
2. **CSKD**: Chen et al., "Cross-Architecture Knowledge Distillation" (CVPR 2022)
3. **CKA**: Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)
4. **Spaced KD**: "Knowledge Distillation with Spaced Repetition" (ICLR 2025)
5. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (CVPR 2024)
6. **Flash Attention**: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism" (ICLR 2024)
