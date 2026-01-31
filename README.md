# Inductive Bias Mismatch in Heterogeneous Knowledge Distillation

Negative transfer phenomenon in CNN-to-ViT distillation. Negative transfer in CNN→ViT distillation stems from conflicting inductive biases—not weak teachers.

1. **Alignment > Capacity**: Weaker teacher (ConvNeXt, 93.1%) produces better student (90.11%) than stronger teacher (ResNet-18, 95.1% → 90.05%)
2. **Structure is 99.5% Sufficient**: Self-supervised structural distillation (DINOv2 → DeiT) achieves 89.69% using only geometric alignment
3. **The "Locality Curse"**: CNN-distilled ViTs develop local attention (layer 2: 1.85 patch distance) vs DINOv2-distilled global attention (3.49, +88.6%)

## Usage

All training uses `torchrun` for distributed execution:

### Standard Training

```bash
# Train CNN teacher
torchrun --nproc_per_node=1 main.py train configs/cifar_improved_config.yaml

# Train ResNet-18 teacher
torchrun --nproc_per_node=1 main.py train configs/resnet18_cifar_config.yaml

# Train ConvNeXt V2 teacher
torchrun --nproc_per_node=1 main.py train configs/convnext_v2_cifar_config.yaml

# Train DeiT baseline (no distillation)
torchrun --nproc_per_node=1 main.py train configs/deit_cifar_config.yaml
```

### CNN → DeiT Distillation

```bash
# First train a teacher, then distill
torchrun --nproc_per_node=1 main.py train-distill configs/deit_cifar_config.yaml
```

### DINOv2 → DeiT Structural Distillation

```bash
# CKA structural alignment (primary method)
torchrun --nproc_per_node=1 main.py train-ss-distill configs/deit_ss_distill_cka_cifar_config.yaml

# Gram matrix ablation
torchrun --nproc_per_node=1 main.py train-ss-distill configs/deit_ss_distill_gram_ablation_config.yaml

# CE-only ablation (no structural loss)
torchrun --nproc_per_node=1 main.py train-ss-distill configs/deit_ss_distill_cifar_ablation_ce_only.yaml
```

### Evaluation

```bash
python main.py evaluate configs/deit_cifar_config.yaml outputs/checkpoints/best_model.pth
```

### Analytics

```bash
# Run all analytics (hessian, attention, cka)
python main.py analyze configs/deit_cifar_config.yaml outputs/checkpoints/best_model.pth

# Specific metrics
python main.py analyze configs/deit_cifar_config.yaml outputs/checkpoints/best_model.pth \
    --metrics attention,cka \
    --output-dir outputs/analytics
```

## Available Configs

| Config | Description |
|--------|-------------|
| `cifar_improved_config.yaml` | AdaptiveCNN teacher |
| `resnet18_cifar_config.yaml` | ResNet-18 teacher |
| `convnext_v2_cifar_config.yaml` | ConvNeXt V2 teacher |
| `deit_cifar_config.yaml` | DeiT student (standard/distill) |
| `deit_ss_distill_cka_cifar_config.yaml` | DINOv2→DeiT with CKA loss |
| `deit_ss_distill_gram_ablation_config.yaml` | Gram matrix ablation |
| `deit_ss_distill_cifar_ablation_ce_only.yaml` | CE-only baseline |
| `mnist_improved_config.yaml` | MNIST training |

## Results

### Transfer Efficiency

| Teacher | Teacher Acc | Student Acc | Transfer Efficiency |
|---------|-------------|-------------|---------------------|
| None (baseline) | — | 86.02% | — |
| ResNet-18 | 95.10% | 90.05% | 94.7% |
| ConvNeXt V2 | 93.10% | **90.11%** | **96.8%** |
| DINOv2 (structural) | N/A | 89.69% | — |

### Attention Distance (The "Locality Curse")

| Layer | ResNet→DeiT | ConvNeXt→DeiT | DINOv2→DeiT |
|-------|-------------|---------------|-------------|
| 2 | **1.85** | 2.20 | **3.49** (+88.6%) |
| 3 | 2.68 | 2.33 | **3.56** (+32.8%) |
| Mean | 3.31 | 3.28 | **3.73** (+12.7%) |

CNN-distilled students exhibit attention collapse in layers 2-6. DINOv2-distilled students maintain global attention.
