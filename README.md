# Dataset Adaptive Image Classification

A modular CNN classification pipeline that dynamically adjusts its architecture based on dataset characteristics. The pipeline implements dataset-specific model configurations, preprocessing strategies, and augmentation techniques to optimize performance across diverse image classification tasks.

## Features

1. **Adaptive Architecture Design**: A single model class that dynamically instantiates dataset-specific layer configurations with ResidualBlocks and SE attention
2. **Multi-GPU Training**: Distributed Data Parallel (DDP) support for efficient multi-GPU training
3. **Advanced Augmentation**: AutoAugment, CutMix, Cutout, and traditional augmentations
4. **Modern Training Techniques**: Mixed precision (AMP), Stochastic Weight Averaging (SWA), label smoothing, cosine annealing
5. **Interpretability Tools**: GradCAM, feature map visualization, confusion matrices, ROC curves

## Architecture

**MNIST Configuration** (6 residual blocks, 709K parameters):
```
Input (1×28×28)
    ↓
Conv(1→32, 3×3) + BN + ReLU + MaxPool     → 32×14×14
    ↓
ResidualBlock(32→32) × 2 + SE              → 32×14×14
    ↓
ResidualBlock(32→64, stride=2) + SE        → 64×7×7
ResidualBlock(64→64) + SE                  → 64×7×7
    ↓
ResidualBlock(64→128, stride=2) + SE       → 128×4×4
ResidualBlock(128→128) + SE                → 128×4×4
    ↓
AdaptiveAvgPool + Classifier(128→64→10)
```

**CIFAR-10 Configuration** (11 residual blocks, 17.6M parameters):
```
Input (3×32×32)
    ↓
Conv(3→64, 3×3) + BN + ReLU               → 64×32×32
    ↓
ResidualBlock(64→64) × 2 + SE              → 64×32×32
    ↓
ResidualBlock(64→128, stride=2) + SE       → 128×16×16
ResidualBlock(128→128) × 2 + SE            → 128×16×16
    ↓
ResidualBlock(128→256, stride=2) + SE      → 256×8×8
ResidualBlock(256→256) × 2 + SE            → 256×8×8
    ↓
ResidualBlock(256→512, stride=2) + SE      → 512×4×4
ResidualBlock(512→512) × 2 + SE            → 512×4×4
    ↓
AdaptiveAvgPool + Classifier(512→256→10)
```

Both architectures employ:
- **Residual Connections** for gradient flow in deep networks
- **Squeeze-and-Excitation (SE) Blocks** for channel attention
- **Batch Normalization** after each convolutional layer
- **Dropout** (p=0.3) for regularization
- **ReLU activations** for non-linearity

## Dataset Preprocessing

**MNIST Pipeline**:
- Grayscale conversion with automatic brightness inversion (threshold: 127)
- Resize to 28×28 pixels
- Normalization: μ=0.1307, σ=0.3081
- **Training augmentation**: Random rotation (±10°), random affine translation (10%), Cutout

**CIFAR-10 Pipeline**:
- **Training augmentation**: AutoAugment (CIFAR10 policy), CutMix (α=1.0), random crop (32×32, padding=4), random horizontal flip, Cutout
- **Test preprocessing**: Resize to 32×32 pixels
- Normalization: μ=[0.4914, 0.4822, 0.4465], σ=[0.2470, 0.2435, 0.2616]

## Training

**Optimization**:
- Optimizer: AdamW (MNIST) / SGD with Nesterov momentum (CIFAR-10)
- Loss function: Label Smoothing Cross-Entropy (smoothing=0.1)
- Gradient clipping: max norm = 1.0
- Learning rate scheduler: Cosine Annealing with warmup
- Stochastic Weight Averaging (SWA) in final 25% of training

**Training Configuration**:
- Multi-GPU: DDP with 2× NVIDIA H100 80GB
- Batch size: 512 per GPU (1024 effective)
- MNIST epochs: 50
- CIFAR-10 epochs: 100
- Mixed precision training (AMP) enabled
- Checkpointing: Save best model based on validation accuracy

## Experimental Setup

### Datasets

**MNIST** (Handwritten Digits):
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28×28 (grayscale)
- Classes: 10 (digits 0-9)

**CIFAR-10** (Natural Images):
- Training samples: 50,000
- Test samples: 10,000
- Image size: 32×32 (RGB)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Macro-averaged across all classes
- **Confusion Matrix**: Per-class error analysis
- **Top-K Accuracy**: Top-1 and Top-5 predictions

## Results

### MNIST Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.75%** |
| **Precision (Macro)** | 0.9975 |
| **Recall (Macro)** | 0.9975 |
| **F1-Score (Macro)** | 0.9975 |
| **Top-3 Accuracy** | 99.99% |
| **AUC (Macro)** | 1.0000 |

### CIFAR-10 Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **95.55%** |
| **Precision (Macro)** | 0.9556 |
| **Recall (Macro)** | 0.9555 |
| **F1-Score (Macro)** | 0.9555 |
| **Top-3 Accuracy** | 99.54% |
| **AUC (Macro)** | 0.9983 |

### Training Time (2× NVIDIA H100 80GB)

| Dataset | Training Time | Epochs |
|---------|---------------|--------|
| MNIST | ~1 minute | 50 |
| CIFAR-10 | ~7 minutes | 100 |

## Usage

### Training

```bash
# Train MNIST on single GPU
python main.py train configs/mnist_improved_config.yaml

# Train MNIST on multiple GPUs with DDP
python main.py train configs/mnist_improved_config.yaml --num-gpus 2

# Train CIFAR-10 on multiple GPUs with DDP
python main.py train configs/cifar_improved_config.yaml --num-gpus 2
```

### Evaluation

```bash
# Evaluate MNIST model
python main.py evaluate configs/mnist_improved_config.yaml ./outputs/checkpoints/best_model.pth

# Evaluate CIFAR-10 model
python main.py evaluate configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth
```

### Single Image Inference

```bash
# Inference with test-time augmentation
python main.py test configs/cifar_improved_config.yaml ./outputs/checkpoints/best_model.pth image.jpg --tta
```

## Project Structure

```
├── configs/                    # Configuration files
│   ├── mnist_improved_config.yaml
│   └── cifar_improved_config.yaml
├── src/
│   ├── config.py              # Configuration management
│   ├── models.py              # AdaptiveCNN with ResidualBlock + SE
│   ├── datasets.py            # Data loading and augmentation
│   ├── training.py            # Trainer and DDPTrainer classes
│   ├── evaluation.py          # Metrics and visualization
│   └── visualization.py       # GradCAM and feature maps
├── outputs/                    # Training outputs
│   ├── checkpoints/           # Model checkpoints
│   └── evaluation/            # Evaluation plots
└── main.py                    # CLI entry point
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- tqdm
- matplotlib
- scikit-learn
