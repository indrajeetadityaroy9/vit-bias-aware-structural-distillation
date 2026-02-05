"""
Configuration management with hierarchical config loading.

Supports two-layer config merging:
1. Global defaults (configs/default.yaml)
2. Specific config (e.g., experiments/baselines/deit_distill_cifar.yaml)

Specific config values override global defaults.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DataConfig:
    """Data loading and augmentation configuration."""

    def __init__(self, dataset=None, batch_size=64, num_workers=4, pin_memory=True,
                 persistent_workers=True, prefetch_factor=2, augmentation=None,
                 normalization=None, data_path="./data", drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.augmentation = augmentation if augmentation is not None else {}
        self.normalization = normalization if normalization is not None else {}
        self.data_path = data_path
        self.drop_last = drop_last


class ModelConfig:
    """Model architecture configuration."""

    def __init__(self, model_type="adaptive_cnn", in_channels=1, num_classes=10,
                 dropout=0.5, use_se=True, architecture=None, classifier_layers=None,
                 pretrained=False, drop_path_rate=0.0):
        self.model_type = model_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_se = use_se
        self.architecture = architecture if architecture is not None else []
        self.classifier_layers = classifier_layers if classifier_layers is not None else []
        self.pretrained = pretrained  # For pretrained models like ConvNeXt V2
        self.drop_path_rate = drop_path_rate  # Stochastic depth


class TrainingConfig:
    """Training hyperparameters and optimization settings."""

    def __init__(self, num_epochs=10, learning_rate=0.001, weight_decay=0.0005,
                 optimizer="adamw", scheduler="cosine", warmup_epochs=5,
                 gradient_clip_val=1.0, gradient_accumulation_steps=1,
                 use_amp=True, early_stopping=True,
                 early_stopping_patience=10, early_stopping_min_delta=0.001,
                 lr_scheduler_params=None, label_smoothing=0.1, use_swa=True,
                 swa_start_epoch=0.75, swa_lr=0.0005,
                 # H100 optimization flags
                 use_bf16=True, use_compile=True, compile_mode='max-autotune',
                 use_fused_optimizer=True, use_tf32=True):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.lr_scheduler_params = lr_scheduler_params if lr_scheduler_params is not None else {}
        self.label_smoothing = label_smoothing
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        # H100 optimization flags
        self.use_bf16 = use_bf16
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.use_fused_optimizer = use_fused_optimizer
        self.use_tf32 = use_tf32


class ViTConfig:
    """Vision Transformer (DeiT) specific configuration."""

    def __init__(self, img_size=32, patch_size=4,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 distillation=True, use_conv_stem=False, **_ignored):
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.distillation = distillation
        self.use_conv_stem = use_conv_stem


class DistillationConfig:
    """Knowledge distillation configuration."""

    def __init__(self, teacher_checkpoint=None, teacher_model_type='adaptive_cnn',
                 distillation_type='hard', alpha=0.5, tau=3.0,
                 distillation_warmup_epochs=0,
                 alpha_schedule='constant', alpha_start=0.0, alpha_end=0.5):
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_model_type = teacher_model_type
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.distillation_warmup_epochs = distillation_warmup_epochs

        self.alpha_schedule = alpha_schedule
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end


class SelfSupervisedDistillationConfig:
    """
    Self-supervised token correlation distillation configuration (CST-style).

    Uses a pretrained self-supervised ViT (DINO/DINOv2) as teacher instead of
    a weaker CNN, distilling token representations and correlations.

    Supports multiple structural loss types:
    - Token representation (L_tok): Cosine/MSE similarity on projected tokens
    - Token correlation (L_rel): KL/Frobenius on correlation matrices
    - CKA (L_cka): Centered Kernel Alignment for structural similarity
    - Gram matrix (L_gram): Direct gram matrix comparison (ablation baseline)
    """

    def __init__(
        self,
        teacher_type='dinov2',
        teacher_model_name='dinov2_vits14',
        teacher_embed_dim=384,
        token_layers=None,
        projection_dim=256,
        lambda_tok=1.0,
        token_loss_type='cosine',
        lambda_rel=0.1,
        correlation_temperature=0.1,
        correlation_loss_type='kl',
        use_pooled_correlation=True,
        rel_warmup_epochs=10,
        projector_warmup_epochs=0,
        use_cka_loss=False,
        lambda_cka=0.5,
        cka_kernel_type='linear',
        cka_warmup_epochs=5,
        use_gram_loss=False,
        lambda_gram=0.5,
        use_dual_augment=True,
        use_cls_only=False,
    ):
        self.teacher_type = teacher_type
        self.teacher_model_name = teacher_model_name
        self.teacher_embed_dim = teacher_embed_dim

        self.token_layers = token_layers if token_layers is not None else [6, 11]
        self.projection_dim = projection_dim
        self.lambda_tok = lambda_tok
        self.token_loss_type = token_loss_type

        self.lambda_rel = lambda_rel
        self.correlation_temperature = correlation_temperature
        self.correlation_loss_type = correlation_loss_type
        self.use_pooled_correlation = use_pooled_correlation

        self.rel_warmup_epochs = rel_warmup_epochs
        self.projector_warmup_epochs = projector_warmup_epochs

        self.use_cka_loss = use_cka_loss
        self.lambda_cka = lambda_cka
        self.cka_kernel_type = cka_kernel_type
        self.cka_warmup_epochs = cka_warmup_epochs

        self.use_gram_loss = use_gram_loss
        self.lambda_gram = lambda_gram

        self.use_dual_augment = use_dual_augment

        self.use_cls_only = use_cls_only


class Config:
    """Main configuration container."""

    def __init__(self, data=None, model=None, training=None,
                 vit=None, distillation=None, ss_distillation=None,
                 experiment_name="default", seed=42, device="cuda", output_dir="./outputs"):
        self.data = data
        self.model = model
        self.training = training
        self.vit = vit
        self.distillation = distillation
        self.ss_distillation = ss_distillation
        self.experiment_name = experiment_name
        self.seed = seed
        self.device = device
        self.output_dir = output_dir


# Module-level config functions

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_defaults_path(config_path: Path) -> Optional[Path]:
    """Find configs/default.yaml by walking up to project root."""
    search = config_path.parent.resolve()
    while search != search.parent:
        candidate = search / 'configs' / 'default.yaml'
        if candidate.exists():
            return candidate
        search = search.parent
    return None


def _parse_config(raw_config: Dict[str, Any], config_path: Optional[Path] = None) -> Config:
    """Parse raw config dictionary into Config object."""
    vit_config = None
    if 'vit' in raw_config:
        vit_config = ViTConfig(**raw_config['vit'])

    distillation_config = None
    if 'distillation' in raw_config:
        distillation_config = DistillationConfig(**raw_config['distillation'])

    ss_distillation_config = None
    if 'ss_distillation' in raw_config:
        ss_distillation_config = SelfSupervisedDistillationConfig(**raw_config['ss_distillation'])

    config = Config(
        data=DataConfig(**raw_config.get('data', {})),
        model=ModelConfig(**raw_config.get('model', {})),
        training=TrainingConfig(**raw_config.get('training', {})),
        vit=vit_config,
        distillation=distillation_config,
        ss_distillation=ss_distillation_config,
        **{k: v for k, v in raw_config.items()
           if k not in ['data', 'model', 'training', 'logging', 'vit', 'distillation', 'ss_distillation']}
    )

    validate_config(config)

    if config_path:
        print(f"config={config_path}")

    return config


def load_config(config_path, merge_defaults: bool = True) -> Config:
    """Load configuration with optional two-layer defaults merging.

    Two-layer merge (explicit > implicit):
    1. Load global defaults from configs/default.yaml (if exists)
    2. Deep merge with specific config (specific values override defaults)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix in ['.yml', '.yaml']:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f) or {}
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if merge_defaults:
        defaults_path = _find_defaults_path(config_path)
        if defaults_path:
            with open(defaults_path, 'r') as f:
                defaults_config = yaml.safe_load(f) or {}
            raw_config = _deep_merge(defaults_config, raw_config)
            print(f"config defaults={defaults_path}")

    return _parse_config(raw_config, config_path)


def validate_config(config: Config) -> None:
    """Validate configuration values."""
    valid_datasets = ['mnist', 'cifar']
    if config.data.dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {config.data.dataset}")

    valid_model_types = [
        'adaptive_cnn',
        'deit',
        'resnet18_cifar',
        'convnext_v2_tiny',
    ]
    if config.model.model_type not in valid_model_types:
        raise ValueError(
            f"Invalid model_type: {config.model.model_type}. "
            f"Valid types: {valid_model_types}"
        )

    if config.data.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if config.model.dropout < 0 or config.model.dropout > 1:
        raise ValueError("Dropout must be between 0 and 1")

    if config.training.num_epochs <= 0:
        raise ValueError("Number of epochs must be positive")

    if config.training.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    valid_optimizers = ['adam', 'adamw', 'sgd']
    if config.training.optimizer.lower() not in valid_optimizers:
        raise ValueError(f"Invalid optimizer: {config.training.optimizer}")

    valid_schedulers = ['step', 'cosine', 'plateau']
    if config.training.scheduler.lower() not in valid_schedulers:
        raise ValueError(f"Invalid scheduler: {config.training.scheduler}")


def save_config(config: Config, save_path) -> None:
    """Save configuration to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'device': config.device,
        'output_dir': config.output_dir
    }

    if config.vit is not None:
        config_dict['vit'] = config.vit.__dict__
    if config.distillation is not None:
        config_dict['distillation'] = config.distillation.__dict__
    if config.ss_distillation is not None:
        config_dict['ss_distillation'] = config.ss_distillation.__dict__

    if save_path.suffix in ['.yml', '.yaml']:
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    print(f"config saved={save_path}")
