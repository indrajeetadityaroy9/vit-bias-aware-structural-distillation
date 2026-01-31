"""
Model registry for unified model creation.

Provides decorator-based registration and factory function.

Usage:
    @register_model('my_model')
    def my_model_factory(config):
        return MyModel(config)

    # Then in __init__.py or elsewhere:
    from . import my_module  # Triggers registration

    # Create model:
    model = create_model('my_model', config)
"""
from typing import Any, Callable, Dict, List, Optional

# Global model registry
_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model factory function or class.

    Usage:
        @register_model('deit')
        class DeiT(nn.Module):
            def __init__(self, config):
                ...

    Args:
        name: Name to register the model under

    Returns:
        Decorator function
    """
    def decorator(cls_or_fn: Callable) -> Callable:
        if name in _MODEL_REGISTRY:
            print(f"Model '{name}' already registered. Overwriting.")
        _MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


def create_model(name: str, config: Any) -> Any:
    """Create a model by name using the registry.

    The config can be a dict or a Config object. The registry function
    will receive it as-is.

    Args:
        name: Registered model name (e.g., 'deit', 'adaptive_cnn')
        config: Configuration dict or Config object

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not registered
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available: {available}")

    model_cls = _MODEL_REGISTRY[name]
    return model_cls(config)


def list_models() -> List[str]:
    """List all registered model names.

    Returns:
        List of registered model names
    """
    return list(_MODEL_REGISTRY.keys())
