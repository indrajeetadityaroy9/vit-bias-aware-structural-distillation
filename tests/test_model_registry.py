"""
Tests for the model registry.

Verifies that:
1. All expected models are registered
2. create_model() works correctly
3. Models can be created with config dicts
"""
import pytest
import torch


class TestModelRegistry:
    """Tests for model registration and creation."""

    def test_list_models(self):
        """Verify all expected models are registered."""
        from src.modeling import list_models

        models = list_models()

        # Core models should be registered
        assert 'deit' in models
        assert 'adaptive_cnn' in models
        assert 'resnet18_cifar' in models

    def test_create_deit_model(self):
        """Verify DeiT model can be created."""
        from src.modeling import create_model

        config = {
            'in_channels': 3,
            'num_classes': 10,
            'img_size': 32,
            'patch_size': 4,
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'distillation': True,
        }

        model = create_model('deit', config)

        assert model is not None
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        # During training, DeiT with distillation returns tuple
        if isinstance(out, tuple):
            assert out[0].shape == (2, 10)
            assert out[1].shape == (2, 10)
        else:
            assert out.shape == (2, 10)

    def test_create_adaptive_cnn_cifar(self):
        """Verify AdaptiveCNN for CIFAR can be created."""
        from src.modeling import create_model

        config = {
            'in_channels': 3,
            'num_classes': 10,
            'dataset': 'cifar',
            'use_se': True,
        }

        model = create_model('adaptive_cnn', config)

        assert model is not None
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_create_adaptive_cnn_mnist(self):
        """Verify AdaptiveCNN for MNIST can be created."""
        from src.modeling import create_model

        config = {
            'in_channels': 1,
            'num_classes': 10,
            'dataset': 'mnist',
            'use_se': True,
        }

        model = create_model('adaptive_cnn', config)

        assert model is not None
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_create_resnet18_cifar(self):
        """Verify ResNet-18 CIFAR can be created."""
        from src.modeling import create_model

        config = {
            'in_channels': 3,
            'num_classes': 10,
        }

        model = create_model('resnet18_cifar', config)

        assert model is not None
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_unknown_model_raises_error(self):
        """Verify unknown model name raises ValueError."""
        from src.modeling import create_model

        with pytest.raises(ValueError, match="Unknown model"):
            create_model('nonexistent_model', {})

    def test_convnext_registration(self):
        """Verify ConvNeXt V2 is registered if timm available."""
        from src.modeling import list_models

        models = list_models()

        # ConvNeXt should be registered if timm is installed
        # We installed it in the venv, so it should be there
        assert 'convnext_v2_tiny' in models


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
