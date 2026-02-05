"""
Tests for the adaptive data pipeline.

Verifies:
1. DATASET_REGISTRY and IMAGENET_STATS correctness
2. apply_batch_mixing output shapes and target normalization
3. get_dataset_info registry lookup
4. DINO_NUM_LAYERS lookup table
5. get_transforms output shapes
"""

import pytest
import torch


class TestDatasetRegistry:
    """Verify registry correctness."""

    def test_registry_has_all_datasets(self):
        from src.data import DATASET_REGISTRY
        assert 'mnist' in DATASET_REGISTRY
        assert 'cifar' in DATASET_REGISTRY

    def test_cifar_stats(self):
        from src.data import DATASET_REGISTRY
        cifar = DATASET_REGISTRY['cifar']
        assert cifar['mean'] == (0.4914, 0.4822, 0.4465)
        assert cifar['std'] == (0.2470, 0.2435, 0.2616)
        assert cifar['image_size'] == 32
        assert cifar['in_channels'] == 3
        assert cifar['num_classes'] == 10
        assert len(cifar['class_names']) == 10

    def test_mnist_stats(self):
        from src.data import DATASET_REGISTRY
        mnist = DATASET_REGISTRY['mnist']
        assert mnist['mean'] == (0.1307,)
        assert mnist['std'] == (0.3081,)
        assert mnist['image_size'] == 28
        assert mnist['in_channels'] == 1
        assert mnist['num_classes'] == 10

    def test_imagenet_stats(self):
        from src.data import IMAGENET_STATS
        assert IMAGENET_STATS['mean'] == (0.485, 0.456, 0.406)
        assert IMAGENET_STATS['std'] == (0.229, 0.224, 0.225)


class TestBatchMixing:
    """Verify GPU batch mixing."""

    def test_mixup_output_shape(self):
        from src.data import apply_batch_mixing
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        mixed_img, mixed_tgt = apply_batch_mixing(images, targets, 10, mixup_alpha=0.8)
        assert mixed_img.shape == (8, 3, 32, 32)
        assert mixed_tgt.shape == (8, 10)

    def test_cutmix_output_shape(self):
        from src.data import apply_batch_mixing
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        mixed_img, mixed_tgt = apply_batch_mixing(images, targets, 10, cutmix_alpha=1.0)
        assert mixed_img.shape == (8, 3, 32, 32)
        assert mixed_tgt.shape == (8, 10)

    def test_both_mixup_cutmix_targets_sum_to_one(self):
        from src.data import apply_batch_mixing
        images = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 10, (8,))
        mixed_img, mixed_tgt = apply_batch_mixing(images, targets, 10, mixup_alpha=0.8, cutmix_alpha=1.0)
        assert mixed_tgt.sum(dim=1).allclose(torch.ones(8))

    def test_no_mixing_returns_onehot(self):
        from src.data import apply_batch_mixing
        images = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        out_img, out_tgt = apply_batch_mixing(images, targets, 10, mixup_alpha=0.0, cutmix_alpha=0.0)
        assert torch.equal(out_img, images)
        assert out_tgt.sum(dim=1).allclose(torch.ones(4))
        # Each row should be one-hot
        assert (out_tgt.max(dim=1).values == 1.0).all()

    def test_mixup_targets_sum_to_one(self):
        from src.data import apply_batch_mixing
        images = torch.randn(16, 3, 32, 32)
        targets = torch.randint(0, 10, (16,))
        _, mixed_tgt = apply_batch_mixing(images, targets, 10, mixup_alpha=1.0)
        assert mixed_tgt.sum(dim=1).allclose(torch.ones(16))


class TestGetDatasetInfo:
    """Verify registry-based dataset info lookup."""

    def test_cifar_info(self):
        from src.data import get_dataset_info

        class MockData:
            dataset = 'cifar'
        class MockConfig:
            data = MockData()

        info = get_dataset_info(MockConfig())
        assert info['num_classes'] == 10
        assert info['in_channels'] == 3
        assert info['image_size'] == 32
        assert len(info['classes']) == 10

    def test_mnist_info(self):
        from src.data import get_dataset_info

        class MockData:
            dataset = 'mnist'
        class MockConfig:
            data = MockData()

        info = get_dataset_info(MockConfig())
        assert info['num_classes'] == 10
        assert info['in_channels'] == 1
        assert info['image_size'] == 28

    def test_unknown_dataset_raises(self):
        from src.data import get_dataset_info

        class MockData:
            dataset = 'imagenet'
        class MockConfig:
            data = MockData()

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info(MockConfig())


class TestDinoNumLayers:
    """Verify DINO layer count lookup."""

    def test_layer_lookup(self):
        from src.models.teachers import DINO_NUM_LAYERS
        assert DINO_NUM_LAYERS['dinov2_vits14'] == 12
        assert DINO_NUM_LAYERS['dinov2_vitb14'] == 12
        assert DINO_NUM_LAYERS['dinov2_vitl14'] == 24
        assert DINO_NUM_LAYERS['dinov2_vitg14'] == 40

    def test_all_embed_dims_have_layers(self):
        from src.models.teachers import DINO_NUM_LAYERS, DINO_EMBED_DIMS
        for model_name in DINO_EMBED_DIMS:
            assert model_name in DINO_NUM_LAYERS, f"Missing layer count for {model_name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
