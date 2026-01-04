"""
Tests for hierarchical config loading and merging.

Verifies that:
1. Global defaults are loaded from configs/defaults.yaml
2. Specific config values override defaults
3. Deep merging works correctly for nested dicts
"""
import pytest
import tempfile
import yaml
from pathlib import Path


class TestConfigMerging:
    """Tests for two-layer config merging."""

    def test_load_config_from_hierarchical_path(self):
        """Verify config loads from hierarchical path."""
        from src.core import ConfigManager

        config = ConfigManager.load_config('configs/cifar10/baselines/deit_tiny.yaml')

        # Check that config loaded correctly
        assert config.experiment_name == 'deit_baseline_cifar'
        assert config.data.dataset == 'cifar'
        assert config.data.batch_size == 2048

    def test_defaults_are_merged(self):
        """Verify defaults from defaults.yaml are merged."""
        from src.core import ConfigManager

        config = ConfigManager.load_config('configs/cifar10/baselines/deit_tiny.yaml')

        # These should come from defaults.yaml
        assert config.training.use_bf16 is True
        assert config.training.use_compile is True
        assert config.seed == 42

    def test_specific_overrides_defaults(self):
        """Verify specific config values override defaults."""
        from src.core import ConfigManager

        config = ConfigManager.load_config('configs/mnist/adaptive_cnn.yaml')

        # These are overridden in mnist config
        assert config.data.batch_size == 64
        assert config.data.num_workers == 4
        # use_compile should be False for MNIST config
        assert config.training.use_compile is False

    def test_deep_merge_nested_dicts(self):
        """Verify deep merge works for nested dictionaries."""
        from src.core.config import ConfigManager

        # Test the internal _deep_merge method
        base = {
            'data': {'batch_size': 64, 'num_workers': 4},
            'training': {'lr': 0.001, 'epochs': 100}
        }
        override = {
            'data': {'batch_size': 128},
            'training': {'lr': 0.01}
        }

        merged = ConfigManager._deep_merge(base, override)

        # Check that override takes precedence
        assert merged['data']['batch_size'] == 128
        assert merged['training']['lr'] == 0.01
        # Check that non-overridden values are preserved
        assert merged['data']['num_workers'] == 4
        assert merged['training']['epochs'] == 100

    def test_load_without_defaults(self):
        """Verify loading without defaults merging."""
        from src.core import ConfigManager

        # Load without merging defaults
        config = ConfigManager.load_config(
            'configs/cifar10/baselines/deit_tiny.yaml',
            merge_defaults=False
        )

        # Should still work, just won't have defaults merged
        assert config.experiment_name == 'deit_baseline_cifar'


class TestConfigValidation:
    """Tests for config validation."""

    def test_valid_config_passes_validation(self):
        """Verify valid config passes validation."""
        from src.core import ConfigManager

        # Should not raise
        config = ConfigManager.load_config('configs/cifar10/baselines/deit_tiny.yaml')
        assert config is not None

    def test_invalid_dataset_fails(self):
        """Verify invalid dataset raises error."""
        from src.core import Config, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
        from src.core.config import ConfigManager

        config = Config(
            data=DataConfig(dataset='invalid_dataset'),
            model=ModelConfig(),
            training=TrainingConfig(),
            logging=LoggingConfig()
        )

        with pytest.raises(ValueError, match="Invalid dataset"):
            ConfigManager.validate_config(config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
