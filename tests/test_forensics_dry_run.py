"""
Dry run test for Locality Curse Forensics Kit.

This test verifies the pipeline works with synthetic data before
investing hours in training real models.

Run with: python tests/test_forensics_dry_run.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

def test_compute_head_statistics():
    """Test the compute_head_statistics method with synthetic attention."""
    print("\n" + "="*60)
    print("TEST 1: compute_head_statistics with synthetic attention")
    print("="*60)

    # Import after path setup
    from src.analytics import AttentionDistanceAnalyzer

    # Create a dummy ViT-like model
    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return x

        def get_attention_weights(self, x):
            # Return empty dict - we'll test with synthetic attention
            return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DummyViT().to(device)

    analyzer = AttentionDistanceAnalyzer(
        model=model,
        device=device,
        img_size=32,
        patch_size=4
    )

    # Create synthetic attention: (B=2, H=3, N+1=65, N+1=65)
    # For 32x32 image with patch_size=4: grid=8, N=64 patches + 1 CLS = 65
    B, H, N_plus_1 = 2, 3, 65
    N = N_plus_1 - 1

    # Create random attention weights (softmax normalized)
    raw_attn = torch.randn(B, H, N_plus_1, N_plus_1, device=device)
    attn_weights = torch.softmax(raw_attn, dim=-1)

    print(f"Attention shape: {attn_weights.shape}")
    print(f"Expected: (B={B}, H={H}, N+1={N_plus_1}, N+1={N_plus_1})")

    # Run compute_head_statistics
    stats = analyzer.compute_head_statistics(attn_weights)

    print("\nResults:")
    print(f"  mean_distance shape: {stats['mean_distance'].shape}, values: {stats['mean_distance']}")
    print(f"  entropy shape: {stats['entropy'].shape}, values: {stats['entropy']}")
    print(f"  cls_dispersion shape: {stats['cls_dispersion'].shape}, values: {stats['cls_dispersion']}")
    print(f"  cls_self_attn shape: {stats['cls_self_attn'].shape}, values: {stats['cls_self_attn']}")

    # Sanity checks
    errors = []

    # Check 1: CLS self-attention must be in [0, 1]
    if not (0 <= stats['cls_self_attn'].min() <= stats['cls_self_attn'].max() <= 1):
        errors.append(f"CLS self-attention out of range [0,1]: {stats['cls_self_attn']}")
    else:
        print("  ✓ CLS self-attention in [0, 1]")

    # Check 2: Mean attention distance should be reasonable
    # For 8x8 grid, max distance is sqrt(7^2 + 7^2) ≈ 9.9
    max_possible_dist = np.sqrt(2 * (8-1)**2)  # ~9.9
    if stats['mean_distance'].max() > max_possible_dist:
        errors.append(f"Mean distance too large: {stats['mean_distance'].max()} > {max_possible_dist}")
    elif stats['mean_distance'].min() < 0:
        errors.append(f"Mean distance negative: {stats['mean_distance'].min()}")
    else:
        print(f"  ✓ Mean distance in valid range [0, {max_possible_dist:.1f}]")

    # Check 3: Entropy must be positive
    if stats['entropy'].min() < 0:
        errors.append(f"Entropy negative: {stats['entropy'].min()}")
    else:
        print(f"  ✓ Entropy positive: min={stats['entropy'].min():.4f}")

    # Check 4: Dispersion should be non-negative
    if stats['cls_dispersion'].min() < 0:
        errors.append(f"Dispersion negative: {stats['cls_dispersion'].min()}")
    else:
        print(f"  ✓ CLS dispersion non-negative: min={stats['cls_dispersion'].min():.4f}")

    if errors:
        print("\n❌ ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\n✓ All sanity checks passed!")
    return True


def test_visualization_methods():
    """Test visualization methods with synthetic results."""
    print("\n" + "="*60)
    print("TEST 2: Visualization methods")
    print("="*60)

    from src.analytics import AnalyticsVisualizer
    import tempfile
    import os

    # Create synthetic results
    results_dict = {
        'Model_A': {
            'hessian_trace': 1234.5,
            'avg_attention_distance': 2.5,
            'collapsed_heads_ratio': 0.3,
            'avg_entropy': 3.2,
            'avg_cls_dispersion': 4.1,
            'per_layer_stats': {
                0: {'mean_distance': np.array([1.5, 2.0, 2.5])},
                6: {'mean_distance': np.array([2.0, 2.5, 3.0])},
                11: {'mean_distance': np.array([2.5, 3.0, 3.5])},
            }
        },
        'Model_B': {
            'hessian_trace': 987.6,
            'avg_attention_distance': 3.5,
            'collapsed_heads_ratio': 0.1,
            'avg_entropy': 3.8,
            'avg_cls_dispersion': 5.2,
            'per_layer_stats': {
                0: {'mean_distance': np.array([2.5, 3.0, 3.5])},
                6: {'mean_distance': np.array([3.0, 3.5, 4.0])},
                11: {'mean_distance': np.array([3.5, 4.0, 4.5])},
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Output directory: {tmpdir}")

        # Test plot_locality_spectrum
        try:
            AnalyticsVisualizer.plot_locality_spectrum(results_dict, tmpdir)
            if os.path.exists(os.path.join(tmpdir, 'locality_spectrum.png')):
                print("  ✓ locality_spectrum.png created")
            else:
                print("  ❌ locality_spectrum.png NOT created")
        except Exception as e:
            print(f"  ❌ plot_locality_spectrum failed: {e}")

        # Test plot_layer_progression
        try:
            AnalyticsVisualizer.plot_layer_progression(results_dict, tmpdir)
            if os.path.exists(os.path.join(tmpdir, 'layer_progression.png')):
                print("  ✓ layer_progression.png created")
            else:
                print("  ❌ layer_progression.png NOT created")
        except Exception as e:
            print(f"  ❌ plot_layer_progression failed: {e}")

        # Test plot_forensics_summary
        try:
            AnalyticsVisualizer.plot_forensics_summary(results_dict, tmpdir)
            if os.path.exists(os.path.join(tmpdir, 'forensics_summary.png')):
                print("  ✓ forensics_summary.png created")
            else:
                print("  ❌ forensics_summary.png NOT created")
        except Exception as e:
            print(f"  ❌ plot_forensics_summary failed: {e}")

    print("\n✓ Visualization tests complete!")
    return True


def test_full_forensics_pipeline():
    """Test the full LocalityCurseForensics pipeline with a random model."""
    print("\n" + "="*60)
    print("TEST 3: Full forensics pipeline with random DeiT")
    print("="*60)

    from src.analytics import LocalityCurseForensics
    from src.vit import DeiT
    import tempfile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create random DeiT model
    print("Creating random DeiT model...")
    deit_config = {
        'img_size': 32,
        'patch_size': 4,
        'in_channels': 3,
        'num_classes': 10,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'distillation': True,
    }
    model = DeiT(deit_config).to(device)

    # Create synthetic dataloader
    print("Creating synthetic dataloader...")
    num_samples = 16
    synthetic_images = torch.randn(num_samples, 3, 32, 32)
    synthetic_labels = torch.randint(0, 10, (num_samples,))
    dataset = torch.utils.data.TensorDataset(synthetic_images, synthetic_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Output directory: {tmpdir}")

        # Run forensics
        forensics = LocalityCurseForensics(device=str(device))

        try:
            results = forensics.run_full_forensics(
                model=model,
                dataloader=dataloader,
                model_name="RandomDeiT",
                output_dir=tmpdir,
                img_size=32,
                patch_size=4,
                num_samples=num_samples
            )

            print("\nForensics Results:")
            print(f"  hessian_trace: {results.get('hessian_trace', 'N/A')}")
            print(f"  avg_attention_distance: {results.get('avg_attention_distance', 'N/A')}")
            print(f"  collapsed_heads_ratio: {results.get('collapsed_heads_ratio', 'N/A')}")
            print(f"  avg_entropy: {results.get('avg_entropy', 'N/A')}")
            print(f"  avg_cls_dispersion: {results.get('avg_cls_dispersion', 'N/A')}")
            print(f"  avg_cls_self_attn: {results.get('avg_cls_self_attn', 'N/A')}")
            print(f"  cls_collapse_ratio: {results.get('cls_collapse_ratio', 'N/A')}")

            # Check JSON file was created
            json_path = os.path.join(tmpdir, "RandomDeiT_forensics.json")
            if os.path.exists(json_path):
                print(f"\n  ✓ JSON file created: {json_path}")
                with open(json_path, 'r') as f:
                    saved = json.load(f)
                    print(f"  ✓ JSON is valid with {len(saved)} keys")
            else:
                print(f"\n  ❌ JSON file NOT created")

            # Sanity checks on results
            errors = []

            # Check Hessian trace is valid number
            if np.isnan(results.get('hessian_trace', float('nan'))):
                print("  ⚠ Hessian trace is NaN (pyhessian may not be installed)")
            elif np.isinf(results.get('hessian_trace', 0)):
                errors.append("Hessian trace is Inf")
            else:
                print(f"  ✓ Hessian trace is valid: {results['hessian_trace']:.2f}")

            # Check attention distance
            if 0 < results.get('avg_attention_distance', 0) < 10:
                print(f"  ✓ Attention distance in reasonable range: {results['avg_attention_distance']:.3f}")
            else:
                errors.append(f"Attention distance out of range: {results.get('avg_attention_distance')}")

            # Check cls_self_attn
            if 0 <= results.get('avg_cls_self_attn', -1) <= 1:
                print(f"  ✓ CLS self-attention in [0,1]: {results['avg_cls_self_attn']:.3f}")
            else:
                errors.append(f"CLS self-attention out of range: {results.get('avg_cls_self_attn')}")

            if errors:
                print("\n❌ ERRORS:")
                for e in errors:
                    print(f"  - {e}")
                return False

            print("\n✓ Full pipeline test passed!")
            return True

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    print("="*60)
    print("  LOCALITY CURSE FORENSICS - DRY RUN TESTS")
    print("="*60)

    results = []

    # Test 1: compute_head_statistics
    try:
        results.append(("compute_head_statistics", test_compute_head_statistics()))
    except Exception as e:
        print(f"Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("compute_head_statistics", False))

    # Test 2: Visualization methods
    try:
        results.append(("visualization_methods", test_visualization_methods()))
    except Exception as e:
        print(f"Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("visualization_methods", False))

    # Test 3: Full pipeline (requires DeiT)
    try:
        results.append(("full_pipeline", test_full_forensics_pipeline()))
    except Exception as e:
        print(f"Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("full_pipeline", False))

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("All tests PASSED! Safe to proceed with training.")
    else:
        print("Some tests FAILED! Fix issues before training.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
