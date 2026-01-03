"""
Analytics runners and the Locality Curse Forensics toolkit.

Provides:
- AnalyticsRunner: Unified runner for all analytics
- LocalityCurseForensics: Complete diagnostic toolkit for the Locality Curse study
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .metrics import HessianAnalyzer, AttentionDistanceAnalyzer, CKAAnalyzer
from .visualization import AnalyticsVisualizer

logger = logging.getLogger(__name__)


class AnalyticsRunner:
    """
    Unified runner for all analytics.

    Usage:
        runner = AnalyticsRunner(model, config, device)
        results = runner.run_all(dataloader, metrics=['hessian', 'attention', 'cka'])
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        teacher_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: Trained model to analyze
            config: Configuration object
            device: CUDA device
            teacher_model: Optional teacher for CKA comparison
        """
        self.model = model
        self.config = config
        self.device = device
        self.teacher_model = teacher_model

        # Unwrap compiled model
        if hasattr(model, '_orig_mod'):
            logger.info("Unwrapping torch.compile model for analytics")
            self.model = model._orig_mod

    def run_all(
        self,
        dataloader: DataLoader,
        metrics: List[str] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run specified analytics.

        Args:
            dataloader: Data for analysis
            metrics: List of metrics to compute ('hessian', 'attention', 'cka')
            save_path: Optional path to save results

        Returns:
            Dict with all computed metrics
        """
        if metrics is None:
            metrics = ['hessian', 'attention', 'cka']

        results = {}

        if 'hessian' in metrics:
            logger.info("=" * 60)
            logger.info("Running Hessian Analysis")
            logger.info("=" * 60)
            try:
                criterion = nn.CrossEntropyLoss()
                analyzer = HessianAnalyzer(
                    self.model, criterion, self.device,
                    num_samples=getattr(self.config, 'hessian_samples', 1024)
                )
                results['hessian'] = analyzer.run_full_analysis(dataloader)
            except Exception as e:
                logger.error(f"Hessian analysis failed: {e}")
                results['hessian'] = {'error': str(e)}

        if 'attention' in metrics:
            logger.info("=" * 60)
            logger.info("Running Attention Distance Analysis")
            logger.info("=" * 60)
            try:
                img_size = getattr(self.config.vit, 'img_size', 32) if hasattr(self.config, 'vit') else 32
                patch_size = getattr(self.config.vit, 'patch_size', 4) if hasattr(self.config, 'vit') else 4
                analyzer = AttentionDistanceAnalyzer(
                    self.model, self.device,
                    img_size=img_size, patch_size=patch_size
                )
                results['attention'] = analyzer.compute_mean_attention_distance(dataloader)
            except Exception as e:
                logger.error(f"Attention analysis failed: {e}")
                results['attention'] = {'error': str(e)}

        if 'cka' in metrics:
            logger.info("=" * 60)
            logger.info("Running CKA Analysis")
            logger.info("=" * 60)
            try:
                analyzer = CKAAnalyzer(
                    self.model, self.teacher_model, self.device,
                    kernel_type=getattr(self.config, 'cka_kernel', 'linear')
                )
                layer_indices = list(range(12))
                results['cka'] = analyzer.compute_layer_cka_matrix(
                    dataloader, layer_indices
                )
            except Exception as e:
                logger.error(f"CKA analysis failed: {e}")
                results['cka'] = {'error': str(e)}

        # Save results
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Analytics results saved to {save_path}")

        return results


class LocalityCurseForensics:
    """
    Unified diagnostic toolkit for proving the Locality Curse hypothesis.

    The Locality Curse: CNN teachers damage ViT students by forcing local
    attention patterns, resulting in poor generalization.

    This class compares student models trained with different teachers.
    It does NOT compare teacher directly (methodologically cleaner).

    Expected three-way comparison:
        CNN-Distilled (Low MAD) < Baseline (Medium MAD) < DINO-Distilled (High MAD)

    The gap between Baseline and CNN-Distilled represents the "active harm"
    done by the CNN teacher.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize forensics toolkit.

        Args:
            device: Device for computations ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.visualizer = AnalyticsVisualizer()

    def run_full_forensics(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str,
        output_dir: str,
        img_size: int = 32,
        patch_size: int = 4,
        num_samples: int = 512
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic suite on a single model.

        Args:
            model: ViT model to analyze
            dataloader: Validation dataloader
            model_name: Name for this model (used in outputs)
            output_dir: Directory to save results
            img_size: Input image size
            patch_size: ViT patch size
            num_samples: Number of samples for analysis

        Returns:
            Dict with all forensics metrics
        """
        results = {'model_name': model_name}
        os.makedirs(output_dir, exist_ok=True)

        model = model.to(self.device)
        model.eval()

        # Collect samples
        inputs_list = []
        targets_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = torch.zeros(inputs.size(0), dtype=torch.long)

            inputs_list.append(inputs)
            targets_list.append(targets)
            count += inputs.size(0)
            if count >= num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:num_samples].to(self.device)
        targets = torch.cat(targets_list, dim=0)[:num_samples].to(self.device)

        # 1. Hessian Trace (Generalization Proxy)
        logger.info(f"[{model_name}] Computing Hessian trace...")
        try:
            hessian_analyzer = HessianAnalyzer(model, nn.CrossEntropyLoss(), self.device)
            mini_dataset = torch.utils.data.TensorDataset(inputs[:100], targets[:100])
            mini_loader = DataLoader(mini_dataset, batch_size=32)
            trace_results = hessian_analyzer.compute_trace(mini_loader)
            results['hessian_trace'] = trace_results.get('trace', float('nan'))
        except Exception as e:
            logger.warning(f"Hessian computation failed: {e}")
            results['hessian_trace'] = float('nan')

        # 2. Attention Metrics (The Curse)
        logger.info(f"[{model_name}] Computing attention metrics...")
        attn_analyzer = AttentionDistanceAnalyzer(model, self.device, img_size, patch_size)
        attention_weights = attn_analyzer._extract_attention_weights(inputs)

        if not attention_weights:
            logger.warning(f"[{model_name}] No attention weights captured")
            results['per_layer_stats'] = {}
            results['avg_attention_distance'] = float('nan')
            results['collapsed_heads_ratio'] = float('nan')
            results['avg_entropy'] = float('nan')
            results['avg_cls_dispersion'] = float('nan')
            results['avg_cls_self_attn'] = float('nan')
            results['cls_collapse_ratio'] = float('nan')
        else:
            per_layer_stats = {}
            all_distances = []
            all_entropies = []
            all_dispersions = []
            all_cls_self = []

            grid_size = img_size // patch_size

            for layer_idx, attn in attention_weights.items():
                stats = attn_analyzer.compute_head_statistics(attn, grid_size=grid_size)
                per_layer_stats[layer_idx] = stats
                all_distances.extend(stats['mean_distance'].tolist())
                all_entropies.append(stats['entropy'].mean())
                all_dispersions.append(stats['cls_dispersion'].mean())
                all_cls_self.extend(stats['cls_self_attn'].tolist())

            results['per_layer_stats'] = per_layer_stats
            results['avg_attention_distance'] = float(np.mean(all_distances))
            results['collapsed_heads_ratio'] = float(np.mean(np.array(all_distances) < 1.5))
            results['avg_entropy'] = float(np.mean(all_entropies))
            results['avg_cls_dispersion'] = float(np.mean(all_dispersions))

            # Safety check: CLS self-attention (collapse indicator)
            results['avg_cls_self_attn'] = float(np.mean(all_cls_self))
            results['cls_collapse_ratio'] = float(np.mean(np.array(all_cls_self) > 0.9))

            # Warn if collapse detected
            if results['cls_collapse_ratio'] > 0.5:
                logger.warning(f"[{model_name}] High CLS self-attention detected "
                             f"({results['cls_collapse_ratio']*100:.1f}% of heads). "
                             "This may indicate feature collapse, not locality curse.")

        # Save results
        save_path = f"{output_dir}/{model_name.replace(' ', '_')}_forensics.json"
        with open(save_path, 'w') as f:
            json_results = {}
            for k, v in results.items():
                if k == 'per_layer_stats':
                    json_results[k] = {
                        str(layer): {
                            stat_name: arr.tolist() if hasattr(arr, 'tolist') else arr
                            for stat_name, arr in stats.items()
                        }
                        for layer, stats in v.items()
                    }
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2, default=float)

        logger.info(f"[{model_name}] Forensics complete. Results saved to {save_path}")

        return results

    def compare_models(
        self,
        models_dict: Dict[str, nn.Module],
        dataloader: DataLoader,
        output_dir: str,
        img_size: int = 32,
        patch_size: int = 4,
        num_samples: int = 512
    ) -> Dict[str, Dict]:
        """
        Compare multiple student models and generate visualizations.

        Args:
            models_dict: Dict mapping model names to models
            dataloader: Validation dataloader
            output_dir: Directory to save results
            img_size: Input image size
            patch_size: ViT patch size
            num_samples: Number of samples for analysis

        Returns:
            Dict mapping model names to their forensics results
        """
        all_results = {}

        for name, model in models_dict.items():
            logger.info(f"Running forensics on: {name}")
            results = self.run_full_forensics(
                model, dataloader, name, output_dir,
                img_size, patch_size, num_samples
            )
            all_results[name] = results

        # Generate comparison plots
        logger.info("Generating comparison visualizations...")
        self.visualizer.plot_locality_spectrum(all_results, output_dir)
        self.visualizer.plot_layer_progression(all_results, output_dir)
        self.visualizer.plot_forensics_summary(all_results, output_dir)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, all_results: Dict[str, Dict]):
        """Print formatted summary of forensics results."""
        print("\n" + "="*60)
        print("         LOCALITY CURSE FORENSICS SUMMARY")
        print("="*60)

        for name, r in all_results.items():
            print(f"\n{name}:")
            print(f"  Hessian Trace:        {r.get('hessian_trace', 'N/A'):>10.2f}")
            print(f"  Avg Attention Dist:   {r.get('avg_attention_distance', 'N/A'):>10.3f}")
            print(f"  Collapsed Heads:      {r.get('collapsed_heads_ratio', 0)*100:>10.1f}%")
            print(f"  Avg Entropy:          {r.get('avg_entropy', 'N/A'):>10.3f}")
            print(f"  CLS Dispersion:       {r.get('avg_cls_dispersion', 'N/A'):>10.3f}")
            print(f"  CLS Self-Attention:   {r.get('avg_cls_self_attn', 'N/A'):>10.3f}")

            if r.get('cls_collapse_ratio', 0) > 0.1:
                print(f"  WARNING: {r.get('cls_collapse_ratio', 0)*100:.1f}% heads may have collapsed")

        print("\n" + "="*60)
        print("Expected ordering (Locality Curse hypothesis):")
        print("  CNN-Distilled < Baseline < DINO-Distilled (attention distance)")
        print("  CNN-Distilled > Baseline > DINO-Distilled (Hessian trace)")
        print("="*60 + "\n")


__all__ = ['AnalyticsRunner', 'LocalityCurseForensics']
