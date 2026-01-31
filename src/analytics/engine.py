"""
Analytics runner for model analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import HessianAnalyzer, AttentionDistanceAnalyzer, CKAAnalyzer


def run_analytics(
    model: nn.Module,
    config: Any,
    device: torch.device,
    dataloader: DataLoader,
    metrics: List[str] = None,
    teacher_model: Optional[nn.Module] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run specified analytics on a model.

    Args:
        model: Trained model to analyze
        config: Configuration object
        device: CUDA device
        dataloader: Data for analysis
        metrics: List of metrics to compute ('hessian', 'attention', 'cka')
        teacher_model: Optional teacher for CKA comparison
        save_path: Optional path to save results

    Returns:
        Dict with all computed metrics
    """
    if metrics is None:
        metrics = ['hessian', 'attention', 'cka']

    if hasattr(model, '_orig_mod'):
        model = model._orig_mod

    results = {}

    if 'hessian' in metrics:
        try:
            criterion = nn.CrossEntropyLoss()
            analyzer = HessianAnalyzer(
                model, criterion, device,
                num_samples=getattr(config, 'hessian_samples', 1024)
            )
            results['hessian'] = analyzer.run_full_analysis(dataloader)
        except Exception as e:
            results['hessian'] = {'error': str(e)}

    if 'attention' in metrics:
        try:
            img_size = getattr(config.vit, 'img_size', 32) if hasattr(config, 'vit') else 32
            patch_size = getattr(config.vit, 'patch_size', 4) if hasattr(config, 'vit') else 4
            analyzer = AttentionDistanceAnalyzer(
                model, device,
                img_size=img_size, patch_size=patch_size
            )
            results['attention'] = analyzer.compute_mean_attention_distance(dataloader)
        except Exception as e:
            results['attention'] = {'error': str(e)}

    if 'cka' in metrics:
        try:
            analyzer = CKAAnalyzer(
                model, teacher_model, device,
                kernel_type=getattr(config, 'cka_kernel', 'linear')
            )
            layer_indices = list(range(12))
            results['cka'] = analyzer.compute_layer_cka_matrix(
                dataloader, layer_indices
            )
        except Exception as e:
            results['cka'] = {'error': str(e)}

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"analytics saved={save_path}")

    return results


__all__ = ['run_analytics']
