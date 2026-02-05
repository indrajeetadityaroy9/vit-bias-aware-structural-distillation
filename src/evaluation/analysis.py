"""
Analytics runner for model analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.analyzers import HessianAnalyzer, AttentionDistanceAnalyzer, CKAAnalyzer


def run_analytics(
    model: nn.Module,
    config: Any,
    device: torch.device,
    dataloader: DataLoader,
    metrics: List[str] = None,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run specified analytics on a model.

    Args:
        model: Trained model to analyze
        config: Configuration object (expects config.vit for ViT params)
        device: CUDA device
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
        criterion = nn.CrossEntropyLoss()
        analyzer = HessianAnalyzer(model, criterion, device, num_samples=1024)
        results['hessian'] = analyzer.run_full_analysis(dataloader)

    if 'attention' in metrics:
        analyzer = AttentionDistanceAnalyzer(
            model, device,
            img_size=config.vit.img_size, patch_size=config.vit.patch_size
        )
        results['attention'] = analyzer.compute_mean_attention_distance(dataloader)

    if 'cka' in metrics:
        analyzer = CKAAnalyzer(model, None, device, kernel_type='linear')
        layer_indices = list(range(12))
        results['cka'] = analyzer.compute_layer_cka_matrix(dataloader, layer_indices)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"analytics saved={save_path}")

    return results
