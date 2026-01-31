"""
Loss landscape geometry analysis using PyHessian.

Computes:
- Trace: Sum of eigenvalues (curvature/sharpness measure)
- Top eigenvalues: Largest eigenvalues (dominant curvature directions)
- Eigenvalue density: Distribution of curvature

Expected results:
- CNN-distilled: High trace (sharp landscape, poor generalization)
- DINO-distilled: Low trace (flat landscape, good generalization)

Reference: Ghorbani et al., "Spaced Knowledge Distillation", NeurIPS 2020.

Requires: pip install git+https://github.com/amirgholami/PyHessian.git
"""

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

try:
    from pyhessian import hessian
except ImportError:
    hessian = None


class HessianAnalyzer:
    """
    Loss landscape curvature analysis using PyHessian.

    Note: Disable torch.compile before running Hessian analysis (double_backward issues).
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        num_samples: int = 1024,
        batch_size: int = 64,
    ):
        """
        Args:
            model: Trained model to analyze
            criterion: Loss function (e.g., CrossEntropyLoss)
            device: CUDA device
            num_samples: Number of samples for Hessian estimation
            batch_size: Batch size for Hessian computation
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size

        if hasattr(model, '_orig_mod'):
            self.model = model._orig_mod

        self.model.eval()
        self.model.to(device)

    def _prepare_data(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect samples from dataloader for Hessian computation."""
        inputs_list = []
        targets_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[-1]  # Handle dual-augment datasets
            else:
                inputs, targets = batch

            inputs_list.append(inputs)
            targets_list.append(targets)
            count += inputs.size(0)

            if count >= self.num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:self.num_samples]
        targets = torch.cat(targets_list, dim=0)[:self.num_samples]

        return inputs.to(self.device), targets.to(self.device)

    def compute_trace(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute Hessian trace using Hutchinson's method.

        Returns:
            Dict with 'trace', 'trace_std' (standard deviation across iterations)
        """
        if hessian is None:
            raise ImportError(
                "pyhessian required for Hessian analysis. "
                "Install with: pip install git+https://github.com/amirgholami/PyHessian.git"
            )

        inputs, targets = self._prepare_data(dataloader)

        # Create Hessian computer
        hessian_comp = hessian(
            self.model,
            self.criterion,
            data=(inputs, targets),
            cuda=self.device.type == 'cuda'
        )

        # Compute trace using Hutchinson's estimator
        trace, trace_std = hessian_comp.trace(maxIter=50, tol=1e-3)

        return {
            'trace': float(np.mean(trace)),
            'trace_std': float(trace_std) if trace_std is not None else 0.0,
            'num_samples': len(inputs)
        }

    def compute_top_eigenvalues(
        self,
        dataloader: DataLoader,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Compute top eigenvalues of the Hessian.

        Args:
            dataloader: Data for Hessian computation
            top_n: Number of top eigenvalues to compute

        Returns:
            Dict with 'eigenvalues', 'eigenvalue_ratio' (max/min ratio)
        """
        if hessian is None:
            raise ImportError(
                "pyhessian required for Hessian analysis. "
                "Install with: pip install git+https://github.com/amirgholami/PyHessian.git"
            )

        inputs, targets = self._prepare_data(dataloader)

        hessian_comp = hessian(
            self.model,
            self.criterion,
            data=(inputs, targets),
            cuda=self.device.type == 'cuda'
        )

        # Compute top eigenvalues using power iteration
        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=100, tol=1e-4, top_n=top_n)

        eigenvalues = [float(e) for e in top_eigenvalues]
        ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10) if len(eigenvalues) > 1 else 1.0

        return {
            'eigenvalues': eigenvalues,
            'max_eigenvalue': eigenvalues[0] if eigenvalues else 0.0,
            'eigenvalue_ratio': ratio,
            'num_samples': len(inputs)
        }

    def run_full_analysis(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run complete Hessian analysis."""
        results = {}

        # Trace
        trace_results = self.compute_trace(dataloader)
        results.update(trace_results)

        # Top eigenvalues
        eigen_results = self.compute_top_eigenvalues(dataloader)
        results['eigenvalues'] = eigen_results['eigenvalues']
        results['max_eigenvalue'] = eigen_results['max_eigenvalue']
        results['eigenvalue_ratio'] = eigen_results['eigenvalue_ratio']

        return results


__all__ = ['HessianAnalyzer']
