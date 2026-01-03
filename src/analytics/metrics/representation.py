"""
Centered Kernel Alignment (CKA) similarity analyzer.

Computes CKA similarity between:
- Teacher and student intermediate representations
- Different layers within the same model

Produces heatmaps showing semantic alignment quality.

Expected results:
- Good distillation: Clear diagonal pattern (layer-to-layer alignment)
- Poor distillation: Scattered or weak correlations

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019.
"""

import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CKAAnalyzer:
    """
    Centered Kernel Alignment (CKA) similarity analyzer.
    """

    def __init__(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module],
        device: torch.device,
        kernel_type: str = 'linear',
    ):
        """
        Args:
            model1: First model (e.g., student)
            model2: Second model (e.g., teacher), or None for self-CKA
            device: CUDA device
            kernel_type: 'linear' or 'rbf' kernel
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device
        self.kernel_type = kernel_type

        self.model1.eval()
        self.model1.to(device)
        if model2 is not None:
            self.model2.eval()
            self.model2.to(device)

    def _compute_gram(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gram matrix from features."""
        if self.kernel_type == 'linear':
            return X @ X.T
        else:  # rbf
            sq_dist = torch.cdist(X, X, p=2) ** 2
            sigma = torch.median(sq_dist).item() + 1e-10
            return torch.exp(-sq_dist / (2 * sigma))

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Center gram matrix."""
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def _hsic(self, K1: torch.Tensor, K2: torch.Tensor) -> float:
        """Compute HSIC between two gram matrices."""
        n = K1.shape[0]
        return float((K1 * K2).sum() / ((n - 1) ** 2))

    def compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute CKA between two feature matrices.

        Args:
            X: Features from model 1 (N, D1)
            Y: Features from model 2 (N, D2)

        Returns:
            CKA similarity value in [0, 1]
        """
        # Compute and center gram matrices
        K_X = self._center_gram(self._compute_gram(X))
        K_Y = self._center_gram(self._compute_gram(Y))

        # Compute CKA
        hsic_XY = self._hsic(K_X, K_Y)
        hsic_XX = self._hsic(K_X, K_X)
        hsic_YY = self._hsic(K_Y, K_Y)

        cka = hsic_XY / (np.sqrt(hsic_XX * hsic_YY) + 1e-10)
        return float(cka)

    def _extract_layer_features(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_indices: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Extract features from specified layers."""
        features = {}

        # Try forward_with_intermediates if available (for DeiT)
        if hasattr(model, 'forward_with_intermediates'):
            with torch.no_grad():
                results = model.forward_with_intermediates(inputs, layer_indices=layer_indices)
                for idx in layer_indices:
                    if idx in results['intermediates']:
                        feat = results['intermediates'][idx]
                        features[idx] = feat.mean(dim=1) if feat.dim() == 3 else feat
        else:
            # Use hooks for generic models
            hooks = []
            captured = {}

            def make_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    if output.dim() == 4:  # (B, C, H, W)
                        captured[idx] = output.mean(dim=(2, 3))
                    elif output.dim() == 3:  # (B, N, D)
                        captured[idx] = output.mean(dim=1)
                    else:
                        captured[idx] = output
                return hook

            # Get model layers
            module = model.module if hasattr(model, 'module') else model
            if hasattr(module, 'blocks'):
                for idx in layer_indices:
                    if idx < len(module.blocks):
                        hook = module.blocks[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)
            elif hasattr(module, 'features'):
                for idx in layer_indices:
                    if idx < len(module.features):
                        hook = module.features[idx].register_forward_hook(make_hook(idx))
                        hooks.append(hook)

            with torch.no_grad():
                _ = model(inputs)

            for hook in hooks:
                hook.remove()

            features = captured

        return features

    def compute_layer_cka_matrix(
        self,
        dataloader: DataLoader,
        layer_indices1: List[int],
        layer_indices2: Optional[List[int]] = None,
        num_samples: int = 512
    ) -> Dict[str, Any]:
        """
        Compute CKA matrix between layers of two models.

        Args:
            dataloader: Data for feature extraction
            layer_indices1: Layer indices for model1
            layer_indices2: Layer indices for model2 (defaults to layer_indices1)
            num_samples: Number of samples to use

        Returns:
            Dict with 'cka_matrix' (2D array) and layer indices
        """
        if layer_indices2 is None:
            layer_indices2 = layer_indices1 if self.model2 is None else layer_indices1

        # Collect samples
        inputs_list = []
        count = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs_list.append(inputs)
            count += inputs.size(0)
            if count >= num_samples:
                break

        inputs = torch.cat(inputs_list, dim=0)[:num_samples].to(self.device)

        logger.info(f"Computing CKA matrix on {len(inputs)} samples...")

        # Extract features
        features1 = self._extract_layer_features(self.model1, inputs, layer_indices1)
        if self.model2 is not None:
            features2 = self._extract_layer_features(self.model2, inputs, layer_indices2)
        else:
            features2 = features1
            layer_indices2 = layer_indices1

        # Compute CKA matrix
        n_layers1 = len(layer_indices1)
        n_layers2 = len(layer_indices2)
        cka_matrix = np.zeros((n_layers1, n_layers2))

        for i, idx1 in enumerate(tqdm(layer_indices1, desc="Computing CKA")):
            if idx1 not in features1:
                continue
            feat1 = features1[idx1].reshape(len(inputs), -1)

            for j, idx2 in enumerate(layer_indices2):
                if idx2 not in features2:
                    continue
                feat2 = features2[idx2].reshape(len(inputs), -1)

                cka_matrix[i, j] = self.compute_cka(feat1, feat2)

        result = {
            'cka_matrix': cka_matrix.tolist(),
            'layer_indices_1': layer_indices1,
            'layer_indices_2': layer_indices2,
            'num_samples': len(inputs)
        }

        logger.info(f"CKA matrix computed: {n_layers1}x{n_layers2}")
        return result


__all__ = ['CKAAnalyzer']
