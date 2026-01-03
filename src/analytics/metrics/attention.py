"""
Attention distance analysis for Vision Transformers.

Computes the mean distance between query and attended key positions
weighted by attention scores. Higher distance = longer-range dependencies.

Expected results:
- CNN-distilled: Short distances (local attention patterns like CNNs)
- DINO-distilled: Long distances (global semantic attention)

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
"""

import logging
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class AttentionDistanceAnalyzer:
    """
    Mean attention distance analysis for Vision Transformers.

    Computes the mean distance between query and attended key positions
    weighted by attention scores.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: int = 32,
        patch_size: int = 4,
    ):
        """
        Args:
            model: ViT model with attention weight extraction capability
            device: CUDA device
            img_size: Input image size
            patch_size: Patch size for tokenization
        """
        self.model = model
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size

        # Compute patch grid
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Precompute distance matrix for patches
        self.distance_matrix = self._compute_distance_matrix()

        self.model.eval()
        self.model.to(device)

    def _compute_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances between patch positions."""
        # Create grid of patch centers
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid_size),
            torch.arange(self.grid_size),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).float()

        # Compute pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (N, N)

        return dist.to(self.device)

    def _extract_attention_weights(
        self,
        inputs: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract attention weights from all layers.

        Args:
            inputs: Input images (B, C, H, W)

        Returns:
            Dict mapping layer_idx to attention weights (B, num_heads, N, N)
        """
        attention_weights = {}

        # Check if model has get_attention_weights method
        if hasattr(self.model, 'get_attention_weights'):
            attention_weights = self.model.get_attention_weights(inputs)
        else:
            # Fallback: Use hooks to capture attention
            hooks = []
            captured = {}

            def make_hook(layer_idx):
                def hook(module, input, output):
                    if hasattr(module, 'attn_weights'):
                        captured[layer_idx] = module.attn_weights
                return hook

            # Register hooks on attention modules
            module = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(module, 'blocks'):
                for i, block in enumerate(module.blocks):
                    if hasattr(block, 'attn'):
                        hook = block.attn.register_forward_hook(make_hook(i))
                        hooks.append(hook)

            # Forward pass
            with torch.no_grad():
                _ = self.model(inputs)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            attention_weights = captured

        return attention_weights

    def compute_mean_attention_distance(
        self,
        dataloader: DataLoader,
        num_samples: int = 512
    ) -> Dict[str, Any]:
        """
        Compute mean attention distance per layer.

        Args:
            dataloader: Data for attention analysis
            num_samples: Number of samples to analyze

        Returns:
            Dict with per-layer distances and overall mean
        """
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

        logger.info(f"Computing mean attention distance on {len(inputs)} samples...")

        # Get attention weights
        attention_weights = self._extract_attention_weights(inputs)

        if not attention_weights:
            logger.warning("No attention weights captured. Model may not support attention extraction.")
            return {'error': 'No attention weights captured'}

        # Compute mean distance per layer
        layer_distances = {}

        for layer_idx, attn in attention_weights.items():
            # attn: (B, num_heads, N, N) - includes special tokens
            num_special = attn.shape[-1] - self.num_patches
            if num_special > 0:
                attn = attn[:, :, num_special:, num_special:]

            # Normalize attention weights
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)

            # Compute weighted mean distance
            distances = self.distance_matrix.unsqueeze(0).unsqueeze(0)
            weighted_dist = (attn * distances).sum(dim=-1)
            mean_dist = weighted_dist.mean().item()

            layer_distances[f'layer_{layer_idx}'] = mean_dist

        # Overall mean
        overall_mean = np.mean(list(layer_distances.values()))

        result = {
            'layer_distances': layer_distances,
            'mean_distance': overall_mean,
            'num_layers': len(layer_distances),
            'num_samples': len(inputs)
        }

        logger.info(f"Mean attention distance: {overall_mean:.4f}")
        for layer, dist in layer_distances.items():
            logger.info(f"  {layer}: {dist:.4f}")

        return result

    def compute_head_statistics(
        self,
        attn_weights: torch.Tensor,
        grid_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive statistics for every attention head.

        Handles DeiT models with distillation token (2 special tokens) automatically.

        Args:
            attn_weights: (B, H, N_total, N_total) where N_total = N_patches + num_special_tokens
            grid_size: Spatial grid size (e.g., 8 for 32x32 image with patch_size=4)

        Returns:
            dict: {
                'mean_distance': (H,),
                'entropy': (H,),
                'cls_dispersion': (H,),
                'cls_self_attn': (H,)
            }
        """
        B, H, N_total, _ = attn_weights.shape

        # Infer grid_size from attention shape if not provided
        if grid_size is None:
            for gs in [8, 7, 14, 16]:
                if gs ** 2 < N_total:
                    grid_size = gs
                    break
            if grid_size is None:
                grid_size = int(np.sqrt(N_total - 1))

        # Calculate number of patches and special tokens
        N_patches = grid_size ** 2
        num_special = N_total - N_patches

        if num_special < 1:
            logger.warning(f"Invalid attention shape: N_total={N_total}, grid_size={grid_size}")
            num_special = 1
        elif num_special > 2:
            logger.warning(f"Unusual number of special tokens: {num_special}")

        # 1. Patch-to-Patch Mean Attention Distance
        patch_attn = attn_weights[:, :, num_special:, num_special:]
        patch_attn = patch_attn / (patch_attn.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute distance matrix
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=attn_weights.device, dtype=torch.float32),
            torch.arange(grid_size, device=attn_weights.device, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)

        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1))
        dist_matrix = dist_matrix.unsqueeze(0).unsqueeze(0)

        weighted_dist = (patch_attn * dist_matrix).sum(dim=-1)
        mean_distance = weighted_dist.mean(dim=(0, 2)).cpu().numpy()

        # 2. Attention Entropy
        attn_flat = patch_attn.mean(dim=2)
        entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum(dim=-1)
        entropy = entropy.mean(dim=0).cpu().numpy()

        # 3. CLS Spatial Dispersion
        cls_attn = attn_weights[:, :, 0, num_special:]
        cls_attn = cls_attn.reshape(B, H, grid_size, grid_size)
        cls_attn = cls_attn / (cls_attn.sum(dim=(-2, -1), keepdim=True) + 1e-10)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=attn_weights.device, dtype=torch.float32),
            torch.arange(grid_size, device=attn_weights.device, dtype=torch.float32),
            indexing='ij'
        )
        center_y = (cls_attn * grid_y).sum(dim=(-2, -1))
        center_x = (cls_attn * grid_x).sum(dim=(-2, -1))

        var_y = (cls_attn * (grid_y - center_y.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-2, -1))
        var_x = (cls_attn * (grid_x - center_x.unsqueeze(-1).unsqueeze(-1))**2).sum(dim=(-2, -1))
        dispersion = (var_y + var_x).mean(dim=0).cpu().numpy()

        # 4. CLS Self-Attention Score
        cls_self_attn = attn_weights[:, :, 0, 0].mean(dim=0).cpu().numpy()

        return {
            'mean_distance': mean_distance,
            'entropy': entropy,
            'cls_dispersion': dispersion,
            'cls_self_attn': cls_self_attn
        }


__all__ = ['AttentionDistanceAnalyzer']
