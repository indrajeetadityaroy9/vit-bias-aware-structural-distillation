"""
Structural distillation losses using representational similarity.

Implements:
- CKALoss: Centered Kernel Alignment for structural similarity
- GramMatrixLoss: Frobenius norm on gram matrices (ablation baseline)
- LayerWiseStructuralLoss: Multi-layer structural distillation

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019 (CKA paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses.token import ProjectionHead


class CKALoss(nn.Module):
    """
    Centered Kernel Alignment (CKA) Loss for structural distillation.

    CKA measures representational similarity by comparing gram matrices of
    activations. This captures structural (relational) knowledge rather than
    just token-wise matching.

    CKA = HSIC(K_S, K_T) / sqrt(HSIC(K_S, K_S) * HSIC(K_T, K_T))
    Loss = 1 - CKA

    Reference: Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
    """

    def __init__(self, kernel_type='linear', eps=1e-8, normalize_features=True):
        """
        Args:
            kernel_type: 'linear' or 'rbf' kernel for gram matrix computation
            eps: Small constant for numerical stability
            normalize_features: If True, L2 normalize features before gram computation
                              (recommended for numerical stability across batch sizes)
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.eps = eps
        self.normalize_features = normalize_features

    def _compute_gram(self, x):
        """
        Compute gram matrix (kernel matrix) from feature representations.

        Args:
            x: Feature tensor (B, N, D) - batch of token sequences

        Returns:
            K: Gram matrix (B, N, N)
        """
        # Optional L2 normalization for numerical stability
        if self.normalize_features:
            x = F.normalize(x, p=2, dim=-1)

        if self.kernel_type == 'linear':
            # Linear kernel: K = X @ X^T
            K = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
        else:  # rbf
            # RBF kernel: K_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            # Use median heuristic for sigma
            x_norm = (x ** 2).sum(-1, keepdim=True)  # (B, N, 1)
            dist_sq = x_norm + x_norm.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
            sigma = torch.median(dist_sq.view(-1)).item() + self.eps
            K = torch.exp(-dist_sq / (2 * sigma))
        return K

    def _center_gram(self, K):
        """
        Center the gram matrix (subtract row/column means).

        Args:
            K: Gram matrix (B, N, N)

        Returns:
            K_centered: Centered gram matrix (B, N, N)
        """
        B, N, _ = K.shape
        # Centering matrix: H = I - (1/N) * 1 * 1^T
        # K_centered = H @ K @ H
        row_mean = K.mean(dim=2, keepdim=True)  # (B, N, 1)
        col_mean = K.mean(dim=1, keepdim=True)  # (B, 1, N)
        total_mean = K.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)

        K_centered = K - row_mean - col_mean + total_mean
        return K_centered

    def _hsic(self, K1, K2):
        """
        Compute Hilbert-Schmidt Independence Criterion (HSIC).

        HSIC(K1, K2) = (1/(N-1)^2) * trace(K1_centered @ K2_centered)

        Args:
            K1, K2: Centered gram matrices (B, N, N)

        Returns:
            hsic: HSIC value (B,)
        """
        B, N, _ = K1.shape
        # trace(K1 @ K2) = sum(K1 * K2^T)
        hsic = (K1 * K2).sum(dim=(1, 2)) / ((N - 1) ** 2 + self.eps)
        return hsic

    def _compute_batch_cka(self, student_tokens, teacher_tokens):
        """
        Compute batch-wise CKA for CLS-only mode (single token per sample).

        Instead of computing token-token correlations within each sample,
        compute sample-sample correlations across the batch.

        Args:
            student_tokens: (B, 1, D_s) - CLS tokens
            teacher_tokens: (B, 1, D_t) - CLS tokens

        Returns:
            cka: Scalar CKA similarity
        """
        # Squeeze to (B, D)
        s = student_tokens.squeeze(1)  # (B, D_s)
        t = teacher_tokens.squeeze(1)  # (B, D_t)

        # L2 normalize for numerical stability
        if self.normalize_features:
            s = F.normalize(s, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)

        # Compute batch gram matrices (B, B)
        K_s = s @ s.T  # (B, B)
        K_t = t @ t.T  # (B, B)

        # Center gram matrices
        B = K_s.shape[0]
        row_mean_s = K_s.mean(dim=1, keepdim=True)
        col_mean_s = K_s.mean(dim=0, keepdim=True)
        total_mean_s = K_s.mean()
        K_s_centered = K_s - row_mean_s - col_mean_s + total_mean_s

        row_mean_t = K_t.mean(dim=1, keepdim=True)
        col_mean_t = K_t.mean(dim=0, keepdim=True)
        total_mean_t = K_t.mean()
        K_t_centered = K_t - row_mean_t - col_mean_t + total_mean_t

        # Compute HSIC values
        hsic_st = (K_s_centered * K_t_centered).sum() / ((B - 1) ** 2 + self.eps)
        hsic_ss = (K_s_centered * K_s_centered).sum() / ((B - 1) ** 2 + self.eps)
        hsic_tt = (K_t_centered * K_t_centered).sum() / ((B - 1) ** 2 + self.eps)

        # Compute CKA
        denominator = torch.sqrt(hsic_ss * hsic_tt + self.eps)
        cka = hsic_st / denominator

        return cka

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute CKA loss between student and teacher representations.

        Args:
            student_tokens: (B, N_s, D_s) student token representations
            teacher_tokens: (B, N_t, D_t) teacher token representations
                           (should be interpolated to match N_s)

        Returns:
            loss: 1 - CKA (minimize to maximize CKA similarity)
        """
        B, N, _ = student_tokens.shape

        # CLS-only mode: use batch-wise CKA instead of token-wise
        if N == 1:
            cka = self._compute_batch_cka(student_tokens, teacher_tokens)
            loss = 1.0 - cka
            return loss

        # Standard token-wise CKA for multiple tokens
        # Compute gram matrices
        K_s = self._compute_gram(student_tokens)  # (B, N, N)
        K_t = self._compute_gram(teacher_tokens)  # (B, N, N)

        # Center gram matrices
        K_s = self._center_gram(K_s)
        K_t = self._center_gram(K_t)

        # Compute HSIC values
        hsic_st = self._hsic(K_s, K_t)  # Cross-similarity
        hsic_ss = self._hsic(K_s, K_s)  # Student self-similarity
        hsic_tt = self._hsic(K_t, K_t)  # Teacher self-similarity

        # Compute CKA
        denominator = torch.sqrt(hsic_ss * hsic_tt + self.eps)
        cka = hsic_st / denominator  # (B,)

        # Loss = 1 - CKA (minimize to maximize alignment)
        loss = 1.0 - cka.mean()

        return loss


class GramMatrixLoss(nn.Module):
    """
    Gram Matrix Loss for structural distillation (ablation baseline).

    Directly compares gram matrices using Frobenius norm.
    Simpler than CKA but less invariant to affine transformations.

    Loss = ||G_S - G_T||_F / num_elements
    """

    def __init__(self, normalize=True, eps=1e-8):
        """
        Args:
            normalize: Whether to L2-normalize features before gram computation
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def _compute_gram(self, x):
        """
        Compute normalized gram matrix.

        Args:
            x: Feature tensor (B, N, D)

        Returns:
            G: Gram matrix (B, N, N)
        """
        if self.normalize:
            x = F.normalize(x, dim=-1)
        G = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
        return G

    def forward(self, student_tokens, teacher_tokens):
        """
        Compute gram matrix loss.

        Args:
            student_tokens: (B, N_s, D_s) student representations
            teacher_tokens: (B, N_t, D_t) teacher representations

        Returns:
            loss: Frobenius norm difference between gram matrices
        """
        G_s = self._compute_gram(student_tokens)  # (B, N, N)
        G_t = self._compute_gram(teacher_tokens)  # (B, N, N)

        # Frobenius norm of difference, normalized by number of elements
        diff = G_s - G_t
        loss = torch.norm(diff, p='fro', dim=(1, 2))  # (B,)
        loss = loss / (G_s.shape[1] ** 2)  # Normalize by N^2
        loss = loss.mean()

        return loss


class LayerWiseStructuralLoss(nn.Module):
    """
    Layer-wise structural distillation with CKA and optional Gram matrix losses.

    Applies structural loss at multiple intermediate layers with learnable
    projection heads for dimension alignment.
    """

    def __init__(self, student_dim, teacher_dim, projection_dim, num_layers,
                 use_cka=True, use_gram=False, cka_kernel='linear'):
        """
        Args:
            student_dim: Student embedding dimension
            teacher_dim: Teacher embedding dimension
            projection_dim: Projection head output dimension
            num_layers: Number of layers to match
            use_cka: Whether to use CKA loss
            use_gram: Whether to use Gram matrix loss (ablation)
            cka_kernel: 'linear' or 'rbf' kernel for CKA
        """
        super().__init__()
        self.use_cka = use_cka
        self.use_gram = use_gram
        self.num_layers = num_layers

        # Projectors for dimension alignment
        self.student_projectors = nn.ModuleList([
            ProjectionHead(student_dim, projection_dim)
            for _ in range(num_layers)
        ])
        self.teacher_projectors = nn.ModuleList([
            ProjectionHead(teacher_dim, projection_dim)
            for _ in range(num_layers)
        ])

        # Loss functions
        if use_cka:
            self.cka_loss = CKALoss(kernel_type=cka_kernel)
        if use_gram:
            self.gram_loss = GramMatrixLoss(normalize=True)

    def forward(self, student_intermediates, teacher_intermediates, layer_indices):
        """
        Compute layer-wise structural loss.

        Args:
            student_intermediates: Dict[layer_idx] -> (B, N_s, D_s)
            teacher_intermediates: Dict[layer_idx] -> (B, N_s, D_t)
            layer_indices: List of layer indices

        Returns:
            loss: Combined structural loss
            loss_dict: Per-layer and per-loss-type breakdowns
        """
        total_cka_loss = 0.0
        total_gram_loss = 0.0
        loss_dict = {}

        for i, layer_idx in enumerate(layer_indices):
            student_tokens = student_intermediates[layer_idx]
            teacher_tokens = teacher_intermediates[layer_idx]

            # Project to common space
            proj_student = self.student_projectors[i](student_tokens)
            proj_teacher = self.teacher_projectors[i](teacher_tokens)

            if self.use_cka:
                layer_cka = self.cka_loss(proj_student, proj_teacher)
                total_cka_loss += layer_cka
                loss_dict[f'cka_loss_layer_{layer_idx}'] = layer_cka.item()

            if self.use_gram:
                layer_gram = self.gram_loss(proj_student, proj_teacher)
                total_gram_loss += layer_gram
                loss_dict[f'gram_loss_layer_{layer_idx}'] = layer_gram.item()

        # Average over layers
        num_layers = len(layer_indices)
        if self.use_cka:
            total_cka_loss = total_cka_loss / num_layers
            loss_dict['cka_loss_total'] = total_cka_loss.item()
        if self.use_gram:
            total_gram_loss = total_gram_loss / num_layers
            loss_dict['gram_loss_total'] = total_gram_loss.item()

        # Combined loss
        total_loss = total_cka_loss if self.use_cka else 0.0
        if self.use_gram:
            total_loss = total_loss + total_gram_loss

        return total_loss, loss_dict
