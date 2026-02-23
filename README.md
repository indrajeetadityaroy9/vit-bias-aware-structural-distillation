# MCSD: Manifold-Consistent Structural Distillation

MCSD is an architecture-agnostic knowledge distillation framework that transfers teacher inductive bias through token structure by enforcing geometric consistency at inter-model projection boundaries. It replaces approximate computations with exact formulations, introduces Stiefel manifold constraints at critical projection points, and uses statistical weighting

MCSD operates on four nested Riemannian manifolds:

| Manifold | Role | Distance |
|----------|------|----------|
| **St(n, d)** — Stiefel | Inter-model projections (phi_s, phi_t, teacher_proj) | Polar decomposition retraction |
| **Gr(d, k)** — Grassmannian | Layer selection via subspace comparison | d^2 = k - \|\|U^T V\|\|^2_F |
| **BW(d)** — Bures-Wasserstein | Distributional token matching | Cholesky + SVD exact formulation |
| **Proc(n)** — Procrustes | Structural geometry alignment | D^2 = tr(K_s) + tr(K_t) - 2\|\|R_s^T R_t\|\|_* |

## Loss Decomposition

```
L_total = L_CE + Sum_i( w_i * rho_i(t) * L_i ) + L_reg
```

where w_i are scale-invariant weights (z-score UW-SO), rho_i(t) are gradient-gated activation ramps, L_i in {L_RSD, L_Proc, L_Attn, L_BW}, and L_reg is Grassmannian selector regularization via Lagrangian dual ascent.

1. **SVD-based Bures-Wasserstein distance** (`losses/spectral.py`) — Exact d^2_BW via Cholesky factorization + SVD nuclear norm, with auto diagonal/non-diagonal selection based on N vs D ratio and auto mean/covariance balance.

2. **Procrustes relational loss** (`losses/relational.py`) — Dimension-agnostic structural matching via nuclear norm on centered cross-products, with optional attention weighting. Teacher is NOT detached (sole gradient path to the cross-attention projector).

3. **Scale-invariant multi-task weighting** (`losses/weighting.py`) — Z-score normalized UW-SO: zero hyperparameters, zero persistent state. Replaces inverse-loss softmax with variance-derived estimates.

4. **Phase-aware Grassmannian layer selection** (`losses/layer_selector.py`) — Projects teacher layers to a shared subspace via Stiefel-constrained projections, computes manifold distances for temperature-scaled mixing weights. Spectral concentration gating via z-score normalization. Marchenko-Pastur auto-rank for subspace dimensionality.

5. **Stiefel-constrained inter-model projections** (`models/projector.py`, `losses/layer_selector.py`) — Polar decomposition retraction (W = U V^T from SVD of W) enforced after each optimizer step on phi_s, phi_t, and teacher_proj. Preserves inner-product geometry at cross-model boundaries.

## Adaptive Mechanisms

- **Gradient-gated scheduling** — Components activate permanently once their loss CV drops below 1.0 (first component always active). Replaces manual warmup fraction tuning.
- **Lagrangian dual ascent** — Adaptive regularization for diversity and reconstruction losses in the layer selector, with automatic multiplier updates.
- **Marchenko-Pastur auto-rank** — Random matrix theory determines subspace rank from the spectral distribution of teacher representations, eliminating manual rank selection.
- **Auto mean/covariance balance** — BW loss dynamically scales mean vs covariance terms based on their running ratio.

The five MCSD-specific parameters:

| Parameter | Default | Role |
|-----------|---------|------|
| `num_extraction_points` | 4 | Number of student layers for feature extraction |
| `layer_selector_temperature` | 2.0 | Softmax temperature for layer mixing weights |
| `layer_selector_grass_cov_eps` | 1e-4 | Covariance regularization in Grassmannian distance |
| `rel_attn_weighted` | true | Enable attention weighting in relational loss |
| `grad_gate_window` | 100 | Window size for gradient-gated scheduling CV |
