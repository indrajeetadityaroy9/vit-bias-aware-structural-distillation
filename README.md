# BASD: Bias-Aware Structural Distillation via Grassmannian Geometry

Architecture-agnostic knowledge distillation framework that transfers teacher inductive biases beyond logits by jointly aligning feature content statistics (`D x D`), token-level relations (`B x K x D`), attention routing (`H x N x N`), and singular-value spectral structure. All components operate on generic `(B, N, D)` token sequences with no spatial-grid or modality-specific assumptions.

The unified structural objective `L_ce + L_rsd + L_rel + L_attn + L_spectral + L_selector_reg` is optimized end-to-end with Grassmannian geometry-guided adaptive teacher-layer selection, learnable cross-attention token alignment, learned head-space alignment via 1x1 convolution, parameter-free analytical loss balancing (UW-SO), staggered warmup ramps, and a dual-view training path (clean teacher view, strongly augmented student view) with MixUp/CutMix on the student branch.

1. **Grassmannian projection metric for adaptive layer selection.** Layer compatibility is measured on the Grassmannian manifold `Gr(d, k)` via the projection metric `d_p^2(U_s, U_l) = k - ||U_s^T U_l||_F^2`, where `U_s` and `U_l` are top-k subspaces extracted from feature covariances in a learned common space. This metric is basis-invariant, theoretically grounded, and converts to temperature-controlled softmax mixing weights over all teacher layers per student extraction point.

2. **Learnable common-space projections.** Architecture-agnostic linear maps `phi_s: R^D_s -> R^d` and `phi_t: R^D_t -> R^d` with orthogonal initialization project heterogeneous student and teacher representations into a shared `d`-dimensional space. A reconstruction loss `L_recon = ||z_mixed - z_s||^2 / d` provides gradient signal to the projections without requiring backward through eigendecomposition.

3. **Subspace orthogonality penalty.** A weight-modulated diversity regularizer `L_orth = sum_{l<l'} w_l * w_{l'} * ||U_l^T U_{l'}||_F^2` penalizes selected teacher layers for having aligned principal subspaces, encouraging the selector to mix geometrically complementary layers while gradient flows through the mixing weights to the learnable temperatures.

## Architecture

- **Teacher**: frozen DINOv2 (`dinov2_vits14`), intermediate tokens and attentions extracted by hooks.
- **Student**: DeiT-Tiny style ViT returning logits and selected-layer intermediates.
- **Token alignment**: per-layer cross-attention projector maps teacher tokens to student dimension.
- **Attention alignment**: 1x1 Conv2d head aligner maps student attention heads to teacher head space.
- **Adaptive layer selection** (Grassmannian):
  - Learnable projections `phi_s`, `phi_t` map student/teacher tokens to a common `d`-dimensional space.
  - Top-k subspaces extracted via eigendecomposition (detached from autograd for numerical stability).
  - Grassmannian projection metric distances converted to per-extraction-point softmax mixing weights with learnable temperatures.
  - Subspace orthogonality penalty + reconstruction loss returned as combined regularization.
- **Distillation losses** (all architecture-agnostic):
  - `L_rsd`: Redundancy suppression via cross-correlation between projected student and teacher features.
  - `L_rel`: Virtual relation matching via random token pair sampling with normalized pairwise differences.
  - `L_attn`: KL divergence on attention maps with learned head alignment and resolution interpolation.
  - `L_spectral`: SVD-based singular-value spectrum matching on the token-feature matrix.
  - Token count alignment between student and teacher uses 1D linear interpolation along the sequence dimension.
- **Loss orchestration**:
  - Four distillation losses are warmup-ramped with staggered cosine schedules and weighted by analytical UW-SO (inverse-loss softmax).
  - Final objective: `L_ce + L_weighted_distillation + L_selector_reg`.

## References

**Grassmannian layer selection:**
1. Grassmannian Deep Networks, https://arxiv.org/abs/2511.08628
2. Grassmannian Geodesic Alignment, https://arxiv.org/abs/2507.17998
3. Relative Geodesic Representations on the Grassmannian, https://arxiv.org/abs/2506.01599

**Distillation losses:**
4. Cross-Architecture Distillation Made Simple with Redundancy Suppression, https://arxiv.org/abs/2507.21844
5. Barlow Twins: Self-Supervised Learning via Redundancy Reduction, https://arxiv.org/abs/2103.03230
6. VRM: Knowledge Distillation via Virtual Relation Matching, https://arxiv.org/abs/2502.20760
7. Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation, https://arxiv.org/abs/2403.01479
8. SpectralKD: A Unified Framework for Interpreting and Distilling Vision Transformers via Spectral Analysis, https://arxiv.org/abs/2412.19055

**Loss orchestration:**
9. Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning, https://arxiv.org/abs/2408.07985

**Student and teacher architectures:**
10. Training data-efficient image transformers & distillation through attention, https://arxiv.org/abs/2012.12877
11. DINOv2: Learning Robust Visual Features without Supervision, https://arxiv.org/abs/2304.07193
