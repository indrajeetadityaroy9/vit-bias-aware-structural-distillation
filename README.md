# BASD: Bias-Aware Structural Distillation for Vision Transformers

Transfer teacher inductive biases beyond logits by jointly aligning feature content statistics (`D x D`), token-level relations (`B x K x D`), attention routing (`H x N x N`), and spatial frequency structure (2D FFT real+imaginary), implemented as one cohesive end-to-end BASD system where all components are optimized together; using a unified structural objective `L_ce + L_rsd + L_rel + L_attn + L_spectral`, adaptive spectral teacher-layer mixing at each student extraction point, learnable teacher-to-student token alignment via cross-attention projectors, learned attention head-space alignment via 1x1 convolution before attention KL, parameter-free analytical loss balancing (UW-SO style inverse-loss softmax) with staggered warmup ramps, and a dual-view training path (clean teacher view, strongly augmented student view) with MixUp/CutMix on the student branch.

- Teacher: frozen `dinov2_vits14`, intermediate tokens + attentions extracted by hooks.
- Student: custom DeiT-Tiny style ViT returning logits and selected-layer intermediates.
- Alignment:
  - Token alignment: per-layer cross-attention projector.
  - Attention alignment: head aligner (1x1 conv) across stacked heads.
- Loss orchestration:
  - Adaptive spectral layer selector mixes teacher layers per extraction point.
  - Four distillation losses are warmup-ramped and then weighted by analytical UW-SO.
  - Final objective adds CE + weighted distillation + selector entropy regularization.

## References

1. Training data-efficient image transformers & distillation through attention, https://arxiv.org/abs/2012.12877  
2. DINOv2: Learning Robust Visual Features without Supervision, https://arxiv.org/abs/2304.07193  
3. Barlow Twins: Self-Supervised Learning via Redundancy Reduction, https://arxiv.org/abs/2103.03230  
4. Cross-Architecture Distillation Made Simple with Redundancy Suppression, https://arxiv.org/abs/2507.21844  
5. VRM: Knowledge Distillation via Virtual Relation Matching, https://arxiv.org/abs/2502.20760  
6. SpectralKD: A Unified Framework for Interpreting and Distilling Vision Transformers via Spectral Analysis, https://arxiv.org/abs/2412.19055  
7. Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning, https://arxiv.org/abs/2408.07985  
8. Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation, https://arxiv.org/abs/2403.01479  
9. ALP-KD: Attention-Based Layer Projection for Knowledge Distillation, https://arxiv.org/abs/2012.14022  
10. Revisiting Intermediate-Layer Matching in Knowledge Distillation: Layer-Selection Strategy Doesn't Matter (Much), https://arxiv.org/abs/2502.04499 
