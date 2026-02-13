# BASD: Bias-Aware Structural Distillation (DINOv2 -> DeiT)

Training mechanism for transferring representations from a frozen `DINOv2 ViT-S/14` teacher to a `DeiT-Tiny`-style student with structural distillation losses.

Standard KD with logits or a single feature loss under-utilizes the teacher when teacher/student token geometry and embedding spaces differ.

The objective is to train a DeiT student using a unified multi-axis distillation objective that jointly aligns:
- content statistics (`D x D`)
- token structure (`N x N`)
- attention routing (`H x N x N`)
- frequency structure (2D spectrum)

1. Orthogonal feature decomposition for distillation:
   content + structure + routing + spectral constraints in one objective.
2. Learnable teacher->student token alignment:
   cross-attention projector instead of naive geometric resizing.
3. Automatic loss balancing:
   homoscedastic uncertainty weighting instead of manual lambdas.
4. Staggered curriculum:
   cosine warmup ramps per component with fixed offsets.
5. Deterministic training path:
   one canonical `torchrun -m src <config>` route.

## Integrated Mechanism

Given student logits and intermediate tokens/attention from selected layers:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{CE}
 + \mathcal{W}(
r_{rsd}\mathcal{L}_{rsd},
r_{gram}\mathcal{L}_{gram},
r_{attn}\mathcal{L}_{attn},
r_{spec}\mathcal{L}_{spec})
\]

where `r_*` are warmup ramps and `W` is uncertainty weighting.

### Teacher/Student Coupling
- Teacher: frozen DINOv2 loaded from `torch.hub`
- Student: DeiT with intermediate feature/attention extraction
- Layer hooks extract teacher tokens and attention in one forward pass
- Cross-attention projectors align teacher token count to student token count

### Distillation Components
- `L_rsd` (content): cross-correlation identity matching with off-diagonal suppression
- `L_gram` (structure): token-token Gram matrix matching
- `L_attn` (routing): KL divergence between teacher/student attention distributions
- `L_spec` (spectral): radial-band FFT magnitude matching

### Optimization
- DDP (`nccl`) with `torchrun`
- BF16 autocast
- AdamW + linear warmup + cosine decay
- SWA late-phase averaging
- early stopping on validation loss
- gradient clipping

## References

1. DeiT (ICML 2021): https://arxiv.org/abs/2012.12877  
   Student transformer design and distillation context (`models/deit.py`).
2. DINOv2 (TMLR 2024): https://arxiv.org/abs/2304.07193  
   Pretrained teacher loaded/frozen in `models/teachers.py`.
3. Barlow Twins (ICML 2021): https://arxiv.org/abs/2103.03230  
   Redundancy-reduction principle behind `L_rsd`.
4. Redundancy Suppression Distillation: https://arxiv.org/abs/2507.21844  
   Token-level cross-correlation loss (`training/losses/token.py`).
5. DINOv3: https://arxiv.org/abs/2508.10104  
   Structural Gram loss (`training/losses/structural.py`).
6. SDKD Spectral Distillation: https://arxiv.org/abs/2507.02939  
   FFT band-matching loss (`training/losses/frequency.py`).
7. Uncertainty Weighting (CVPR 2018): https://arxiv.org/abs/1705.07115  
   Adaptive multi-loss weighting (`training/losses/curriculum.py`).