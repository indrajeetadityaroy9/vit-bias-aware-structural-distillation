"""
Analysis metrics (pure computation, no plotting).

Provides:
- HessianAnalyzer: Loss landscape curvature analysis
- AttentionDistanceAnalyzer: Mean attention distance, entropy, CLS dispersion
- CKAAnalyzer: Centered Kernel Alignment similarity
"""

from src.evaluation.analyzers.geometry import HessianAnalyzer
from src.evaluation.analyzers.attention import AttentionDistanceAnalyzer
from src.evaluation.analyzers.cka import CKAAnalyzer
