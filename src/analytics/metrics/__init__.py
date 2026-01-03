"""
Analysis metrics (pure computation, no plotting).

Provides:
- HessianAnalyzer: Loss landscape curvature analysis
- AttentionDistanceAnalyzer: Mean attention distance, entropy, CLS dispersion
- CKAAnalyzer: Centered Kernel Alignment similarity
"""

from .geometry import HessianAnalyzer
from .attention import AttentionDistanceAnalyzer
from .representation import CKAAnalyzer

__all__ = [
    'HessianAnalyzer',
    'AttentionDistanceAnalyzer',
    'CKAAnalyzer',
]
