"""
Research analytics module for the Locality Curse study.

Provides:
- Metrics: Hessian trace, attention distance, CKA
- Visualization: Publication-quality plots
- LocalityCurseForensics: Complete diagnostic toolkit
"""

from .metrics import (
    HessianAnalyzer,
    AttentionDistanceAnalyzer,
    CKAAnalyzer,
)

from .visualization import AnalyticsVisualizer

from .engine import (
    AnalyticsRunner,
    LocalityCurseForensics,
)

__all__ = [
    # Metrics
    'HessianAnalyzer',
    'AttentionDistanceAnalyzer',
    'CKAAnalyzer',
    # Runners
    'AnalyticsRunner',
    'LocalityCurseForensics',
    # Visualization
    'AnalyticsVisualizer',
]
