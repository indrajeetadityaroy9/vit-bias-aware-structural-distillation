"""
Analytics module for model analysis.

Provides Hessian trace, attention distance, and CKA metrics.
"""

from .metrics import HessianAnalyzer, AttentionDistanceAnalyzer, CKAAnalyzer
from .engine import run_analytics

__all__ = [
    'HessianAnalyzer',
    'AttentionDistanceAnalyzer',
    'CKAAnalyzer',
    'run_analytics',
]
