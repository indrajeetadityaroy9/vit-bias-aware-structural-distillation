"""
ConvNeXt V2 with registry decorator.

Only available if timm is installed.
"""
import logging

logger = logging.getLogger(__name__)

try:
    from src.teachers import ConvNeXtV2Tiny, HAS_TIMM
    from src.modeling.registry import register_model

    if HAS_TIMM:
        # Register ConvNeXt V2 Tiny with the model registry
        register_model('convnext_v2_tiny')(ConvNeXtV2Tiny)
        __all__ = ['ConvNeXtV2Tiny']
    else:
        __all__ = []
        logger.debug("timm not available, ConvNeXt V2 not registered")

except ImportError:
    __all__ = []
    logger.debug("Could not import ConvNeXtV2Tiny")
