"""
Loss functions for classification, distillation, and structural alignment.
"""

from src.training.losses.distillation import DistillationLoss
from src.training.losses.token import TokenCorrelationLoss
from src.training.losses.structural import CKALoss
from src.training.losses.combined import SelfSupervisedDistillationLoss
