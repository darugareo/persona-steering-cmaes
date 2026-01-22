"""
Baseline methods for persona steering comparison.
Implements 7 baseline approaches for rigorous evaluation.
"""

from .base import BaseMethod
from .prompt_persona import PromptPersonaMethod
from .meandiff import MeanDiffMethod
from .pca_steering import PCASteeringMethod
from .random_search import RandomSearchMethod
from .grid_search import GridSearchMethod
from .proposed import ProposedMethod

__all__ = [
    'BaseMethod',
    'PromptPersonaMethod',
    'MeanDiffMethod',
    'PCASteeringMethod',
    'RandomSearchMethod',
    'GridSearchMethod',
    'ProposedMethod',
]
