"""Differentiable runtimes that execute a connectome."""

from .dynamics import ACTIVATIONS, resolve_activation
from .hash import SparseExpander, k_wta
from .model import ConnectomeModel, InputProjection, count_parameters
from .parameterisation import BiologicalWeights, EdgeWeights, FixedWeights, FreeWeights
from .recurrent import ConnectomeRNN
from .weights import WEIGHT_STRATEGIES, initial_edge_weights, resolve_strategy

__all__ = [
    "ConnectomeRNN",
    "ConnectomeModel",
    "InputProjection",
    "count_parameters",
    "BiologicalWeights",
    "EdgeWeights",
    "FreeWeights",
    "FixedWeights",
    "ACTIVATIONS",
    "resolve_activation",
    "WEIGHT_STRATEGIES",
    "initial_edge_weights",
    "resolve_strategy",
    "SparseExpander",
    "k_wta",
]
