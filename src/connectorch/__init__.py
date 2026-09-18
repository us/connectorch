"""ConnecTorch: compile biological connectomes into trainable PyTorch networks.

    arbitrary connectome -> Connectome IR -> compiler/runtime -> torch.nn.Module

Biology defines the graph. PyTorch defines the learning.
"""

from . import datasets, io, nn, transforms
from .exceptions import (
    BackendError,
    ConnectomeValidationError,
    ConnectorchError,
    ConnectorchMemoryError,
)
from .ir import Connectome

__version__ = "0.1.0"

__all__ = [
    "Connectome",
    "nn",
    "io",
    "datasets",
    "transforms",
    "ConnectorchError",
    "ConnectomeValidationError",
    "ConnectorchMemoryError",
    "BackendError",
    "__version__",
]
