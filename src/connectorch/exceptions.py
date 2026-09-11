"""Exception types raised by ConnecTorch.

All exceptions inherit from :class:`ConnectorchError` so user code can catch the
whole family with a single ``except``.
"""

from __future__ import annotations

__all__ = [
    "ConnectorchError",
    "ConnectomeValidationError",
    "ConnectorchMemoryError",
    "BackendError",
]


class ConnectorchError(Exception):
    """Base class for every ConnecTorch error."""


class ConnectomeValidationError(ConnectorchError, ValueError):
    """A connectome's node or edge table violates the IR contract.

    Raised during construction, never later, so that an invalid graph can never
    reach the neural runtime.
    """


class ConnectorchMemoryError(ConnectorchError, MemoryError):
    """An operation was refused because it would allocate a dense ``N x N`` tensor.

    ConnecTorch computes the allocation up front and refuses rather than letting
    the allocator fail mid-training.
    """


class BackendError(ConnectorchError):
    """A propagation backend was requested that cannot serve the given model."""
