"""Dense propagation. The correctness oracle, never the production path.

This backend materialises the full ``[N, N]`` adjacency, which is exactly what the
rest of the library exists to avoid. It is here so that every sparse backend has
something unambiguous to be tested against, and so that tiny graphs have a path
with no sparse-kernel subtleties at all.

It refuses to run on anything big enough to matter.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Propagator, _check_shapes, register

__all__ = ["DensePropagator"]


@register("dense")
class DensePropagator(Propagator):
    """Build ``A[target, source]`` densely and return ``A @ h``.

    Raises
    ------
    ConnectorchMemoryError
        At construction, if the dense adjacency would exceed one gibibyte.
    """

    supports_sparse_backward = False

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        super().__init__(edge_index, num_nodes)
        self.check_dense_cost(torch.float32, reason="was requested")
        self.register_buffer("source", edge_index[0].contiguous(), persistent=False)
        self.register_buffer("target", edge_index[1].contiguous(), persistent=False)

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Return ``[N, B]`` messages for a ``[N, B]`` state."""
        _check_shapes(self, h, edge_weight)
        self.check_dense_cost(edge_weight.dtype, reason="was requested")
        adjacency = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=edge_weight.dtype, device=edge_weight.device
        )
        adjacency = adjacency.index_put((self.target, self.source), edge_weight, accumulate=True)
        return adjacency @ h
