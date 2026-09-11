"""Gather-scatter propagation. The backend that trains.

``messages = zeros(N, B).index_add_(0, target, h[source] * edge_weight)``

Memory is O(E * B) and never depends on N^2, in the forward pass and in the
backward pass alike. This is what makes training on a 166,691-node connectome
possible at all.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Propagator, _check_shapes, register

__all__ = ["ScatterPropagator"]


@register("scatter")
class ScatterPropagator(Propagator):
    """Propagate by gathering source states and scattering weighted messages.

    Notes
    -----
    ``index_add_`` uses atomics on CUDA, so the summation order of messages
    arriving at the same target is not fixed run to run. Results can therefore
    differ in the last bits of a float32 accumulation. Enable
    ``torch.use_deterministic_algorithms(True)`` for bitwise-reproducible runs, at
    some cost in speed.
    """

    supports_sparse_backward = True

    source: Tensor
    target: Tensor

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        super().__init__(edge_index, num_nodes)
        self.register_buffer("source", edge_index[0].contiguous(), persistent=False)
        self.register_buffer("target", edge_index[1].contiguous(), persistent=False)

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Return ``[N, B]`` messages for a ``[N, B]`` state."""
        _check_shapes(self, h, edge_weight)
        messages = h.index_select(0, self.source) * edge_weight.unsqueeze(-1)
        out = torch.zeros(
            (self.num_nodes, h.shape[1]), dtype=messages.dtype, device=messages.device
        )
        return out.index_add(0, self.target, messages)
