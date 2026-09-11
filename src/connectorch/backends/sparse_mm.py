"""Sparse matrix multiplication propagation. Fast forward, expensive backward.

Uses a CSR adjacency ``A[target, source]`` and ``torch.sparse.mm``. Because the IR
stores edges sorted by ``(target, source)``, the CSR row pointers can be built
directly from a bincount of the targets: no ``coalesce()``, no value permutation,
and therefore no risk of a trainable weight vector silently drifting out of
alignment with its edges.

Use this for inference and for fixed-weight connectome simulation. Do not use it
for training a large graph: see :meth:`SparseMMPropagator.forward`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..exceptions import BackendError
from .base import Propagator, _check_shapes, register

__all__ = ["SparseMMPropagator"]


@register("sparse_mm")
class SparseMMPropagator(Propagator):
    """Propagate with ``torch.sparse.mm`` over a CSR adjacency.

    Notes
    -----
    The backward of ``torch.sparse.mm`` with respect to the sparse operand
    materialises a dense ``[N, N]`` intermediate. This is a known open PyTorch
    bug (pytorch/pytorch#41128, #49683), not something a caller can avoid.
    Measured on an NVIDIA GB10 with N=50,000 and 500k edges at batch 32, forward+backward peaked at 9.40 GiB,
    which is the size of a dense float32 ``[N, N]`` (9.3 GiB), against 0.24 GiB
    for the scatter backend. At N=166,691 the same path tries to allocate 111 GiB
    and fails. Forward only it is the fastest backend by a wide margin: 0.16 ms
    against scatter's 1.76 ms in the same configuration.

    Constructing the module with trainable weights is refused up front by
    :func:`~connectorch.backends.base.build_propagator` for any graph where that
    dense gradient would exceed a gibibyte.
    """

    supports_sparse_backward = False

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        super().__init__(edge_index, num_nodes)
        source, target = edge_index[0], edge_index[1]
        if self.num_edges > 1:
            if bool((target[1:] < target[:-1]).any()):
                raise BackendError(
                    "edge_index must be sorted by target index for the CSR backend. "
                    "Connectome guarantees this ordering; edge_index was modified."
                )
            # A CSR tensor cannot hold two entries at the same (row, col). Its
            # forward silently sums them, but its backward returns one gradient
            # per *unique* coordinate, which no longer lines up with edge_weight.
            # Refuse here, where we can say what to do about it.
            duplicated = (target[1:] == target[:-1]) & (source[1:] == source[:-1])
            if bool(duplicated.any()):
                count = int(duplicated.sum())
                raise BackendError(
                    f'backend="sparse_mm" cannot represent parallel edges: '
                    f"{count:,} edge(s) repeat a (source, target) pair that another "
                    "edge already occupies.\n"
                    "A CSR adjacency has one value per coordinate, so gradients "
                    "would no longer align with edge_weight.\n"
                    "Either build the connectome with aggregate_parallel_edges=True "
                    '(the default), or use backend="scatter", which handles parallel '
                    "edges natively."
                )
        counts = torch.bincount(target, minlength=num_nodes)
        crow = torch.zeros(num_nodes + 1, dtype=torch.int64, device=edge_index.device)
        torch.cumsum(counts, dim=0, out=crow[1:])
        self.register_buffer("crow_indices", crow, persistent=False)
        self.register_buffer("col_indices", source.contiguous(), persistent=False)

    def rebuild(self, edge_index: Tensor) -> None:
        """Recompute the CSR layout after the topology changed underneath us.

        ``load_state_dict`` writes a new ``edge_index`` straight into the owning
        module's buffer. The row pointers derived from the old one would still be
        sitting here, and the model would quietly propagate along the topology it
        used to have.
        """
        device = self.crow_indices.device
        rebuilt = type(self)(edge_index.to(device), self.num_nodes)
        self.num_edges = rebuilt.num_edges
        self.crow_indices = rebuilt.crow_indices.to(device)
        self.col_indices = rebuilt.col_indices.to(device)

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Return ``[N, B]`` messages for a ``[N, B]`` state."""
        _check_shapes(self, h, edge_weight)
        if edge_weight.requires_grad:
            self.check_dense_cost(
                edge_weight.dtype,
                reason=(
                    "received edge weights that require gradients, and its "
                    "backward pass materialises a dense adjacency gradient"
                ),
            )
        adjacency = torch.sparse_csr_tensor(
            self.crow_indices,
            self.col_indices,
            edge_weight,
            size=(self.num_nodes, self.num_nodes),
            check_invariants=False,
        )
        return torch.sparse.mm(adjacency, h)
