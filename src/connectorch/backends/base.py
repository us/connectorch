"""Message-passing backends.

A backend computes one propagation step

``messages[target] = sum over edges (target <- source) of edge_weight * h[source]``

for a state ``h`` of shape ``[num_nodes, batch]``. The adjacency is conceptually
``A[target, source] = edge_weight``, so a step is ``A @ h``.

State is node-major (``[N, B]``, not ``[B, N]``) because that is the orientation
sparse matrix multiplication and index_add both want; the public module transposes
once at its boundary instead of once per recurrent step.

The edge weights are passed in per call rather than held by the backend, so a
trainable ``nn.Parameter`` owned by the module flows through autograd without the
backend caching a stale view of it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..exceptions import BackendError, ConnectorchMemoryError

__all__ = ["Propagator", "build_propagator", "BACKENDS"]


class Propagator(nn.Module):
    """Base class for a one-step message-passing operator.

    Parameters
    ----------
    edge_index:
        ``[2, E]`` int64 tensor; row 0 is the source index, row 1 the target index.
        Must already be in canonical order (ascending by target, then source),
        which :class:`~connectorch.ir.Connectome` guarantees.
    num_nodes:
        Number of nodes ``N``.
    """

    #: Whether this backend can compute gradients w.r.t. edge weights without
    #: materialising a dense ``[N, N]`` tensor.
    supports_sparse_backward: bool = True

    #: Whether the existing per-edge/batch saved-activation estimate applies.
    #: False does not mean the backend uses no activation or workspace memory.
    saves_edge_batch_activations: bool = True

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        super().__init__()
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise BackendError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        self.num_nodes = int(num_nodes)
        self.num_edges = int(edge_index.shape[1])

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Propagate ``h`` (``[N, B]``) one step and return the messages (``[N, B]``)."""
        raise NotImplementedError

    def rebuild(self, edge_index: Tensor) -> None:
        """Recompute any derived layout after the topology changed.

        Backends that only keep the endpoint arrays refresh those; a backend that
        precomputes a different structure overrides this.
        """
        device = next((b.device for b in self.buffers()), edge_index.device)
        self.num_edges = int(edge_index.shape[1])
        for name, row in (("source", 0), ("target", 1)):
            if hasattr(self, name):
                setattr(self, name, edge_index[row].contiguous().to(device))

    def check_dense_cost(self, dtype: torch.dtype, *, reason: str) -> None:
        """Refuse up front if this backend would allocate a dense ``[N, N]`` tensor.

        The estimate is computed from the graph, never hardcoded, so the message
        tells the user the real number of gibibytes they were about to ask for.
        """
        itemsize = torch.empty((), dtype=dtype).element_size()
        gib = self.num_nodes * self.num_nodes * itemsize / 2**30
        if gib > _DENSE_LIMIT_GIB:
            raise ConnectorchMemoryError(
                f'backend="{getattr(self, "backend_name", type(self).__name__)}" '
                f"{reason} for {self.num_nodes:,} nodes.\n"
                f"A {dtype} [{self.num_nodes:,} x {self.num_nodes:,}] adjacency "
                f"needs {gib:,.1f} GiB.\n"
                'Use backend="scatter", which allocates O(E) instead.'
            )


#: Largest dense ``[N, N]`` allocation allowed before a backend refuses, in GiB.
_DENSE_LIMIT_GIB = 1.0


def _check_shapes(propagator: Propagator, h: Tensor, edge_weight: Tensor) -> None:
    if h.shape[0] != propagator.num_nodes:
        raise BackendError(
            f"state has {h.shape[0]} rows but the graph has {propagator.num_nodes} "
            "nodes. State must be node-major, shaped [num_nodes, batch]."
        )
    if edge_weight.shape != (propagator.num_edges,):
        raise BackendError(
            f"edge_weight must have shape [{propagator.num_edges}], got {tuple(edge_weight.shape)}."
        )


BACKENDS: dict[str, type[Propagator]] = {}


def register(name: str):  # type: ignore[no-untyped-def]
    """Class decorator registering a propagator under ``name``."""

    def wrap(cls: type[Propagator]) -> type[Propagator]:
        BACKENDS[name] = cls
        cls.backend_name = name  # type: ignore[attr-defined]
        return cls

    return wrap


def build_propagator(
    name: str,
    edge_index: Tensor,
    num_nodes: int,
    *,
    trainable: bool,
) -> Propagator:
    """Instantiate a backend by name, resolving ``"auto"``.

    Parameters
    ----------
    name:
        ``"auto"``, ``"dense"``, ``"sparse_mm"``, ``"scatter"`` or ``"metal_csr"``.
    trainable:
        Whether the edge weights will require gradients. This decides ``"auto"``
        and gates the backends that cannot produce gradients cheaply.

    Notes
    -----
    ``"auto"`` picks ``"scatter"`` for trainable weights and ``"sparse_mm"``
    otherwise. That rule comes from measurement, not preference: the backward of
    ``torch.sparse.mm`` with respect to a sparse operand materialises a dense
    ``[N, N]`` gradient. On an NVIDIA GB10 at N=50,000 with 500k edges and batch
    32, forward+backward cost 74.6 ms and 9.40 GiB through COO and 3.28 ms and
    0.24 GiB through scatter, where a dense float32 ``[N, N]`` is 9.3 GiB. Forward
    only, CSR is the fastest option at 0.16 ms against scatter's 1.76 ms.
    """
    from . import dense, metal_csr, scatter, sparse_mm  # noqa: F401 (registration side effect)

    if name == "auto":
        name = "scatter" if trainable else "sparse_mm"
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise BackendError(
            f"unknown backend {name!r}; available: {sorted(BACKENDS) + ['auto']}."
        ) from None

    propagator = cls(edge_index, num_nodes)
    if trainable and not propagator.supports_sparse_backward:
        propagator.check_dense_cost(
            torch.float32,
            reason=(
                "was requested with trainable weights, and its backward pass "
                "materialises a dense adjacency gradient"
            ),
        )
    return propagator
