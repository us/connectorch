"""Native Apple Metal CSR propagation with state and edge-weight gradients.

Adapted from original backend contributions in
https://github.com/fernando-neto-ai/fly-wordbrain. These contributions are
provided under ConnecTorch's MIT license; see the repository LICENSE.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..exceptions import BackendError
from ._metal_csr import TrainableCSR
from .base import Propagator, register

__all__ = ["MetalCSRPropagator"]


def _validated_edges(edge_index: Tensor, num_nodes: int) -> NDArray[np.int64]:
    if (
        isinstance(num_nodes, bool | np.bool_)
        or not isinstance(num_nodes, int | np.integer)
        or not 0 < num_nodes < np.iinfo(np.int32).max
    ):
        raise BackendError("metal_csr requires a positive integer node count below 2^31-1")
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.layout != torch.strided
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.shape[1] >= np.iinfo(np.int32).max
    ):
        raise BackendError("Expected strided edge_index[2, E] with fewer than 2^31-1 edges")
    if edge_index.requires_grad:
        raise BackendError("Topology must not require gradients")
    if edge_index.device.type not in ("cpu", "mps"):
        raise BackendError("metal_csr supports only CPU and MPS")
    if edge_index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise BackendError("edge_index must contain integer endpoints")
    edges = edge_index.detach().cpu().numpy().astype(np.int64, copy=True)
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise BackendError("edge_index endpoint is outside the node range")
    if edges.shape[1] > 1:
        if np.any(np.diff(edges[1]) < 0):
            raise BackendError("metal_csr requires edge_index sorted by target")
        order = np.lexsort((edges[0], edges[1]))
        if np.any(np.all(edges[:, order[1:]] == edges[:, order[:-1]], axis=0)):
            raise BackendError(
                'metal_csr does not support parallel edges; use backend="scatter" or '
                "build the connectome with aggregate_parallel_edges=True. "
                "Edges are never silently coalesced."
            )
    return edges


@register("metal_csr")
class MetalCSRPropagator(Propagator):
    """Node-major CSR propagation without an edge-by-batch message tensor.

    MPS accepts float32 states and values; CPU float32/float64 provides an
    explicit correctness reference. Both state and supplied edge values support
    first-order gradients. Topology is immutable between explicit rebuilds.

    Layout validation and uploads occur once on first use, including under
    ``torch.inference_mode()``. Numerical calls never fall back to CPU. Derived
    topology is nonpersistent, preserving cross-backend state-dict portability.
    """

    backend_name = "metal_csr"
    supports_sparse_backward = True
    saves_edge_batch_activations = False
    edge_index: Tensor

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        _validated_edges(edge_index, num_nodes)
        super().__init__(edge_index, num_nodes)
        # Normal tensors have version counters even when constructed while the
        # caller is in inference mode. They can be reused by subsequent training.
        with torch.inference_mode(False):
            endpoints = edge_index.detach().clone()
        self.register_buffer("edge_index", endpoints, persistent=False)
        self._runtime: TrainableCSR | None = None
        self._signature: tuple[Any, ...] | None = None
        self.register_load_state_dict_post_hook(_clear_loaded_layout)

    def _clear_runtime(self) -> None:
        self._runtime = None
        self._signature = None

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> MetalCSRPropagator:
        self._clear_runtime()
        # nn.Module.to() can itself be called under inference_mode; do not let
        # the destination endpoint buffer lose its version counter.
        with torch.inference_mode(False):
            super()._apply(fn, recurse=recurse)
            if self.edge_index.is_inference():
                self.edge_index = self.edge_index.clone()
        return self

    def rebuild(self, edge_index: Tensor) -> None:
        """Validate replacement endpoints before discarding a working layout."""
        _validated_edges(edge_index, self.num_nodes)
        with torch.inference_mode(False):
            endpoints = edge_index.detach().to(self.edge_index.device).clone()
        self.edge_index = endpoints
        self.num_edges = int(endpoints.shape[1])
        self._clear_runtime()

    def _get_runtime(self) -> TrainableCSR:
        signature = (
            id(self.edge_index),
            self.edge_index._version,
            self.edge_index.data_ptr(),
            self.edge_index.device,
            self.edge_index.dtype,
            tuple(self.edge_index.shape),
            self.num_nodes,
            self.num_edges,
        )
        if self._runtime is not None and signature != self._signature:
            raise RuntimeError("Topology changed without rebuilding the metal_csr propagator")
        if self._runtime is None:
            edges = _validated_edges(self.edge_index, self.num_nodes)
            ptr = np.r_[0, np.cumsum(np.bincount(edges[1], minlength=self.num_nodes))]
            self._runtime = TrainableCSR(
                ptr, edges[0], (self.num_nodes, self.num_nodes), device=self.edge_index.device
            )
            self._signature = signature
        return self._runtime

    def forward(self, h: Tensor, edge_weight: Tensor) -> Tensor:
        """Return node-major ``[N, B]`` messages in canonical edge-value order."""
        if not isinstance(h, Tensor) or h.ndim != 2 or h.shape[0] != self.num_nodes:
            raise BackendError("metal_csr state must have shape [num_nodes, batch]")
        return self._get_runtime().mm(h.T, edge_weight).T.contiguous()

    def verify_frozen(self) -> bool:
        """Explicit host audit, including endpoint writes made through .data."""
        runtime = self._get_runtime()
        current = self.edge_index.detach().cpu().numpy()
        rows = runtime._buffers["row"].detach().cpu().numpy()
        columns = runtime._buffers["col"].detach().cpu().numpy()
        if not np.array_equal(current[0], columns) or not np.array_equal(current[1], rows):
            raise RuntimeError("Propagator endpoints differ from the execution layout")
        return runtime.verify_frozen()

    def metadata(self) -> dict[str, Any]:
        """Report kernel provenance, layout identity, and native launch counts."""
        return {**self._get_runtime().metadata(), "connectorch_backend": self.backend_name}


def _clear_loaded_layout(module: MetalCSRPropagator, incompatible_keys: object) -> None:
    module._clear_runtime()
    module.num_edges = int(module.edge_index.shape[1])
    _validated_edges(module.edge_index, module.num_nodes)
