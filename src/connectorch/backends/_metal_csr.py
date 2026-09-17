"""Sparse multiplication with dynamic edge values and native Apple GPU gradients.

``TrainableCSR(ptr, indices, shape, device).mm(h, values)`` computes ``h @ W.T``.
Values and their gradients always use the supplied CSR edge order, including
unsorted columns and explicitly stored zero values. The topology is fixed;
callers decide how to parameterize values and preserve their biological signs.

Metal forward and state backward use CSR SIMD reductions. The edge backward
directly sums over batch, so no [edges, batch] message tensor is materialized
or saved. First-order autograd only; CPU float64 supports numerical gradcheck.

Adapted from original Metal kernel contributions in
https://github.com/fernando-neto-ai/fly-wordbrain. These contributions are
provided under ConnecTorch's MIT license; see the repository LICENSE.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from ..exceptions import BackendError

__all__ = ["TrainableCSR"]


METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void dynamic_csr_forward(
    device float* output, const device float* input,
    const device int* ptr, const device int* col,
    const device float* value, const device int* shape,
    uint tid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint rows = uint(shape[0]), cols = uint(shape[1]);
    const uint out = tid / 32, row = out % rows, batch = out / rows;
    float partial = 0.0f;
    for (uint e = uint(ptr[row]) + uint(lane); e < uint(ptr[row + 1]); e += 32)
        partial += value[e] * input[batch * cols + uint(col[e])];
    const float total = simd_sum(partial);
    if (lane == 0) output[out] = total;
}

kernel void dynamic_csr_transpose(
    device float* output, const device float* input,
    const device int* ptr, const device int* col,
    const device int* canonical_edge, const device float* value,
    const device int* shape,
    uint tid [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const uint rows = uint(shape[1]), cols = uint(shape[0]);
    const uint out = tid / 32, row = out % rows, batch = out / rows;
    float partial = 0.0f;
    for (uint e = uint(ptr[row]) + uint(lane); e < uint(ptr[row + 1]); e += 32)
        partial += value[canonical_edge[e]] * input[batch * cols + uint(col[e])];
    const float total = simd_sum(partial);
    if (lane == 0) output[out] = total;
}

kernel void dynamic_csr_edge_backward(
    device float* output, const device float* input,
    const device float* grad_output,
    const device int* row, const device int* col,
    const device int* shape, constant uint& batch_size,
    uint edge [[thread_position_in_grid]]) {
    const uint rows = uint(shape[0]), cols = uint(shape[1]);
    float total = 0.0f;
    for (uint b = 0; b < batch_size; ++b)
        total += grad_output[b * rows + uint(row[edge])] * input[b * cols + uint(col[edge])];
    output[edge] = total;
}
"""

METAL_SOURCE_SHA256 = hashlib.sha256(METAL_SOURCE.encode()).hexdigest()
_LIBRARY: Any = None
_COMPILE_LOCK = threading.Lock()


def _require_mps() -> None:
    if not torch.backends.mps.is_available() or not callable(
        getattr(torch.mps, "compile_shader", None)
    ):
        raise BackendError(
            "metal_csr requires an available Apple MPS device and torch.mps.compile_shader; "
            "no CPU fallback is provided"
        )


def _compiled_library() -> Any:
    global _LIBRARY
    if _LIBRARY is None:
        _require_mps()
        with _COMPILE_LOCK:
            if _LIBRARY is None:
                _LIBRARY = torch.mps.compile_shader(METAL_SOURCE)
    return _LIBRARY


def _index_array(value: Tensor | ArrayLike, name: str) -> NDArray[np.int64]:
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise BackendError(f"Topology cannot require gradients: {name}")
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise BackendError(f"{name} must be a one-dimensional integer array")
    if value.size and (value.min() < 0 or value.max() >= np.iinfo(np.int32).max):
        raise BackendError(f"{name} entries must be nonnegative and below 2^31-1")
    return value.astype(np.int64, copy=True)


def _fingerprint(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainableCSR:
    """An immutable topology cache; dynamic values are supplied on every call.

    The object is deliberately not a module: graph indices are derived caches,
    not checkpoint parameters. Reconstruct it when changing the model's device
    or topology. Input arrays are copied. ``verify_frozen`` performs an explicit
    host audit; ordinary ``mm`` calls check tensor identity/version without a
    device synchronization or host data transfer.
    """

    def __init__(
        self,
        ptr: Tensor | ArrayLike,
        indices: Tensor | ArrayLike,
        shape: Sequence[int],
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise BackendError("Invalid metal_csr device") from error
        if device.type not in ("cpu", "mps"):
            raise BackendError("TrainableCSR supports only CPU and MPS")
        if device.type == "mps":
            _require_mps()
        if len(shape) != 2 or any(
            isinstance(x, bool | np.bool_)
            or not isinstance(x, int | np.integer)
            or x < 1
            or x >= np.iinfo(np.int32).max
            for x in shape
        ):
            raise BackendError("shape must contain two positive integer dimensions below 2^31-1")
        self.shape = torch.Size(map(int, shape))
        rows, columns = self.shape
        ptr, indices = _index_array(ptr, "ptr"), _index_array(indices, "indices")
        if (
            len(ptr) != rows + 1
            or ptr[0] != 0
            or ptr[-1] != len(indices)
            or np.any(np.diff(ptr) < 0)
        ):
            raise BackendError("CSR pointer lengths, endpoints, or monotonicity are inconsistent")
        if len(indices) and indices.max() >= columns:
            raise BackendError("CSR column index out of bounds")
        self.nnz = int(len(indices))
        edge_rows = np.repeat(np.arange(rows, dtype=np.int64), np.diff(ptr))
        # Sorting only for validation leaves the original edge order untouched.
        order = np.lexsort((indices, edge_rows))
        if self.nnz > 1 and np.any(
            (edge_rows[order[1:]] == edge_rows[order[:-1]])
            & (indices[order[1:]] == indices[order[:-1]])
        ):
            raise BackendError(
                "Duplicate edges are unsupported; topology must not be silently coalesced"
            )
        transpose_edges = np.argsort(indices, kind="stable")
        transpose_ptr = np.r_[0, np.cumsum(np.bincount(indices, minlength=columns))]
        arrays = {
            "ptr": ptr,
            "col": indices,
            "row": edge_rows,
            "transpose_ptr": transpose_ptr,
            "transpose_col": edge_rows[transpose_edges],
            "transpose_edge": transpose_edges,
            "shape": np.array(self.shape),
        }
        if device.type == "cpu":
            # Native torch CSR requires sorted rows. Metal accepts the canonical
            # ordering directly; CPU uses a derived permutation without changing
            # the caller's edge values or gradient order.
            arrays.update(sorted_col=indices[order], sorted_edge=order)
        dtype = torch.int32 if device.type == "mps" else torch.int64
        # Lazy construction is allowed inside inference_mode. Cache tensors must
        # still have version counters and remain usable by later training calls.
        with torch.inference_mode(False):
            self._buffers = {
                name: torch.tensor(value, dtype=dtype, device=device)
                for name, value in arrays.items()
            }
        self._identities = {
            name: (id(value), value._version) for name, value in self._buffers.items()
        }
        self._fingerprint = _fingerprint(self._buffers)
        self._shape_at_construction = self.shape
        self._nnz_at_construction = self.nnz
        self.kernel_launch_count = {"forward": 0, "state_backward": 0, "edge_backward": 0}

    @property
    def device(self) -> torch.device:
        return self._buffers["ptr"].device

    def _check_cache(self) -> None:
        if (
            self.shape != self._shape_at_construction
            or self.nnz != self._nnz_at_construction
            or self._buffers.keys() != self._identities.keys()
            or any(
                (id(value), value._version) != self._identities[name]
                for name, value in self._buffers.items()
            )
        ):
            raise RuntimeError("TrainableCSR topology cache was mutated; reconstruct it explicitly")

    def verify_frozen(self) -> bool:
        """Audit every canonical and derived cache, including writes via .data."""
        self._check_cache()
        if _fingerprint(self._buffers) != self._fingerprint:
            raise RuntimeError("TrainableCSR topology cache fingerprint changed")
        return True

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "metal_trainable_csr" if self.device.type == "mps" else "cpu_trainable_csr",
            "shape": list(self.shape),
            "edges": self.nnz,
            "source_sha256": METAL_SOURCE_SHA256,
            "topology_sha256": self._fingerprint,
            "edge_order": "supplied CSR order",
            "edge_weight_gradients": True,
            "first_order_only": True,
            "dense_adjacency": False,
            "edge_batch_intermediate": False,
            "per_step_host_transfers": False,
            "host_fallback": False,
            "kernel_launch_count": dict(self.kernel_launch_count),
        }

    def mm(self, h: Tensor, values: Tensor) -> Tensor:
        self._check_cache()
        if not isinstance(h, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise BackendError("h and values must be torch tensors")
        if h.layout != torch.strided or values.layout != torch.strided:
            raise BackendError("h and values must have strided dense layout")
        if h.device != self.device or values.device != self.device:
            raise BackendError("h and values must be resident on the topology cache device")
        supported = (
            (torch.float32,) if self.device.type == "mps" else (torch.float32, torch.float64)
        )
        if h.dtype not in supported or values.dtype != h.dtype:
            raise BackendError(
                "h and values must have matching float32 dtype (CPU also accepts float64)"
            )
        if h.ndim != 2 or h.shape[1] != self.shape[1] or values.shape != (self.nnz,):
            raise BackendError("Expected h[batch, columns] and values[edges]")
        # The shaders index outputs with uint32, including their SIMD lanes.
        if h.shape[0] * max(self.shape) * 32 >= 2**32:
            raise BackendError("Batch and graph dimensions exceed Metal dispatch indexing limits")
        return _TrainableMM.apply(h, values, self)

    def _multiply(self, h: Tensor, values: Tensor, transpose: bool = False) -> Tensor:
        batch = h.shape[0]
        rows = self.shape[1] if transpose else self.shape[0]
        if batch == 0 or self.nnz == 0:
            return torch.zeros((batch, rows), dtype=h.dtype, device=h.device)
        buffers = self._buffers
        if self.device.type == "cpu":
            ptr = buffers["transpose_ptr"] if transpose else buffers["ptr"]
            col = buffers["transpose_col"] if transpose else buffers["sorted_col"]
            edge = buffers["transpose_edge"] if transpose else buffers["sorted_edge"]
            matrix = torch.sparse_csr_tensor(
                ptr,
                col,
                values[edge],
                size=tuple(reversed(self.shape)) if transpose else self.shape,
                check_invariants=False,
            )  # Fully validated at construction.
            return torch.sparse.mm(matrix, h.T).T.contiguous()
        output = torch.empty((batch, rows), dtype=h.dtype, device=h.device)
        if transpose:
            _compiled_library().dynamic_csr_transpose(
                output,
                h.contiguous(),
                buffers["transpose_ptr"],
                buffers["transpose_col"],
                buffers["transpose_edge"],
                values.contiguous(),
                buffers["shape"],
                threads=int(batch * rows * 32),
                group_size=128,
            )
        else:
            _compiled_library().dynamic_csr_forward(
                output,
                h.contiguous(),
                buffers["ptr"],
                buffers["col"],
                values.contiguous(),
                buffers["shape"],
                threads=int(batch * rows * 32),
                group_size=128,
            )
        self.kernel_launch_count["state_backward" if transpose else "forward"] += 1
        return output

    def _edge_gradient(self, h: Tensor, grad_output: Tensor) -> Tensor:
        if h.shape[0] == 0 or self.nnz == 0:
            return torch.zeros((self.nnz,), dtype=h.dtype, device=h.device)
        buffers = self._buffers
        if self.device.type == "cpu":
            result = torch.zeros((self.nnz,), dtype=h.dtype, device=h.device)
            # O(edges) temporary storage, independent of batch size.
            for b in range(h.shape[0]):
                result.add_(h[b, buffers["col"]] * grad_output[b, buffers["row"]])
            return result
        result = torch.empty((self.nnz,), dtype=h.dtype, device=h.device)
        _compiled_library().dynamic_csr_edge_backward(
            result,
            h.contiguous(),
            grad_output.contiguous(),
            buffers["row"],
            buffers["col"],
            buffers["shape"],
            int(h.shape[0]),
            threads=self.nnz,
            group_size=128,
        )
        self.kernel_launch_count["edge_backward"] += 1
        return result


class _TrainableMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, h: Tensor, values: Tensor, graph: TrainableCSR) -> Tensor:
        ctx.graph = graph
        ctx.save_for_backward(h, values)
        return graph._multiply(h, values)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor | None, Tensor | None, None]:
        (grad_output,) = grad_outputs
        if torch.is_grad_enabled():
            raise RuntimeError(
                "TrainableCSR supports first-order gradients only; create_graph/higher-order derivatives are unsupported"
            )
        ctx.graph._check_cache()
        h, values = ctx.saved_tensors
        dh = (
            ctx.graph._multiply(grad_output, values, transpose=True)
            if ctx.needs_input_grad[0]
            else None
        )
        dv = ctx.graph._edge_gradient(h, grad_output) if ctx.needs_input_grad[1] else None
        return dh, dv, None
