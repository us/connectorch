"""The recurrent connectome runtime: a connectome as a ``torch.nn.Module``.

The update rule, for state ``h`` over ``N`` neurons and a batch of ``B``::

    m_t     = A @ h_t + u_t
    h_{t+1} = (1 - leak) * h_t + leak * activation(m_t + bias)

where ``A[target, source] = edge_weight`` has a nonzero entry only where the
connectome has an edge, and ``u_t`` is the external drive injected at the input
nodes.

The topology lives in buffers and the weights in a parameter. Gradient descent can
change what a connection is worth. It cannot create a connection that the animal
does not have.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor, nn

from ..backends import build_propagator
from ..exceptions import ConnectorchError
from ..ir import Connectome
from .dynamics import resolve_activation
from .weights import WEIGHT_STRATEGIES, initial_edge_weights, resolve_strategy

__all__ = ["ConnectomeRNN"]

_TRAJECTORY_WARN_ELEMENTS = 100_000_000

#: Fraction of a device's free memory that stored activations may claim before
#: the user is warned. Below this there is nothing worth interrupting them about.
_ACTIVATION_BUDGET_FRACTION = 0.5

#: Fallback budget when a device cannot report its free memory, as CPU cannot.
_ACTIVATION_BUDGET_BYTES = 4 * 2**30


def _memory_budget(device: torch.device) -> int:
    """How many bytes of activations are unremarkable on this device."""
    if device.type == "cuda" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info(device)
        return int(free * _ACTIVATION_BUDGET_FRACTION)
    return _ACTIVATION_BUDGET_BYTES


class ConnectomeRNN(nn.Module):
    """A rate-based recurrent network whose connectivity is a biological connectome.

    Parameters
    ----------
    connectome:
        The wiring diagram. Its canonical edge order fixes the meaning of every
        entry of :attr:`edge_weight`.
    input_nodes:
        Biological node ids that external input is injected into, in the order the
        columns of ``x`` will be given. ``None`` means every node, in index order.
    output_nodes:
        Biological node ids that are read out, in the order they appear in the
        output. ``None`` means every node, in index order.
    weights:
        ``"trainable"`` makes :attr:`edge_weight` an ``nn.Parameter``. Any other
        value from :data:`~connectorch.nn.weights.WEIGHT_STRATEGIES` makes it a
        fixed buffer holding that strategy's values.
    initializer:
        Which strategy provides the initial values when ``weights="trainable"``.
        ``"auto"`` (the default) uses synapse counts if the connectome has them,
        otherwise its ``weight`` column, otherwise a binary adjacency, and records
        the choice on the model. Ignored when the weights are fixed.
    activation:
        Name from :data:`~connectorch.nn.dynamics.ACTIVATIONS`, or a callable.
    leak:
        Mixing coefficient in ``(0, 1]``. ``1.0`` replaces the state each step;
        smaller values give the neuron a memory of its previous state.
    bias:
        Add a per-neuron learnable bias inside the activation.
    backend:
        ``"auto"``, ``"scatter"``, ``"sparse_mm"`` or ``"dense"``. ``"auto"``
        selects ``"scatter"`` when the weights are trainable, because the backward
        pass of sparse matrix multiplication materialises a dense ``[N, N]``
        gradient, and ``"sparse_mm"`` otherwise. The forward advantage of
        ``"sparse_mm"`` is large on CUDA and at larger batches, and reverses on
        CPU at batch 1, where ``"scatter"`` is faster; see the benchmarks.

    Examples
    --------
    >>> import torch, connectorch as ct
    >>> brain = ct.Connectome.from_edges(
    ...     source=["A", "B", "C", "C"],
    ...     target=["B", "C", "A", "B"],
    ...     weight=[1.0, 0.5, 0.2, 0.8],
    ... )
    >>> model = ct.nn.ConnectomeRNN(brain, weights="trainable", initializer="weight")
    >>> y = model(torch.randn(8, brain.num_nodes), steps=5)
    >>> y.shape
    torch.Size([8, 5, 3])
    """

    edge_index: Tensor
    edge_weight: Tensor
    input_index: Tensor
    output_index: Tensor

    def __init__(
        self,
        connectome: Connectome,
        *,
        input_nodes: np.ndarray | list | None = None,
        output_nodes: np.ndarray | list | None = None,
        weights: str = "trainable",
        initializer: str = "auto",
        activation: str | Callable[[Tensor], Tensor] = "tanh",
        leak: float = 1.0,
        bias: bool = False,
        backend: str = "auto",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if connectome.num_edges == 0:
            raise ConnectorchError("cannot build a ConnectomeRNN from a connectome with no edges.")
        if not 0.0 < leak <= 1.0:
            raise ValueError(f"leak must be in (0, 1], got {leak}.")

        dtype = dtype or torch.get_default_dtype()
        trainable = weights == "trainable"
        if not trainable and weights not in WEIGHT_STRATEGIES:
            raise ValueError(
                f"unknown weights={weights!r}; use 'trainable' or one of "
                f"{sorted(WEIGHT_STRATEGIES)}."
            )
        strategy = resolve_strategy(connectome, initializer if trainable else weights)

        self.num_nodes = connectome.num_nodes
        self.num_edges = connectome.num_edges
        self.leak = float(leak)
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self._activation = resolve_activation(activation)
        self.weights_mode = weights
        self.initializer = strategy
        self.fingerprint = connectome.fingerprint()

        edge_index = torch.as_tensor(connectome.edge_index, dtype=torch.int64)
        self.register_buffer("edge_index", edge_index)

        values = torch.as_tensor(initial_edge_weights(connectome, strategy), dtype=dtype)
        if trainable:
            self.edge_weight = nn.Parameter(values)
        else:
            self.register_buffer("edge_weight", values)

        input_index = _resolve_nodes(connectome, input_nodes)
        output_index = _resolve_nodes(connectome, output_nodes)
        self.register_buffer("input_index", input_index)
        self.register_buffer("output_index", output_index)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_nodes, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.backend_requested = backend
        self.propagator = build_propagator(
            backend, edge_index, connectome.num_nodes, trainable=trainable
        )
        self.backend = getattr(self.propagator, "backend_name", backend)

        # load_state_dict writes edge_index directly into our buffer; the
        # backend's derived layout has to follow or the model would propagate
        # along the topology it used to have.
        self.register_load_state_dict_post_hook(_rebuild_propagator)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------

    @property
    def num_input_nodes(self) -> int:
        """How many columns ``x`` must have."""
        return int(self.input_index.numel())

    @property
    def num_output_nodes(self) -> int:
        """How many columns the output has."""
        return int(self.output_index.numel())

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        steps: int | None = None,
        *,
        state: Tensor | None = None,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the recurrence and read out the output nodes at every step.

        Parameters
        ----------
        x:
            ``[batch, num_input_nodes]`` for a drive held constant across steps, or
            ``[batch, steps, num_input_nodes]`` for a time-varying drive consumed
            one slice per step.
        steps:
            Number of recurrent steps. Required for a constant drive; for a
            sequence drive it defaults to that sequence's length and must match it.
        state:
            ``[batch, num_nodes]`` initial state. Defaults to zeros. The module is
            stateless between calls: nothing is carried over implicitly.
        return_state:
            Also return the final ``[batch, num_nodes]`` state, for continuing the
            run in a later call.

        Returns
        -------
        Tensor or (Tensor, Tensor)
            Output of shape ``[batch, steps, num_output_nodes]``, and the final
            state if ``return_state`` is set.
        """
        drive, steps = self._prepare_input(x, steps)
        batch = drive.shape[0]
        self._warn_if_trajectory_is_huge(batch, steps)
        self._warn_if_activations_are_huge(batch, steps)

        h = self._prepare_state(state, batch)
        weight = self.edge_weight
        outputs = []

        for t in range(steps):
            messages = self.propagator(h, weight).index_add(0, self.input_index, drive[:, t, :].t())
            if self.bias is not None:
                messages = messages + self.bias.unsqueeze(-1)
            activated = self._activation(messages)
            h = (1.0 - self.leak) * h + self.leak * activated
            outputs.append(h.index_select(0, self.output_index).t())

        y = torch.stack(outputs, dim=1)
        return (y, h.t()) if return_state else y

    # ------------------------------------------------------------------

    def _prepare_input(self, x: Tensor, steps: int | None) -> tuple[Tensor, int]:
        """Normalise ``x`` to ``[batch, steps, num_input_nodes]`` and settle ``steps``."""
        if x.dim() == 2:
            if steps is None:
                raise ValueError(
                    "steps is required for a constant drive of shape "
                    "[batch, num_input_nodes]. Pass steps=..., or give x as "
                    "[batch, steps, num_input_nodes]."
                )
            if x.shape[1] != self.num_input_nodes:
                raise ValueError(
                    f"x has {x.shape[1]} input columns but the model has "
                    f"{self.num_input_nodes} input nodes."
                )
            drive = x.unsqueeze(1).expand(x.shape[0], steps, x.shape[1])
        elif x.dim() == 3:
            if x.shape[2] != self.num_input_nodes:
                raise ValueError(
                    f"x has {x.shape[2]} input columns but the model has "
                    f"{self.num_input_nodes} input nodes."
                )
            if steps is None:
                steps = int(x.shape[1])
            elif steps != x.shape[1]:
                raise ValueError(
                    f"steps={steps} does not match the sequence length {x.shape[1]} "
                    "of x. Leave steps unset to use the sequence length."
                )
            drive = x
        else:
            raise ValueError(
                f"x must be [batch, num_input_nodes] or [batch, steps, "
                f"num_input_nodes], got shape {tuple(x.shape)}."
            )

        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}.")
        return drive, int(steps)

    def _prepare_state(self, state: Tensor | None, batch: int) -> Tensor:
        """Return the node-major ``[N, B]`` starting state, in the model's dtype."""
        if state is None:
            return torch.zeros(
                (self.num_nodes, batch),
                dtype=self.edge_weight.dtype,
                device=self.edge_weight.device,
            )
        if state.shape != (batch, self.num_nodes):
            raise ValueError(
                f"state must have shape [batch, num_nodes] = "
                f"[{batch}, {self.num_nodes}], got {tuple(state.shape)}."
            )
        return state.t().contiguous()

    def get_extra_state(self) -> dict[str, object]:
        """Carry the connectome's identity into the checkpoint.

        ``edge_index`` is a persistent buffer, so a checkpoint can replace this
        model's topology. Without the fingerprint travelling alongside it, the
        loaded model would report the connectome it was built from while
        propagating along the one it was given.
        """
        return {
            "fingerprint": self.fingerprint,
            "weights_mode": self.weights_mode,
            "initializer": self.initializer,
            "leak": self.leak,
            "activation": self.activation_name,
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        """Adopt the identity of the checkpoint being loaded."""
        self.fingerprint = str(state.get("fingerprint", self.fingerprint))
        for name in ("weights_mode", "initializer", "activation"):
            if name in state:
                setattr(self, f"{name}_name" if name == "activation" else name, state[name])
        if "leak" in state:
            self.leak = float(state["leak"])  # type: ignore[arg-type]

    def activation_bytes(self, batch: int, steps: int) -> int:
        """Bytes of per-edge activations backpropagation will hold for this call.

        The library refuses to allocate a dense ``[N, N]`` adjacency, but the path
        it recommends instead has its own appetite: every recurrent step stores
        two ``[num_edges, batch]`` tensors for the backward pass, the gathered
        source states and the weighted messages. That is

            2 * num_edges * batch * steps * itemsize

        which is linear in everything and easy to walk into. On MaleCNS
        (25,563,197 connections) a batch of 32 over 16 steps wants about 104 GiB.

        Returns zero when no gradient is being recorded, since nothing is stored.
        """
        if not (self.edge_weight.requires_grad and torch.is_grad_enabled()):
            return 0
        return 2 * self.num_edges * batch * steps * self.edge_weight.element_size()

    def _warn_if_activations_are_huge(self, batch: int, steps: int) -> None:
        """Say what this backward pass will cost before it is paid, not after."""
        needed = self.activation_bytes(batch, steps)
        if needed <= _memory_budget(self.edge_weight.device):
            return
        gib = needed / 2**30
        warnings.warn(
            f"this backward pass will hold about {gib:,.1f} GiB of per-edge "
            f"activations ({self.num_edges:,} edges x batch {batch} x {steps} "
            "steps x 2 tensors).\n"
            "Reduce batch or steps, run inference under torch.no_grad(), or train "
            "in segments by passing the returned state into the next call.",
            stacklevel=3,
        )

    def _warn_if_trajectory_is_huge(self, batch: int, steps: int) -> None:
        """Warn before allocating an output trajectory that dwarfs the model itself."""
        elements = batch * steps * self.num_output_nodes
        if elements > _TRAJECTORY_WARN_ELEMENTS:
            gib = elements * self.edge_weight.element_size() / 2**30
            warnings.warn(
                f"this call will build a [{batch}, {steps}, "
                f"{self.num_output_nodes}] output trajectory, about {gib:.1f} GiB. "
                "Pass output_nodes= to read out fewer neurons, or use fewer steps.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, float | int | str]:
        """Report the numbers that predict whether this network will explode."""
        with torch.no_grad():
            weight = self.edge_weight.detach()
            target = self.edge_index[1]
            in_degree = torch.bincount(target, minlength=self.num_nodes)
            out_degree = torch.bincount(self.edge_index[0], minlength=self.num_nodes)
            row_sum = torch.zeros(self.num_nodes, dtype=weight.dtype, device=weight.device)
            row_sum = row_sum.index_add(0, target, weight.abs())
            return {
                "nodes": self.num_nodes,
                "edges": self.num_edges,
                "backend": self.backend,
                "weights": self.weights_mode,
                "max_in_degree": int(in_degree.max()),
                "max_out_degree": int(out_degree.max()),
                "weight_mean": float(weight.mean()),
                "weight_std": float(weight.std()) if weight.numel() > 1 else 0.0,
                "weight_absmax": float(weight.abs().max()),
                "max_abs_row_sum": float(row_sum.max()),
                "activation_bytes_per_batch_step": 2
                * self.num_edges
                * self.edge_weight.element_size(),
            }

    def extra_repr(self) -> str:
        return (
            f"nodes={self.num_nodes:,}, edges={self.num_edges:,}, "
            f"weights={self.weights_mode!r}, initializer={self.initializer!r}, "
            f"activation={self.activation_name!r}, leak={self.leak}, "
            f"backend={self.backend!r}"
        )


def _rebuild_propagator(module: ConnectomeRNN, incompatible_keys: object) -> None:
    """Post-``load_state_dict`` hook: re-derive the backend from the loaded topology."""
    module.propagator.rebuild(module.edge_index)


def _resolve_nodes(connectome: Connectome, nodes: np.ndarray | list | None) -> Tensor:
    """Map user-supplied node ids to an int64 index tensor, or select all nodes."""
    if nodes is None:
        return torch.arange(connectome.num_nodes, dtype=torch.int64)
    ids = np.asarray(nodes)
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.size == 0:
        raise ConnectorchError("input_nodes/output_nodes cannot be empty.")
    index = np.atleast_1d(np.asarray(connectome.index_of(ids), dtype=np.int64))
    return torch.as_tensor(index, dtype=torch.int64)
