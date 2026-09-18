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
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from ..backends import Propagator, build_propagator
from ..exceptions import ConnectorchError
from ..ir import Connectome
from .dynamics import resolve_activation
from .parameterisation import EdgeWeights, FixedWeights, FreeWeights
from .weights import WEIGHT_STRATEGIES, initial_edge_weights, resolve_strategy

__all__ = ["ConnectomeRNN", "MAX_SYNAPTIC_DELAY"]

_TRAJECTORY_WARN_ELEMENTS = 100_000_000

#: Largest synaptic delay, in recurrent steps. Bounds the history buffer at
#: ``MAX_SYNAPTIC_DELAY + 1`` states of ``[N, B]``. Heterogeneous delays are
#: the mechanism behind every published fly motion computation (fast Mi1/Tm3
#: center against slow Mi4/Mi9 flanks); a uniform delay is only a lag.
MAX_SYNAPTIC_DELAY = 4

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
        ``"trainable"`` gives every connection its own free parameter. Any other
        name from :data:`~connectorch.nn.weights.WEIGHT_STRATEGIES` freezes the
        weights at that strategy's values.

        You can also pass an
        :class:`~connectorch.nn.parameterisation.EdgeWeights` module, which is how
        the biology is kept in the loop rather than optimised away::

            weights=ct.nn.BiologicalWeights(brain, share_by="cell_type", dale=True)
    initializer:
        Which strategy provides the initial values when ``weights="trainable"``.
        ``"auto"`` (the default) uses synapse counts if the connectome has them,
        otherwise its ``weight`` column, otherwise a binary adjacency, and records
        the choice on the model. Ignored when the weights are fixed.
    activation:
        Name from :data:`~connectorch.nn.dynamics.ACTIVATIONS`, or a callable.
        ``"threshold_linear"`` is the graded FlyVis-style output nonlinearity
        (silent below threshold, linear above); with ``bias=True`` the
        threshold is learnable per neuron.
    leak:
        Mixing coefficient in ``(0, 1]``. ``1.0`` replaces the state each step;
        smaller values give the neuron a memory of its previous state.
    leak_by:
        Optional ``{group: leak}`` mapping that gives different neuron groups
        different memories, e.g. fast photoreceptor input and slow integrators.
        Groups are read from the ``leak_column`` node column; neurons whose
        group is absent from the mapping fall back to ``leak``. ``None`` (the
        default) uses ``leak`` for every neuron.
    leak_column:
        Node column that ``leak_by`` keys on. Defaults to ``"cell_type"``.
    delay_by:
        Optional ``{source group: steps}`` mapping of synaptic delays. An edge
        whose *source* neuron belongs to group ``g`` is delivered
        ``delay_by[g]`` steps late, so slow flanks (Mi4/Mi9) and fast centers
        (Mi1/Tm3) arrive at T4 at different times, which is the coincidence
        mechanism behind direction selectivity. Groups are read from the
        ``delay_column`` node column; edges from unmapped groups have delay 0.
        Each value must be an integer in ``[0, MAX_SYNAPTIC_DELAY]``. ``None``
        (the default) is exactly the old behavior: every edge has delay 0.
    delay_column:
        Node column that ``delay_by`` keys on. Defaults to ``"cell_type"``.
    bias:
        Add a per-neuron learnable bias inside the activation.
    backend:
        ``"auto"``, ``"scatter"``, ``"sparse_mm"``, ``"dense"`` or ``"metal_csr"``. ``"auto"``
        selects ``"scatter"`` when the weights are trainable, because the backward
        pass of sparse matrix multiplication materialises a dense ``[N, N]``
        gradient, and ``"sparse_mm"`` otherwise. The forward advantage of
        ``"sparse_mm"`` is large on CUDA and at larger batches, and reverses on
        CPU at batch 1, where ``"scatter"`` is faster; see the benchmarks.
        Select ``"metal_csr"`` explicitly for native float32 Apple GPU training
        on ``device="mps"``; it requires ``torch.mps.compile_shader`` and supports
        first-order gradients without dense adjacency or per-edge/batch messages.

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
    input_index: Tensor
    output_index: Tensor
    weights: EdgeWeights
    leak_vec: Tensor | None
    delay_propagators: nn.ModuleList

    def __init__(
        self,
        connectome: Connectome,
        *,
        input_nodes: np.ndarray | list | None = None,
        output_nodes: np.ndarray | list | None = None,
        weights: str | EdgeWeights = "trainable",
        initializer: str = "auto",
        activation: str | Callable[[Tensor], Tensor] = "tanh",
        leak: float = 1.0,
        leak_by: dict[str, float] | None = None,
        leak_column: str = "cell_type",
        delay_by: dict[str, int] | None = None,
        delay_column: str = "cell_type",
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
        given_module = isinstance(weights, EdgeWeights)
        if isinstance(weights, EdgeWeights):
            if weights.num_edges != connectome.num_edges:
                raise ConnectorchError(
                    f"the weight module covers {weights.num_edges:,} connections "
                    f"but this connectome has {connectome.num_edges:,}. It has to "
                    "be built from the same connectome."
                )
            trainable = any(p.requires_grad for p in weights.parameters())
            strategy = str(weights.describe().get("kind", "custom"))
        else:
            trainable = weights == "trainable"
            if not trainable and weights not in WEIGHT_STRATEGIES:
                raise ValueError(
                    f"unknown weights={weights!r}; use 'trainable', one of "
                    f"{sorted(WEIGHT_STRATEGIES)}, or an EdgeWeights module."
                )
            strategy = resolve_strategy(connectome, initializer if trainable else weights)

        self.num_nodes = connectome.num_nodes
        self.num_edges = connectome.num_edges
        self.leak = float(leak)
        self.activation_name = activation if isinstance(activation, str) else "custom"
        self._activation = resolve_activation(activation)
        self.weights_mode = strategy if given_module else str(weights)
        self.initializer = strategy
        self.fingerprint = connectome.fingerprint()

        edge_index = torch.as_tensor(connectome.edge_index, dtype=torch.int64)
        self.register_buffer("edge_index", edge_index)

        if isinstance(weights, EdgeWeights):
            self.weights = weights
        else:
            values = torch.as_tensor(initial_edge_weights(connectome, strategy), dtype=dtype)
            self.weights = FreeWeights(values) if trainable else FixedWeights(values)

        input_index = _resolve_nodes(connectome, input_nodes)
        output_index = _resolve_nodes(connectome, output_nodes)
        self.register_buffer("input_index", input_index)
        self.register_buffer("output_index", output_index)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_nodes, dtype=dtype))
            if self.activation_name == "threshold_linear":
                # The threshold-linear unit fires above messages == 1. A zero
                # bias would start the whole network silent, with zero gradient
                # everywhere and no way back. Starting the bias at the
                # threshold makes the unit a ReLU at birth; training then
                # moves each neuron's effective threshold where the task pays.
                with torch.no_grad():
                    self.bias.fill_(1.0)
        else:
            self.register_parameter("bias", None)

        self.backend_requested = backend
        self.propagator = build_propagator(
            backend, edge_index, connectome.num_nodes, trainable=trainable
        )
        self.backend = getattr(self.propagator, "backend_name", backend)

        self.leak_by = dict(leak_by) if leak_by is not None else None
        self.leak_column = leak_column
        leak_vec = _leak_vector(connectome, float(leak), leak_by, leak_column, dtype)
        if leak_vec is not None:
            self.register_buffer("leak_vec", leak_vec)
        else:
            self.leak_vec = None

        self.delay_by = {k: int(v) for k, v in delay_by.items()} if delay_by else None
        self.delay_column = delay_column
        delay_values, delay_masks = _delay_groups(connectome, delay_by, delay_column)
        self.delay_values: list[int] = delay_values
        self.max_delay: int = max(delay_values) if delay_values else 0
        self.delay_propagators = nn.ModuleList()
        for i, mask in enumerate(delay_masks):
            self.register_buffer(f"delay_mask_{i}", torch.as_tensor(mask, dtype=torch.int64))
            self.delay_propagators.append(
                build_propagator(
                    backend,
                    edge_index[:, torch.as_tensor(mask)],
                    connectome.num_nodes,
                    trainable=trainable,
                )
            )

        # load_state_dict writes edge_index directly into our buffer; the
        # backend's derived layout has to follow or the model would propagate
        # along the topology it used to have.
        self.register_load_state_dict_post_hook(_rebuild_propagator)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------

    @property
    def edge_weight(self) -> Tensor:
        """The current weight of every connection, in canonical edge order.

        Produced by :attr:`weights`; with a free parameterisation this *is* the
        parameter, with a biological one it is the measured prior times a bounded
        learned gain.
        """
        return self.weights()

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

        if self.max_delay == 0:
            for t in range(steps):
                messages = self.propagator(h, weight).index_add(
                    0, self.input_index, drive[:, t, :].t()
                )
                if self.bias is not None:
                    messages = messages + self.bias.unsqueeze(-1)
                activated = self._activation(messages)
                h = self._apply_leak(h, activated)
                outputs.append(h.index_select(0, self.output_index).t())
        else:
            # Delayed delivery: an edge from source group g is computed from
            # the state `delay_by[g]` steps ago. History starts full of the
            # initial state, i.e. the network rests before the stimulus.
            history: list[Tensor] = [h] * (self.max_delay + 1)
            for t in range(steps):
                messages = torch.zeros_like(h)
                for i, d in enumerate(self.delay_values):
                    propagator = self.delay_propagators[i]
                    mask = getattr(self, f"delay_mask_{i}")
                    messages = messages + propagator(history[-1 - d], weight[mask])
                messages = messages.index_add(0, self.input_index, drive[:, t, :].t())
                if self.bias is not None:
                    messages = messages + self.bias.unsqueeze(-1)
                activated = self._activation(messages)
                h = self._apply_leak(h, activated)
                history.append(h)
                del history[0]
                outputs.append(h.index_select(0, self.output_index).t())

        y = torch.stack(outputs, dim=1)
        return (y, h.t()) if return_state else y

    # ------------------------------------------------------------------

    def _apply_leak(self, h: Tensor, activated: Tensor) -> Tensor:
        """Mix the old state with the new activation, per neuron if configured."""
        if self.leak_vec is not None:
            return (1.0 - self.leak_vec) * h + self.leak_vec * activated
        return (1.0 - self.leak) * h + self.leak * activated

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
                dtype=self.weights.dtype,
                device=self.weights.device,
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
            "leak_by": dict(self.leak_by) if self.leak_by is not None else None,
            "leak_column": self.leak_column,
            "delay_by": dict(self.delay_by) if self.delay_by is not None else None,
            "delay_column": self.delay_column,
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
        for name in ("leak_by", "leak_column", "delay_by", "delay_column"):
            if name in state:
                setattr(self, name, state[name])

    def activation_bytes(self, batch: int, steps: int) -> int:
        """Estimate saved per-edge/batch activations, not total training memory.

        The scatter training path stores two ``[num_edges, batch]`` tensors
        per recurrent step for the backward pass, the gathered
        source states and the weighted messages. That is

            2 * num_edges * batch * steps * itemsize

        which is linear in everything and easy to walk into. On MaleCNS
        (25,563,197 connections) a batch of 32 over 16 steps wants about 104 GiB.

        Returns zero when no gradient is being recorded or when the backend
        avoids saved per-edge/batch tensors, as ``metal_csr`` does. Neuron states,
        edge values, gradients, topology, and other workspace still consume memory.
        """
        trains = any(p.requires_grad for p in self.weights.parameters())
        if not (
            trains and torch.is_grad_enabled() and self.propagator.saves_edge_batch_activations
        ):
            return 0
        return (
            2
            * self.num_edges
            * batch
            * steps
            * torch.empty((), dtype=self.weights.dtype).element_size()
        )

    def _warn_if_activations_are_huge(self, batch: int, steps: int) -> None:
        """Say what this backward pass will cost before it is paid, not after."""
        needed = self.activation_bytes(batch, steps)
        if needed <= _memory_budget(self.weights.device):
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
            gib = elements * torch.empty((), dtype=self.weights.dtype).element_size() / 2**30
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
                "max_delay": self.max_delay,
                "leak_groups": len(self.leak_by) if self.leak_by is not None else 0,
                "activation_bytes_per_batch_step": int(self.propagator.saves_edge_batch_activations)
                * 2
                * self.num_edges
                * torch.empty((), dtype=self.weights.dtype).element_size(),
            }

    def extra_repr(self) -> str:
        extras = ""
        if self.leak_by:
            extras += f", leak_by={len(self.leak_by)} groups"
        if self.max_delay:
            extras += f", delays={self.delay_values}"
        return (
            f"nodes={self.num_nodes:,}, edges={self.num_edges:,}, "
            f"weights={self.weights_mode!r}, initializer={self.initializer!r}, "
            f"activation={self.activation_name!r}, leak={self.leak}, "
            f"backend={self.backend!r}{extras}"
        )


def _leak_vector(
    connectome: Connectome,
    leak: float,
    leak_by: dict[str, float] | None,
    leak_column: str,
    dtype: torch.dtype,
) -> Tensor | None:
    """Per-neuron leak coefficients, or ``None`` when one leak fits all."""
    if leak_by is None:
        return None
    if leak_column not in connectome.node_columns:
        raise ConnectorchError(
            f"no node column {leak_column!r} to read leak groups from; available: "
            f"{list(connectome.node_columns)}."
        )
    for group, value in leak_by.items():
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"leak for group {group!r} must be in (0, 1], got {value}.")
    labels = connectome.nodes.column(leak_column).to_pylist()
    vec = torch.tensor(
        [float(leak_by.get("" if v is None else str(v), leak)) for v in labels],
        dtype=dtype,
    ).unsqueeze(-1)
    return vec


def _delay_groups(
    connectome: Connectome,
    delay_by: dict[str, int] | None,
    delay_column: str,
) -> tuple[list[int], list[np.ndarray]]:
    """Partition edge positions by synaptic delay, from the source neuron's group."""
    if not delay_by:
        return [], []
    if delay_column not in connectome.node_columns:
        raise ConnectorchError(
            f"no node column {delay_column!r} to read delay groups from; available: "
            f"{list(connectome.node_columns)}."
        )
    for group, value in delay_by.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | np.integer)
            or not 0 <= int(value) <= MAX_SYNAPTIC_DELAY
        ):
            raise ValueError(
                f"delay for group {group!r} must be an integer in "
                f"[0, {MAX_SYNAPTIC_DELAY}], got {value}."
            )
    labels = [
        "" if v is None else str(v) for v in connectome.nodes.column(delay_column).to_pylist()
    ]
    source = np.asarray(connectome.edge_index[0])
    per_edge = np.array([int(delay_by.get(labels[s], 0)) for s in source], dtype=np.int64)
    values = [int(d) for d in sorted(np.unique(per_edge))]
    return values, [(per_edge == d).nonzero()[0] for d in values]


def _rebuild_propagator(module: ConnectomeRNN, incompatible_keys: object) -> None:
    """Post-``load_state_dict`` hook: re-derive the backend from the loaded topology."""
    module.propagator.rebuild(module.edge_index)
    for i in range(len(module.delay_values)):
        mask = getattr(module, f"delay_mask_{i}")
        if int(mask.numel()) and int(mask.max()) >= int(module.edge_index.shape[1]):
            raise ConnectorchError(
                "the checkpoint's delay masks do not fit the loaded topology "
                f"(mask indexes edge {int(mask.max())} of "
                f"{int(module.edge_index.shape[1])}). Load a checkpoint built "
                "from the same connectome and delay mapping."
            )
        sub = cast(Propagator, module.delay_propagators[i])
        sub.rebuild(module.edge_index[:, mask])


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
