"""How a connection's number is produced, and what training is allowed to do to it.

The default runtime gives every connection its own free parameter. That keeps the
topology but throws the biology away: after enough updates a pair joined by 2
synapses can end up stronger than a pair joined by 200, and the wiring diagram
survives while everything it was telling you does not.

This module is the alternative. A connection's weight is

    weight = sign * prior * gain

where ``prior`` comes from the measured synapse count, ``sign`` from the
presynaptic neuron's transmitter, and ``gain`` is the only thing gradient descent
touches. The gain is bounded, so learning can modulate what biology measured
without overruling it, and it can be **shared across connections between the same
pair of cell types**, which is both far fewer parameters and closer to how a
nervous system is actually organised: neurons of a type follow a rule, they are
not each wired by hand.

Pass one of these where a weight strategy name would go::

    core = ct.nn.ConnectomeRNN(
        brain,
        weights=ct.nn.BiologicalWeights(brain, share_by="cell_type", dale=True),
    )
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ..exceptions import ConnectorchError
from ..ir import Connectome
from ..ir.schema import SIGN
from .weights import initial_edge_weights

__all__ = ["EdgeWeights", "FreeWeights", "FixedWeights", "BiologicalWeights"]


class EdgeWeights(nn.Module):
    """Produces the ``[num_edges]`` weight vector the runtime propagates with."""

    num_edges: int

    def forward(self) -> Tensor:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the weights, without computing them."""
        return next(self._reference())

    @property
    def device(self) -> torch.device:
        """Device of the weights, without computing them."""
        return next(self._reference_device())

    def _reference(self):  # type: ignore[no-untyped-def]
        for tensor in list(self.parameters()) + list(self.buffers()):
            yield tensor.dtype
        yield torch.get_default_dtype()

    def _reference_device(self):  # type: ignore[no-untyped-def]
        for tensor in list(self.parameters()) + list(self.buffers()):
            yield tensor.device
        yield torch.device("cpu")

    def describe(self) -> dict[str, object]:
        """What this parameterisation is, for the model's repr and provenance."""
        return {"kind": type(self).__name__}


class FreeWeights(EdgeWeights):
    """One unconstrained parameter per connection. Maximum freedom, no biology left."""

    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.num_edges = int(values.numel())
        self.weight = nn.Parameter(values)

    def forward(self) -> Tensor:
        return self.weight

    def describe(self) -> dict[str, object]:
        return {"kind": "free", "parameters": self.num_edges}


class FixedWeights(EdgeWeights):
    """Frozen weights; the connectome is a constant and only what wraps it learns."""

    weight: Tensor

    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.num_edges = int(values.numel())
        self.register_buffer("weight", values)

    def forward(self) -> Tensor:
        return self.weight

    def describe(self) -> dict[str, object]:
        return {"kind": "fixed", "parameters": 0}


class BiologicalWeights(EdgeWeights):
    """Measured synapse counts, modulated by a bounded and optionally shared gain.

    Parameters
    ----------
    connectome:
        The graph. Its synapse counts become the prior and, if present, its
        ``sign`` column becomes the polarity.
    prior:
        Which strategy turns synapse counts into the fixed part of the weight.
        Defaults to ``"normalized_synapse_count"``.
    gain_bounds:
        ``(low, high)`` multiplicative range the gain is confined to. The default
        ``(0.5, 2.0)`` lets learning halve or double what was measured and no
        more. Pass ``None`` for an unbounded gain, which gives up the guarantee.
    share_by:
        Node column whose values group connections that share one gain, typically
        ``"cell_type"``. ``None`` gives every connection its own gain. Sharing is
        by the *pair* of groups: all connections from cell type A to cell type B
        move together.
    dale:
        Hold each neuron's polarity fixed through training, so an excitatory
        neuron cannot become inhibitory. Requires a ``sign`` edge column; see
        :func:`connectorch.transforms.infer_signs`.

    Notes
    -----
    The bound is applied through a sigmoid, so the gain is smooth everywhere and
    can never leave its range regardless of learning rate. It starts at 1.0, which
    means the model starts as the measured connectome and departs from it only as
    far as the data pushes it.
    """

    prior_values: Tensor
    signs: Tensor
    group_index: Tensor

    def __init__(
        self,
        connectome: Connectome,
        *,
        prior: str = "normalized_synapse_count",
        gain_bounds: tuple[float, float] | None = (0.5, 2.0),
        share_by: str | None = None,
        dale: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.num_edges = connectome.num_edges
        self.prior_name = prior
        self.share_by = share_by
        self.dale = bool(dale)

        if gain_bounds is not None:
            low, high = float(gain_bounds[0]), float(gain_bounds[1])
            if not 0 < low < high:
                raise ConnectorchError(
                    f"gain_bounds must satisfy 0 < low < high, got {gain_bounds}."
                )
            if not low <= 1.0 <= high:
                raise ConnectorchError(
                    f"gain_bounds {gain_bounds} exclude 1.0, so the model could "
                    "never express the connectome as measured."
                )
        self.gain_bounds = None if gain_bounds is None else (low, high)

        magnitudes = np.abs(initial_edge_weights(connectome, prior))
        self.register_buffer("prior_values", torch.as_tensor(magnitudes, dtype=dtype))

        if dale:
            if SIGN not in connectome.edge_columns:
                raise ConnectorchError(
                    "dale=True needs a 'sign' edge column saying which connections "
                    "are excitatory and which inhibitory. Produce one with "
                    "connectorch.transforms.infer_signs(brain, mapping), or set "
                    "dale=False to let training choose signs freely."
                )
            signs = connectome.edge_attribute(SIGN).astype(np.float64)
            signs = np.where(signs == 0, 1.0, signs)
        else:
            signs = np.sign(initial_edge_weights(connectome, prior))
            signs = np.where(signs == 0, 1.0, signs)
        self.register_buffer("signs", torch.as_tensor(signs, dtype=dtype))

        group_index, num_groups = _grouping(connectome, share_by)
        self.register_buffer("group_index", torch.as_tensor(group_index, dtype=torch.int64))
        self.num_groups = num_groups
        # Zero maps to a gain of exactly 1.0 under both parameterisations, so the
        # model begins life as the connectome that was measured.
        self.raw_gain = nn.Parameter(torch.zeros(num_groups, dtype=dtype))

    def gain(self) -> Tensor:
        """The multiplier applied to the prior, one value per connection."""
        if self.gain_bounds is None:
            per_group = torch.exp(self.raw_gain)
        else:
            low, high = self.gain_bounds
            # sigmoid(0) = 0.5, and the offset places 1.0 at raw_gain == 0.
            offset = float(np.log((1.0 - low) / (high - 1.0))) if high > 1.0 else 0.0
            per_group = low + (high - low) * torch.sigmoid(self.raw_gain + offset)
        return per_group.index_select(0, self.group_index)

    def forward(self) -> Tensor:
        return self.signs * self.prior_values * self.gain()

    def describe(self) -> dict[str, object]:
        return {
            "kind": "biological",
            "prior": self.prior_name,
            "gain_bounds": self.gain_bounds,
            "share_by": self.share_by,
            "dale": self.dale,
            "parameters": self.num_groups,
            "connections": self.num_edges,
        }

    def extra_repr(self) -> str:
        shared = f"share_by={self.share_by!r}, " if self.share_by else ""
        return (
            f"prior={self.prior_name!r}, gain_bounds={self.gain_bounds}, "
            f"{shared}dale={self.dale}, "
            f"{self.num_groups:,} gains for {self.num_edges:,} connections"
        )


def _grouping(connectome: Connectome, share_by: str | None) -> tuple[np.ndarray, int]:
    """Map each connection to the gain it shares, by (source group, target group)."""
    if share_by is None:
        return np.arange(connectome.num_edges, dtype=np.int64), connectome.num_edges

    if share_by not in connectome.node_columns:
        raise ConnectorchError(
            f"no node column {share_by!r} to share gains by; available: "
            f"{list(connectome.node_columns)}."
        )
    labels = np.asarray(connectome.nodes.column(share_by).to_pylist(), dtype=object)
    # An unlabelled neuron is its own group rather than sharing one big "" group,
    # which would tie thousands of unrelated connections to a single parameter.
    codes = np.empty(labels.size, dtype=np.int64)
    seen: dict[str, int] = {}
    for i, label in enumerate(labels):
        text = "" if label is None else str(label)
        if text == "":
            codes[i] = -(i + 1)
        else:
            codes[i] = seen.setdefault(text, len(seen))

    source, target = connectome.edge_index
    pairs = np.stack([codes[source], codes[target]], axis=1)
    _, group_index = np.unique(pairs, axis=0, return_inverse=True)
    group_index = group_index.reshape(-1).astype(np.int64)
    return group_index, int(group_index.max()) + 1
