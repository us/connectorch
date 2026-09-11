"""Turning a connectome's edge attributes into numbers a network can use.

Biology says neuron 182 connects to neuron 991. It does not say what floating-point
number that connection should be. Every strategy here is an explicit, named choice,
and none of them is a claim about synaptic efficacy. Callers pick one; ConnecTorch
never guesses.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import ConnectorchError
from ..ir import Connectome
from ..ir.schema import SIGN, SYNAPSE_COUNT, WEIGHT

__all__ = ["WEIGHT_STRATEGIES", "initial_edge_weights", "resolve_strategy"]

#: Strategy name -> one-line description, used in docs and error messages.
WEIGHT_STRATEGIES: dict[str, str] = {
    "binary": "1.0 for every edge; connectivity only, no magnitude",
    "synapse_count": "the raw synapse_count column, as float",
    "normalized_synapse_count": (
        "log1p(synapse_count) / sqrt(in-degree of the target), a numerical "
        "initialisation that keeps a large recurrent graph from exploding"
    ),
    "weight": "the edge table's own weight column",
}


def resolve_strategy(connectome: Connectome, strategy: str) -> str:
    """Turn ``"auto"`` into the best strategy this connectome can actually supply.

    A graph carrying only a ``weight`` column cannot be initialised from synapse
    counts it does not have. Rather than failing on the first line a new user
    writes, ``"auto"`` picks the most informative column present. The choice is
    not a guess about biology and it is not hidden: it is stored on the model as
    ``initializer`` and printed in its repr.
    """
    if strategy != "auto":
        return strategy
    if SYNAPSE_COUNT in connectome.edge_columns:
        return "normalized_synapse_count"
    if WEIGHT in connectome.edge_columns:
        return "weight"
    return "binary"


def initial_edge_weights(connectome: Connectome, strategy: str) -> np.ndarray:
    """Compute the edge weight vector a model should start from.

    Parameters
    ----------
    connectome:
        The graph. Edge order is the connectome's canonical order.
    strategy:
        One of :data:`WEIGHT_STRATEGIES`.

    Returns
    -------
    numpy.ndarray
        ``[E]`` float64 array aligned with ``connectome.edge_index``.

    Notes
    -----
    ``normalized_synapse_count`` is a *numerical* initialisation strategy. The
    ``log1p`` compresses the heavy tail of synapse counts and the in-degree
    division keeps each neuron's summed input roughly scale-free, which is what
    stops a 166,691-node recurrent network from diverging in three steps. It is
    not a model of synaptic strength.
    """
    strategy = resolve_strategy(connectome, strategy)
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(
            f"unknown weight strategy {strategy!r}; available: "
            + ", ".join(f"{k!r} ({v})" for k, v in WEIGHT_STRATEGIES.items())
        )

    values = _magnitudes(connectome, strategy)
    return values * _signs(connectome)


def _magnitudes(connectome: Connectome, strategy: str) -> np.ndarray:
    """The unsigned weight each edge starts from, under the chosen strategy."""
    num_edges = connectome.num_edges
    if strategy == "binary":
        return np.ones(num_edges, dtype=np.float64)

    if strategy == "weight":
        return _require_column(connectome, WEIGHT, strategy).astype(np.float64)

    counts = _require_column(connectome, SYNAPSE_COUNT, strategy).astype(np.float64)
    if strategy == "synapse_count":
        return counts

    in_degree = np.bincount(connectome.edge_index[1], minlength=connectome.num_nodes).astype(
        np.float64
    )
    scale = np.sqrt(np.maximum(in_degree[connectome.edge_index[1]], 1.0))
    return np.log1p(counts) / scale


def _signs(connectome: Connectome) -> np.ndarray:
    """Edge polarity from the ``sign`` column, if the connectome carries one.

    An edge marked ``sign=-1`` is inhibitory, and a runtime that ignored that
    would turn it excitatory without saying so. Unknown (``0``) is treated as
    excitatory, since there is nothing better to do with it and pretending an
    unknown edge is silent would be a stronger claim.

    With ``weights="trainable"`` this only sets the starting point: gradient
    descent is free to move a weight across zero afterwards. Enforcing Dale's law
    for the whole of training is a v0.2 concern.
    """
    if SIGN not in connectome.edge_columns:
        return np.ones(connectome.num_edges, dtype=np.float64)
    signs = connectome.edge_attribute(SIGN).astype(np.float64)
    return np.where(signs == 0, 1.0, signs)


def _require_column(connectome: Connectome, name: str, strategy: str) -> np.ndarray:
    try:
        return connectome.edge_attribute(name)
    except KeyError:
        raise ConnectorchError(
            f"weights={strategy!r} needs an edge column {name!r}, but this "
            f"connectome only has {list(connectome.edge_columns)}. "
            "Use weights='binary' if the graph carries no edge magnitudes."
        ) from None
