"""Small synthetic connectomes for tests, examples and benchmarks.

These are not biology. They exist so that a five-line example runs instantly and so
that benchmarks have graphs of a controlled size.
"""

from __future__ import annotations

import numpy as np

from ..ir import Connectome

__all__ = ["chain", "ring", "random_sparse"]


def chain(num_nodes: int = 3, weight: float = 1.0) -> Connectome:
    """A feed-forward chain ``0 -> 1 -> ... -> n-1``."""
    source = np.arange(num_nodes - 1)
    return Connectome.from_edges(
        source=source,
        target=source + 1,
        weight=np.full(num_nodes - 1, float(weight)),
        nodes={"node_id": np.arange(num_nodes)},
        provenance={"source": f"toy.chain(num_nodes={num_nodes})"},
    )


def ring(num_nodes: int = 4, weight: float = 1.0) -> Connectome:
    """A directed ring; every node drives the next, and the last drives the first."""
    source = np.arange(num_nodes)
    return Connectome.from_edges(
        source=source,
        target=(source + 1) % num_nodes,
        weight=np.full(num_nodes, float(weight)),
        nodes={"node_id": source},
        provenance={"source": f"toy.ring(num_nodes={num_nodes})"},
    )


def random_sparse(
    num_nodes: int = 1_000,
    num_edges: int = 10_000,
    *,
    seed: int = 0,
    max_synapses: int = 50,
) -> Connectome:
    """A random sparse graph with synapse counts, for benchmarks and smoke tests.

    Parallel edges are aggregated, so the result usually has slightly fewer than
    ``num_edges`` edges.
    """
    rng = np.random.default_rng(seed)
    return Connectome.from_edges(
        source=rng.integers(0, num_nodes, num_edges),
        target=rng.integers(0, num_nodes, num_edges),
        synapse_count=rng.integers(1, max_synapses, num_edges),
        nodes={"node_id": np.arange(num_nodes)},
        provenance={
            "source": (
                f"toy.random_sparse(num_nodes={num_nodes}, num_edges={num_edges}, seed={seed})"
            )
        },
    )
