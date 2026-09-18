"""Topology-matched control graphs.

Any claim that biological wiring is a useful inductive bias has to be measured
against controls, or the result might just be showing that sparsity helps. These
transforms build the controls: same size, same degrees, different wiring.

A result comparing a connectome only against a dense network is not evidence about
connectomes.
"""

from __future__ import annotations

import numpy as np

from ..ir import Connectome
from ..ir.schema import SIGN, SYNAPSE_COUNT, WEIGHT

__all__ = [
    "random_topology",
    "degree_preserving_rewire",
    "shuffle_edge_weights",
    "shuffle_signs",
    "collapse_ei",
]


def random_topology(connectome: Connectome, *, seed: int = 0) -> Connectome:
    """A random graph with the same node count and *exactly* the same edge count.

    The weakest control, and the one that tells you whether sparsity alone
    explains a result. Degrees are not preserved.

    Drawing endpoints independently would produce duplicate pairs, which the IR
    then aggregates, leaving the control with fewer edges than the graph it is a
    control for. A comparison at a different edge count is not a control at all,
    so this oversamples and deduplicates until the counts match.
    """
    rng = np.random.default_rng(seed)
    num_nodes, num_edges = connectome.num_nodes, connectome.num_edges
    if num_edges > num_nodes * num_nodes:
        raise ValueError(f"cannot draw {num_edges:,} distinct edges from {num_nodes:,} nodes.")

    pairs = np.empty((0, 2), dtype=np.int64)
    while pairs.shape[0] < num_edges:
        wanted = int((num_edges - pairs.shape[0]) * 1.5) + 16
        drawn = np.stack(
            [rng.integers(0, num_nodes, wanted), rng.integers(0, num_nodes, wanted)], axis=1
        )
        pairs = np.unique(np.concatenate([pairs, drawn]), axis=0)
    # np.unique already sorted them; take a random subset so the truncation is
    # not biased towards low node indices.
    pairs = pairs[rng.permutation(pairs.shape[0])[:num_edges]]

    columns = {name: connectome.edge_attribute(name) for name in connectome.edge_columns}
    return _rebuild(
        connectome,
        pairs[:, 0],
        pairs[:, 1],
        columns,
        _control_provenance(connectome, "random_topology", seed),
    )


def degree_preserving_rewire(
    connectome: Connectome, *, seed: int = 0, swaps_per_edge: int = 10
) -> Connectome:
    """Rewire by double-edge swaps, keeping every neuron's in- and out-degree.

    The control that matters. It leaves the degree sequence exactly intact and
    destroys everything else about the wiring, so a difference between this and
    the real connectome cannot be explained by degree distribution.

    Parameters
    ----------
    swaps_per_edge:
        Attempted swaps per edge. Ten is the usual rule of thumb for mixing.

    Notes
    -----
    A swap takes edges ``a -> b`` and ``c -> d`` and produces ``a -> d`` and
    ``c -> b``. Swaps that would create a duplicate of an existing edge are
    rejected, so the edge count is preserved exactly. Edge attributes travel with
    their source endpoint.
    """
    rng = np.random.default_rng(seed)
    source, target = (array.copy() for array in connectome.edge_index)
    num_edges = source.size

    existing = set(zip(source.tolist(), target.tolist(), strict=True))
    accepted = 0
    for _ in range(num_edges * swaps_per_edge):
        i, j = rng.integers(0, num_edges, 2)
        if i == j:
            continue
        a, b, c, d = source[i], target[i], source[j], target[j]
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        existing.add((a, d))
        existing.add((c, b))
        target[i], target[j] = d, b
        accepted += 1

    provenance = _control_provenance(connectome, "degree_preserving_rewire", seed)
    provenance["history"][-1]["accepted_swaps"] = int(accepted)
    return _rebuild(
        connectome,
        source,
        target,
        {name: connectome.edge_attribute(name) for name in connectome.edge_columns},
        provenance,
    )


def shuffle_edge_weights(connectome: Connectome, *, seed: int = 0) -> Connectome:
    """Keep the wiring exactly, permute the synapse counts across edges.

    Separates "which neurons are connected" from "how strongly". If shuffling the
    counts costs nothing, the model was only ever using the topology.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(connectome.num_edges)
    columns = {}
    for name in connectome.edge_columns:
        values = connectome.edge_attribute(name)
        columns[name] = values[permutation] if name in (SYNAPSE_COUNT, WEIGHT) else values

    source, target = connectome.edge_index
    return _rebuild(
        connectome,
        source,
        target,
        columns,
        _control_provenance(connectome, "shuffle_edge_weights", seed),
    )


def shuffle_signs(connectome: Connectome, *, seed: int = 0) -> Connectome:
    """Keep wiring and counts, permute the ``sign`` column across edges.

    Separates "which polarity each connection has" from everything else. If
    shuffling signs costs nothing, the model was not using E/I identity.
    Requires a ``sign`` edge column; see
    :func:`connectorch.transforms.infer_signs`.
    """
    if SIGN not in connectome.edge_columns:
        from ..exceptions import ConnectorchError

        raise ConnectorchError(
            "shuffle_signs needs a 'sign' edge column. Produce one with "
            "connectorch.transforms.infer_signs(brain, mapping)."
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(connectome.num_edges)
    columns = {}
    for name in connectome.edge_columns:
        values = connectome.edge_attribute(name)
        columns[name] = values[permutation] if name == SIGN else values

    source, target = connectome.edge_index
    return _rebuild(
        connectome,
        source,
        target,
        columns,
        _control_provenance(connectome, "shuffle_signs", seed),
    )


def collapse_ei(connectome: Connectome) -> Connectome:
    """Keep wiring and counts, set every edge sign to +1 (no inhibition).

    The FlyVis ablation: collapsing excitatory/inhibitory/mixed types
    performs like random wiring. If this costs nothing, E/I identity is not
    doing work in the task.
    """
    if SIGN not in connectome.edge_columns:
        from ..exceptions import ConnectorchError

        raise ConnectorchError(
            "collapse_ei needs a 'sign' edge column. Produce one with "
            "connectorch.transforms.infer_signs(brain, mapping)."
        )
    columns = {}
    for name in connectome.edge_columns:
        values = connectome.edge_attribute(name)
        columns[name] = np.ones_like(values) if name == SIGN else values

    source, target = connectome.edge_index
    record = dict(connectome.provenance)
    record["history"] = [
        *record.get("history", []),
        {"op": "collapse_ei", "note": "all signs set to +1; inhibition removed"},
    ]
    return _rebuild(connectome, source, target, columns, record)


def _rebuild(
    connectome: Connectome, source: np.ndarray, target: np.ndarray, columns: dict, provenance: dict
) -> Connectome:
    """Rebuild a connectome from rewired endpoints, keeping its aggregation policy.

    A control built with the default ``aggregate_parallel_edges=True`` out of a
    graph that deliberately kept its parallel edges would silently come back
    smaller, and then it is a control for a different graph.
    """
    return Connectome(
        nodes={"node_id": connectome.node_ids, **_node_columns(connectome)},
        edges={
            "source": connectome.node_ids[source],
            "target": connectome.node_ids[target],
            **columns,
        },
        aggregate_parallel_edges=bool(connectome.provenance.get("aggregate_parallel_edges", True)),
        provenance=provenance,
    )


def _node_columns(connectome: Connectome) -> dict[str, np.ndarray]:
    table = connectome.nodes
    return {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name != "node_id"
    }


def _control_provenance(connectome: Connectome, operation: str, seed: int) -> dict:
    record = dict(connectome.provenance)
    record["history"] = [
        *record.get("history", []),
        {
            "op": operation,
            "seed": int(seed),
            "control_for": connectome.fingerprint(),
            "note": "a topology-matched control, not biological data",
        },
    ]
    return record
