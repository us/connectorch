"""NetworkX interoperability.

NetworkX is an optional dependency. It is imported inside the functions so that
``import connectorch`` stays fast and dependency-free for people who do not use it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import ConnectorchError
from ..ir import Connectome

__all__ = ["from_networkx", "to_networkx"]


def _require_networkx():  # type: ignore[no-untyped-def]
    try:
        import networkx
    except ImportError:
        raise ConnectorchError(
            "this needs networkx, which is an optional dependency. "
            "Install it with: pip install 'connectorch[networkx]'"
        ) from None
    return networkx


def from_networkx(
    graph: Any,
    *,
    weight: str | None = "weight",
    synapse_count: str | None = None,
    node_attributes: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Connectome:
    """Build a connectome from a NetworkX directed graph.

    Parameters
    ----------
    graph:
        A ``DiGraph`` or ``MultiDiGraph``. Undirected graphs are refused: a
        connectome is directed, and silently doubling every edge would be a lie
        about the data.
    weight:
        Edge attribute to read as the ``weight`` column, or ``None`` to skip.
        Missing on an edge means 1.0.
    synapse_count:
        Edge attribute to read as ``synapse_count``, or ``None`` to skip.
    node_attributes:
        Node attributes to carry across. Missing values become an empty string.
    """
    _require_networkx()
    if not graph.is_directed():
        raise ConnectorchError(
            "from_networkx needs a directed graph. Convert explicitly with "
            "graph.to_directed() if doubling every edge is what you mean."
        )

    edges = list(graph.edges(data=True))
    if not edges:
        raise ConnectorchError("the graph has no edges.")

    source = np.array([u for u, _, _ in edges])
    target = np.array([v for _, v, _ in edges])
    columns: dict[str, Any] = {}
    if weight is not None:
        columns["weight"] = np.array([float(d.get(weight, 1.0)) for _, _, d in edges])
    if synapse_count is not None:
        columns["synapse_count"] = np.array([int(d.get(synapse_count, 1)) for _, _, d in edges])

    # Always pass the node set explicitly. Inferring it from the edges would drop
    # every isolated neuron, and a neuron with no reconstructed connections is
    # still part of the connectome.
    node_ids = np.array(list(graph.nodes))
    nodes: dict[str, Any] = {"node_id": node_ids}
    for name in node_attributes or ():
        nodes[name] = np.array([str(graph.nodes[n].get(name, "")) for n in node_ids], dtype=str)

    record = dict(provenance or {})
    record.setdefault("source", f"networkx.{type(graph).__name__}")
    return Connectome(
        nodes=nodes, edges={"source": source, "target": target, **columns}, provenance=record
    )


def to_networkx(
    connectome: Connectome, *, weight_column: str | None = None, multigraph: bool | None = None
) -> Any:
    """Export a connectome as a NetworkX graph keyed by biological node id.

    Parameters
    ----------
    weight_column:
        Edge attribute to write as the NetworkX edge weight. Defaults to
        ``weight`` or ``synapse_count``, whichever is present.
    multigraph:
        Return a ``MultiDiGraph`` instead of a ``DiGraph``. ``None`` (the default)
        picks a multigraph only when the connectome actually holds parallel edges,
        since a ``DiGraph`` keeps one edge per pair and would silently drop the
        rest.
    """
    networkx = _require_networkx()
    source_index, target_index = connectome.edge_index
    if multigraph is None:
        pairs = np.stack([source_index, target_index], axis=1)
        multigraph = bool(np.unique(pairs, axis=0).shape[0] != connectome.num_edges)
    graph = networkx.MultiDiGraph() if multigraph else networkx.DiGraph()

    node_table = connectome.nodes.to_pydict()
    ids = node_table.pop("node_id")
    for index, node_id in enumerate(ids):
        graph.add_node(node_id, **{k: v[index] for k, v in node_table.items()})

    node_ids = connectome.node_ids
    if weight_column is None:
        for column in ("weight", "synapse_count"):
            if column in connectome.edge_columns:
                weight_column = column
                break

    values = connectome.edge_attribute(weight_column) if weight_column else None
    for i in range(connectome.num_edges):
        attributes: dict[str, Any] = (
            {str(weight_column): values[i].item()} if values is not None else {}
        )
        graph.add_edge(node_ids[source_index[i]], node_ids[target_index[i]], **attributes)
    return graph
