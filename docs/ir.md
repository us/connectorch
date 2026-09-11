# The Connectome IR

`Connectome` is the dataset-agnostic middle of the pipeline. It knows nothing about
PyTorch.

## Building one

```python
import connectorch as ct

# from arrays
brain = ct.Connectome.from_edges(source=[...], target=[...], synapse_count=[...])

# from tables
from connectorch.io import read_edge_table, connectome_from_table
table = read_edge_table("edges.parquet", columns=["body_pre", "body_post", "weight"])
brain = connectome_from_table(
    table,
    source_column="body_pre",
    target_column="body_post",
    attributes={"weight": "synapse_count"},
)

# from NetworkX
from connectorch.io import from_networkx
brain = from_networkx(graph, weight="weight")
```

## What is validated

Construction fails, loudly and with the offending row named, on: duplicate node
ids, null ids, edges referencing unknown nodes, NaN endpoints, negative synapse
counts, mixed integer and string ids, node columns whose length disagrees with the
id column, and signs outside `{-1, 0, +1}`.

```
ConnectomeValidationError: edge 291 references target node 50038, but that node is
not present in the node table (166691 nodes). 4 edge(s) have unknown endpoints.
Hint: pass nodes=None to infer the node set from the edges, or filter the edge
table first.
```

Self-loops are **preserved**. They are biologically real, and dropping them
silently would be a lie about the data. `filter_edges(drop_self_loops=True)`
removes them when you ask.

Parallel edges are aggregated by default, summing `synapse_count` and `weight`,
and the count merged is written to provenance.

## Querying

```python
import connectorch as ct

brain = ct.datasets.malecns_sample()

brain.where(cell_type="Mi1")                    # -> array of node ids
brain.where(cell_type=["Mi1", "TmY15"])         # membership
brain.where(cell_type="Mi1", side="")           # AND

brain.subgraph(brain.node_ids[:20])             # induced, reindexed
brain.filter_nodes(cell_type="Mi1")
brain.filter_edges(min_synapses=5)
```

Every transform returns a new connectome and appends to `provenance["history"]`.
Nothing mutates in place.

## Saving

```python
import tempfile, pathlib

path = pathlib.Path(tempfile.mkdtemp()) / "brain.ct"
brain.save(path)
brain = ct.Connectome.load(path)
```

```
brain.ct/
    metadata.json      schema version, counts, fingerprint, provenance
    nodes.parquet
    edges.parquet
```

Inspectable with any parquet reader, and no pickle anywhere. `load` verifies the
fingerprint and refuses a file that has been modified or truncated.
