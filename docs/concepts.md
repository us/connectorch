# Concepts

## Topology and parameters are separate things

Biology says:

```
neuron 182 -> neuron 991 exists, with 47 synapses
```

It does not say what floating-point number that connection should be in a neural
network. Those are two different questions, and ConnecTorch keeps them apart.

**Topology** is fixed. It lives in `edge_index`, a registered buffer, and no
optimizer step can change it.

**Parameterization** is an explicit choice you make:

| `weights=` | value |
|---|---|
| `"binary"` | 1.0 everywhere; connectivity only |
| `"synapse_count"` | the raw counts |
| `"normalized_synapse_count"` | `log1p(count) / sqrt(in-degree of target)` |
| `"weight"` | the edge table's own weight column |
| `"trainable"` | an `nn.Parameter`, started from `initializer=` |

`initializer="auto"` (the default) takes synapse counts when the graph has them,
otherwise its `weight` column, otherwise a binary adjacency. The choice it made is
on the model as `.initializer` and in its repr, never hidden.

None of these is a claim about synaptic efficacy. `normalized_synapse_count` is a
*numerical* initialisation: the `log1p` compresses the heavy tail of synapse
counts and the in-degree division keeps summed input roughly scale-free, which is
what stops a 164,587-node recurrent network from diverging in three steps.

ConnecTorch never picks one for you silently.

## Deterministic indexing

Biological ids are large and arbitrary. Internally every node gets a contiguous
index, assigned by **sorting the ids**, not by the order rows happened to appear
in your file.

```python
import connectorch as ct

brain = ct.Connectome.from_edges(source=[100, 5000, 7], target=[5000, 7, 100])
brain.node_ids        # [7, 100, 5000]
brain.index_of(5000)  # 2
brain.id_of(0)        # 7
```

Reading the same file with rows in a different order gives the same indices and
the same `fingerprint()`. That is what makes an experiment reproducible.

## Provenance

Nothing is ever dropped quietly.

```python
brain = ct.Connectome.from_edges([0, 1], [1, 2], synapse_count=[2, 9])

filtered = brain.filter_edges(min_synapses=5)
filtered.provenance["history"][-1]
# {'op': 'filter_edges', 'min_synapses': 5, 'edges_dropped': 1204}
```

Loaders record the dataset, its source URL, its license and every filter applied.

## Fingerprints

```python
brain.fingerprint()   # a short, stable hash of the structure
```

A hash of the structure: node ids, edge endpoints, and the weight columns. It
deliberately excludes free-text provenance, so re-downloading a dataset to a
different directory fingerprints identically. Record it with your results, and
anyone can check they have the same graph you did.
