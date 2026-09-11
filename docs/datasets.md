# Datasets

## MaleCNS v1.0

The male *Drosophila* central nervous system, from Janelia FlyEM, Google Research
and the Cambridge Drosophila Connectomics Group. Released June 2026, published in
Cell that September, CC-BY. The dataset annotates 166,691 neurons; the default
`traced-only` graph compiles to 164,587 nodes and 25,563,197 aggregated
connections. See [References](references.md).

```python
import connectorch as ct

fly = ct.datasets.malecns_sample()              # 100 neurons, offline, in the wheel
fly = ct.datasets.malecns(download=True)        # 164,587 neurons, ~520 MB
```

| variant | file size | what it is |
|---|---|---|
| `"traced-only"` (default) | 508 MB | connections between traced bodies |
| `"significant-only"` | 502 MB | statistically significant connections |
| `"full"` | 1.05 GB | everything at confidence >= 0.5 |

Nothing is downloaded without being asked. With the default `download=None` you get
a message naming the size and the URL first; `download=True` fetches,
`download=False` refuses. Downloads are resumable, checked against the expected
byte count, and written through a `.part` file so an interrupted run never leaves a
truncated file pretending to be complete.

Cache location: `$CONNECTORCH_CACHE`, else `~/.cache/connectorch`.

Metadata columns are renamed so they are valid Python keyword arguments: the source
file's `class` and `type` become `cell_class` and `cell_type`.

```python
import connectorch as ct

fly = ct.datasets.malecns_sample()      # the bundled subset, offline
fly.where(cell_type="Mi1")
```

Full citations for every dataset are on the [References](references.md) page.
If you publish work using one of these connectomes, cite the dataset's authors.

## neuPrint

For interactive work on subsets. Needs `pip install 'connectorch[neuprint]'` and a
token in `NEUPRINT_TOKEN`.

```python
from connectorch.io import from_neuprint
from neuprint import NeuronCriteria

brain = from_neuprint(dataset="male-cns:v1.0", criteria=NeuronCriteria(type="DNp01"))
```

For a whole connectome use the bulk loader instead. Asking a shared public server
for 25 million connections over REST is slow for you and rude to everyone else.

## Your own data

Anything with a source column and a target column works:

```python
from connectorch.io import read_edge_table, connectome_from_table

table = read_edge_table("my_connectome.csv")
brain = connectome_from_table(table, source_column="pre", target_column="post")
```

## Synthetic

```python
ct.datasets.chain(10)
ct.datasets.ring(8)
ct.datasets.random_sparse(num_nodes=10_000, num_edges=100_000, seed=0)
```

Not biology. For tests and benchmarks.

## The CLI

```bash
connectorch datasets list
connectorch datasets info male-cns
connectorch datasets download male-cns --variant traced-only
connectorch info brain.ct
connectorch validate edges.csv
```
