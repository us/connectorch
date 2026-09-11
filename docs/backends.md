# Sparse backends

One recurrent step is

```
messages[target] = sum over edges (target <- source) of edge_weight * h[source]
```

which is `A @ h` for an adjacency `A[target, source]` that is nonzero only where
the connectome has an edge. State is node-major, `[num_nodes, batch]`, because
that is the orientation both sparse matrix multiplication and `index_add` want.

## The three backends

| backend | implementation | role |
|---|---|---|
| `scatter` | `h[source] * w`, then `index_add_` into targets | **the training backend** |
| `sparse_mm` | CSR adjacency, `torch.sparse.mm` | fast forward, fixed weights |
| `dense` | materialises `[N, N]` | correctness oracle, tiny graphs |
| `auto` | `scatter` when weights are trainable, else `sparse_mm` | the default |

## Why `auto` chooses what it chooses

!!! warning "torch.sparse.mm densifies on the backward pass"
    The backward of `torch.sparse.mm` with respect to a sparse operand
    materialises a dense `[N, N]` intermediate before restricting the result to
    the sparsity pattern, even though the gradient it returns is sparse-typed.
    This is a known, still-open PyTorch bug, not a ConnecTorch limitation and not
    something a caller can opt out of:
    [pytorch#41128](https://github.com/pytorch/pytorch/issues/41128),
    [pytorch#49683](https://github.com/pytorch/pytorch/issues/49683).
    `torch.sparse` remains a beta API.

Measured on an NVIDIA GB10 with torch 2.14.0+cu130, N=50,000, 500k edges, batch 32.
A dense float32 `[50,000 x 50,000]` is 9.3 GiB:

| operation | time | peak GPU memory |
|---|---|---|
| COO forward+backward | 74.6 ms | 9.40 GiB |
| CSR forward+backward | 140.4 ms | 11.74 GiB |
| **scatter forward+backward** | **3.3 ms** | **0.24 GiB** |
| COO forward only | 0.49 ms | 0.08 GiB |
| **CSR forward only** | **0.16 ms** | 0.07 GiB |
| scatter forward only | 1.76 ms | 0.18 GiB |

Read the peak memory column twice: 9.40 GiB against a theoretical 9.3 GiB dense
matrix is not a coincidence. At N=164,587 the same path asks for 100.9 GiB and
fails.

So: **train with scatter, infer with CSR.** `backend="auto"` does that by looking
at whether `edge_weight.requires_grad`.

Requesting `sparse_mm` with trainable weights on a large graph is refused before
anything is allocated:

```
ConnectorchMemoryError: backend="sparse_mm" received edge weights that require
gradients, and its backward pass materialises a dense adjacency gradient for
164,587 nodes.
A torch.float32 [164,587 x 164,587] adjacency needs 100.9 GiB.
Use backend="scatter", which allocates O(E) instead.
```

## Reproducing these numbers

```bash
python benchmarks/sparse_backends.py --out results.jsonl
```

`benchmarks/gb10-results.jsonl` holds the run quoted above. No performance claim
in this project is made without a file like that behind it.

## The other memory invariant

Refusing a dense `[N, N]` is only half the job. The path this library recommends
has its own appetite: each recurrent step keeps two `[num_edges, batch]` tensors
for the backward pass, so a training call holds

```
2 * num_edges * batch * steps * itemsize
```

On MaleCNS that is about 6.5 GiB at batch 4 over 8 steps, and 104 GiB at batch 32
over 16 steps. `ConnectomeRNN.activation_bytes(batch, steps)` computes it, the
model warns before a call that would exceed half the device's free memory, and
`torch.no_grad()` reduces it to nothing. To train longer sequences than fit,
run them in segments and carry the state forward:

```python
state = None                                              # doctest: +SKIP
for chunk in chunks:                                      # doctest: +SKIP
    out, state = model(chunk, state=state, return_state=True)
    loss(out).backward()
    state = state.detach()
```

Gradient checkpointing and truncated BPTT as built-in options are a later
version's job; the guard exists so nobody discovers this at hour three of a run.

## Determinism

`index_add_` uses atomics on CUDA, so messages arriving at the same target are
summed in an unspecified order and float32 results can differ in the last bits
between runs. There is no deterministic CUDA kernel for `index_add_`, so
`torch.use_deterministic_algorithms(True)` raises rather than falling back to
one; pass `warn_only=True` if you need that flag on for other reasons in the same
run ([pytorch#108569](https://github.com/pytorch/pytorch/issues/108569)).

## Parallel edges

A CSR tensor holds one value per coordinate. If you build a connectome with
`aggregate_parallel_edges=False` and it contains two edges between the same pair,
`sparse_mm` refuses it: its backward would return one gradient per *unique*
coordinate, which no longer lines up with `edge_weight`. `scatter` handles
parallel edges natively.
