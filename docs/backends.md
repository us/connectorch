# Sparse backends

One recurrent step is

```
messages[target] = sum over edges (target <- source) of edge_weight * h[source]
```

which is `A @ h` for an adjacency `A[target, source]` that is nonzero only where
the connectome has an edge. State is node-major, `[num_nodes, batch]`, because
that is the orientation both sparse matrix multiplication and `index_add` want.

## Available backends

| backend | implementation | role |
|---|---|---|
| `scatter` | `h[source] * w`, then `index_add_` into targets | default training backend |
| `sparse_mm` | CSR adjacency, `torch.sparse.mm` | fast forward, fixed weights |
| `metal_csr` | native Metal CSR forward and backward kernels | explicit Apple GPU training or inference; CPU reference |
| `dense` | materialises `[N, N]` | correctness oracle, tiny graphs |
| `auto` | `scatter` when weights are trainable, else `sparse_mm` | the default |

## Apple GPU: explicit `metal_csr`

Select this backend explicitly; `auto` keeps its existing rules. It supports
first-order gradients for recurrent states and edge values, including values
produced by differentiable weight modules such as `BiologicalWeights`. The
topology stays fixed. Higher-order gradients are not supported.

MPS execution requires an Apple GPU, an MPS-enabled PyTorch installation and a
callable `torch.mps.compile_shader`. Shader compilation is lazy: importing
ConnecTorch or using another backend does not require this API. The original
experimental kernels ran with PyTorch 2.11.0; the development runtime is an
Apple M3 Max (128 GB), macOS 15.6.1 and PyTorch 2.11.0. This is not a new
package-wide minimum version. Check the actual runtime
capabilities, since the base `torch>=2.1` dependency alone is insufficient:

```python
# doctest: +SKIP -- requires an Apple GPU and the Metal shader API
import torch

if not torch.backends.mps.is_available():
    raise RuntimeError("This Python environment has no available MPS device")
if not callable(getattr(torch.mps, "compile_shader", None)):
    raise RuntimeError("This PyTorch build lacks torch.mps.compile_shader")
```

Run with `PYTORCH_ENABLE_MPS_FALLBACK=0`, set before Python starts. The native
backend does not silently copy propagation to CPU when MPS compilation or an
unsupported operation fails. MPS states and edge values must be **float32** on
the same device. Float16, bfloat16 and float64 MPS execution are unsupported.

```python
# doctest: +SKIP -- run with PYTORCH_ENABLE_MPS_FALLBACK=0 on an Apple GPU
import torch
import connectorch as ct

brain = ct.Connectome.from_edges(
    source=["A", "B", "C", "C"],
    target=["B", "C", "A", "B"],
    weight=[1.0, 0.5, 0.2, 0.8],
)
model = ct.nn.ConnectomeRNN(
    brain, weights="trainable", initializer="weight", backend="metal_csr",
    dtype=torch.float32,
).to("mps")
x = torch.randn(2, 4, brain.num_nodes, device="mps", dtype=torch.float32)
initial_state = torch.zeros(2, brain.num_nodes, device="mps", requires_grad=True)
y = model(x, state=initial_state)
y.square().mean().backward()
assert initial_state.grad is not None
assert model.edge_weight.grad is not None
```

The same backend has a CPU reference path accepting float32 or float64. Construct
the model on CPU, or move it to CPU, to compare outputs and gradients using
numerical tolerances. This is an explicit reference device, not an MPS fallback.
See [the runnable example](../examples/apple_metal.py):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/apple_metal.py
python examples/apple_metal.py --device cpu --dtype float64
```

Three Metal kernels compute forward propagation, the transposed propagation for
state gradients, and the batch reduction for edge gradients. They avoid both a
dense `[N, N]` adjacency and `[E, batch]` message tensors. This does **not** make
training memory zero: graph layouts, parameters and gradients, saved recurrent
states, output trajectories, and any optimizer state still consume memory.
Long sequences can still require truncated BPTT. The CUDA timings below do not
measure `metal_csr`; no Apple GPU speedup is claimed here.

CSR/transposed-CSR layouts and compiled shader caches are derived, nonpersistent
state. They are rebuilt for the active device and are not checkpoint contents.
For checkpoint portability, construct the same graph, interface and weight
parameterization with the destination backend, then load the model's
`state_dict`; do not serialize the compiled runtime. Edge values are supplied
on every call so optimizer or weight-module updates are not hidden by a cache.

The first propagation may run inside `torch.inference_mode()`; its derived
layout remains reusable for later training. Construct and move the full model
outside that context if you will train it later: PyTorch parameters and other
tensors created in inference mode cannot generally be saved for backward.

The original kernel contributions were adapted from
[fernando-neto-ai/fly-wordbrain](https://github.com/fernando-neto-ai/fly-wordbrain)
and contributed to this repository under its MIT license. This applies to those
contributions, not to the entire source repository or its third-party assets.
No Fly LLM weights or connectome datasets were copied into this implementation.

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

The default therefore uses **scatter for trainable weights and CSR for fixed
weights**. `backend="auto"` makes that choice; it does not select `metal_csr`.

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

Refusing a dense `[N, N]` is only half the job. The scatter training path
has its own appetite: each recurrent step keeps two `[num_edges, batch]` tensors
for the backward pass, so a training call holds

```
2 * num_edges * batch * steps * itemsize
```

On MaleCNS that is about 6.5 GiB at batch 4 over 8 steps, and 104 GiB at batch 32
over 16 steps. `ConnectomeRNN.activation_bytes(batch, steps)` computes it, the
model warns before a call that would exceed half the device's free memory, and
`torch.no_grad()` avoids those saved backward tensors. This formula measures
scatter message activations, not total model memory or `metal_csr` memory.
To train longer sequences than fit,
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
coordinate, which no longer lines up with `edge_weight`. `metal_csr` also
requires unique source/target pairs and rejects parallel edges. The default
`Connectome` construction aggregates them before compilation. `scatter` handles
unaggregated parallel edges natively.
