# ConnecTorch

**Compile biological connectomes into trainable PyTorch networks.**

Real neurons. Real connectivity. `torch.nn.Module`.

```python
import torch
import connectorch as ct

fly = ct.datasets.malecns_sample()          # 100 real neurons from a fly's CNS
net = ct.nn.ConnectomeRNN(fly, weights="trainable")

y = net(torch.randn(8, fly.num_nodes), steps=10)
y.square().mean().backward()                # gradients through biological wiring

print(net.edge_weight.grad)                 # one gradient per real connection
```

```bash
pip install connectorch
```

## Why

A connectome is a wiring diagram: which neuron connects to which, and how strongly.
Enormous ones now exist. The whole male *Drosophila* central nervous system was
released in June 2026 and published in *Cell* that September: **166,691 annotated
neurons**. ConnecTorch's default traced-only graph compiles to **164,587 nodes and
25,563,197 aggregated connections**, carrying 124 million synapses between them.

ConnecTorch turns one into a sparse recurrent network you can train with ordinary
PyTorch, and keeps one promise while doing it:

> **Gradient descent can change what a connection is worth.
> It cannot create a connection the animal does not have.**

```
arbitrary connectome -> Connectome IR -> compiler/runtime -> torch.nn.Module
```

Nothing in that pipeline is specific to any one species. MaleCNS is the showcase,
not the architecture.

## It actually runs on the whole fly

Measured on an NVIDIA GB10, reproduced with exactly this command:

```bash
python examples/03_malecns_load.py --download --cuda --batch-size 4 --steps 8
```

| | |
|---|---|
| nodes compiled | 164,587 |
| aggregated connections | 25,563,197 |
| synapses they represent | 124,025,046 |
| load and compile | 9.5 s from cache, 3.1 GB RAM |
| trainable parameters | 25,563,197, one per connection |
| forward+backward (batch 4, 8 steps) | 1,129 ms, **7.84 GiB** VRAM |
| a dense adjacency for the same graph | 100.9 GiB |

The library refuses to allocate that dense matrix rather than trying and dying:

```
ConnectorchMemoryError: backend="dense" was requested for 164,587 nodes.
A torch.float32 [164,587 x 164,587] adjacency needs 100.9 GiB.
Use backend="scatter", which allocates O(E) instead.
```

## Quick start

```python
import torch
import connectorch as ct

brain = ct.Connectome.from_edges(
    source=["A", "B", "C", "C"],
    target=["B", "C", "A", "B"],
    weight=[1.0, 0.5, 0.2, 0.8],
)

model = ct.nn.ConnectomeRNN(
    brain,
    weights="trainable",      # or "synapse_count", "normalized_synapse_count", "binary"
    activation="tanh",
    leak=0.5,
)

y = model(torch.randn(8, brain.num_nodes), steps=5)   # [batch, steps, output nodes]
```

Real datasets, with metadata:

```python
fly = ct.datasets.malecns(download=True)       # ~520 MB, cached and resumable
optic = fly.where(cell_class="visual")
model = ct.nn.ConnectomeRNN(fly, input_nodes=optic, output_nodes=fly.where(cell_class="CX"))
```

## Backends

One recurrent step is `A @ h`, where `A[target, source]` is nonzero only where the
connectome has an edge.

| backend | what it does | when |
|---|---|---|
| `scatter` | gather source states, `index_add` into targets | **training** |
| `sparse_mm` | CSR adjacency, `torch.sparse.mm` | inference, fixed weights |
| `dense` | materialises `[N, N]` | correctness oracle, tiny graphs only |
| `auto` | `scatter` if the weights are trainable, else `sparse_mm` | default |

That rule is measured, not assumed. `torch.sparse.mm`'s backward with respect to a
sparse operand materialises a dense `[N, N]` gradient. On the GB10, at N=50,000
with 500k edges and batch 32:

| | forward+backward | peak memory |
|---|---|---|
| `sparse_mm` | 140.4 ms | 11.74 GiB |
| `scatter` | **3.3 ms** | **0.24 GiB** |

A dense float32 `[50,000 x 50,000]` is 9.3 GiB, which is what that memory
figure is. Run `python benchmarks/sparse_backends.py` to reproduce the whole table;
`benchmarks/gb10-results.jsonl` holds the numbers above.

## The other memory invariant

Refusing a dense `[N, N]` is only half the job. Backpropagation through the
scatter backend holds two `[num_edges, batch]` tensors per step, so a training
call costs `2 * edges * batch * steps * itemsize`: about 6.5 GiB on MaleCNS at
batch 4 over 8 steps, and 104 GiB at batch 32 over 16. `model.activation_bytes(
batch, steps)` computes it up front, and the model warns before a call that would
claim more than half the device's free memory, rather than letting you find out
three hours into a run.

## Scientific controls

A claim that biological wiring helps has to be measured against wiring that is
matched in every way except being biological, or the result may just be showing
that sparsity helps.

```python
from connectorch.transforms import degree_preserving_rewire, random_topology

control = degree_preserving_rewire(fly, seed=0)   # same degrees, different wiring
```

Every control keeps the exact edge count and records in its provenance that it is
not biological data.

## Examples

```bash
python examples/01_toy_connectome.py     # build a graph by hand, get gradients
python examples/02_train_connectome.py   # train it; watch the topology not change
python examples/03_malecns_load.py       # load a real connectome
python examples/04_mnist_connectome.py   # MNIST through a fly circuit
```

The last one reaches 90.8% test accuracy in one epoch, in about 7 seconds on a
laptop CPU, with 3,630 real connections in the middle of the network.

## Scientific caveats

A connectome is a structural wiring diagram. Compiling it into a neural network does
not recreate the animal's brain.

- synapse count is not synaptic efficacy;
- neurotransmitter identity does not fix a sign, because the receptor decides;
- cellular dynamics, neuromodulation and gap junctions are absent;
- the reconstruction is incomplete and noisy;
- training the graph on a machine learning objective produces an artificial model
  constrained by biology, not a simulation of the original animal.

Every weight-initialisation strategy in this library is a *numerical* choice and is
documented as one. ConnecTorch never invents a biological interpretation.

## References

The data and the prior work, with full citations on the
[References](docs/references.md) page:

- **MaleCNS v1.0** — Berg, S. et al. *Sexual dimorphism in the complete Drosophila
  male central nervous system connectome.* Cell **189**, 5504–5526.e15 (2026),
  doi:10.1016/j.cell.2026.08.015. 166,691 annotated neurons. CC-BY.
  <https://male-cns.janelia.org/>
- **Transmitter predictions** — Eckstein, N. et al. Cell **187**, 2574–2594 (2024).
- **FlyVis**, the closest published proof the idea works — Lappalainen, J. K. et al.
  Nature **634**, 1132–1140 (2024).
- **Connectome Interpreter**, the nearest prior art — Yin, Y. et al. bioRxiv
  2025.09.29.679410.
- **neuPrint** — Plaza, S. M., Clements, J. et al. Front. Neuroinform. **16**,
  896292 (2022).

If you use a connectome through this library, cite its authors, not just this
library.

## License

MIT, for the library code. Bundled connectome-derived data keeps its own license and
attribution: see [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md). MaleCNS v1.0 is CC-BY;
cite its authors, not this library, when you use it.
