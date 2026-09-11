# ConnecTorch

**Compile biological connectomes into trainable PyTorch networks.**

```python
import torch
import connectorch as ct

fly = ct.datasets.malecns_sample()
net = ct.nn.ConnectomeRNN(fly, weights="trainable")

y = net(torch.randn(8, fly.num_nodes), steps=10)
y.square().mean().backward()
```

A connectome is a wiring diagram: which neuron connects to which. ConnecTorch turns
one into a sparse recurrent `torch.nn.Module`, and keeps one promise while doing it:

!!! note "The invariant"
    Gradient descent can change what a connection is worth.
    It cannot create a connection the animal does not have.

```
arbitrary connectome -> Connectome IR -> compiler/runtime -> torch.nn.Module
```

Nothing in that pipeline is specific to any one species. The Janelia MaleCNS fly
connectome is the showcase, not the architecture.

## Install

```bash
pip install connectorch                 # core: torch, numpy, pyarrow
pip install 'connectorch[neuprint]'     # plus the neuPrint adapter
pip install 'connectorch[networkx]'     # plus NetworkX interop
```

## Related work

ConnecTorch is not the first project to put a connectome in front of PyTorch, and
it deliberately does not duplicate any of them.

- **[FlyVis](https://github.com/TuragaLab/flyvis)** (Lappalainen et al., Nature
  2024) is the strongest proof that connectome-constrained training works. It is
  a model of the *Drosophila* visual system: gather/scatter over an edge list,
  with weights shared per cell-type pair rather than per synapse, and a stimulus
  pipeline built for visual motion. ConnecTorch borrows the gather/scatter
  approach and is dataset-agnostic and task-agnostic instead.
- **[connectome-interpreter](https://github.com/YijieYin/connectome_interpreter)**
  is the nearest prior art: an internal `nn.Module` inside its
  activation-maximisation tool does take an arbitrary sparse weight matrix. It is
  a private helper with one fixed circuit-analysis forward pass, and the rest of
  the package is non-differentiable graph analysis. If you want to *understand* a
  connectome's paths and effective connectivity, use it; ConnecTorch is for
  *training* one.
- **[WormVAE](https://github.com/TuragaLab/wormvae)** constrains a latent variable
  model to *C. elegans* wiring for one specific calcium dataset.

No existing library exposes "arbitrary connectome to a trainable
`torch.nn.Module`" as a general-purpose public API. That is the gap this fills.

## Where to go next

- [Quickstart](quickstart.md), if you want code running in two minutes.
- [Concepts](concepts.md), for how topology and parameters are kept apart.
- [Scientific caveats](caveats.md), before you claim anything about biology.
