# Quickstart

## A graph you typed yourself

```python
import torch
import connectorch as ct

brain = ct.Connectome.from_edges(
    source=["A", "B", "C", "C"],
    target=["B", "C", "A", "B"],
    weight=[1.0, 0.5, 0.2, 0.8],
)
print(brain)
```

```
Connectome(
    nodes=3,
    edges=4,
    directed=True
)
```

## Make it a network

```python
model = ct.nn.ConnectomeRNN(brain, weights="trainable")

x = torch.randn(8, brain.num_nodes)     # [batch, input nodes]
y = model(x, steps=5)                   # [batch, steps, output nodes]

y.square().mean().backward()
print(model.edge_weight.grad)           # one gradient per connection
```

`model` is an ordinary `torch.nn.Module`. `.parameters()`, `.state_dict()`,
`.cuda()`, `.to(torch.float64)`, `.train()`, `.eval()` all work as you expect.

## A real connectome

The bundled sample is 100 real neurons from the fly central nervous system and
needs no network access:

```python
fly = ct.datasets.malecns_sample()
print(fly)                                     # 100 nodes, 3,630 edges
print(fly.provenance["license"])               # CC-BY-4.0
```

The full dataset is half a gigabyte and is never downloaded without being asked:

```python
fly = ct.datasets.malecns(download=True)       # 164,587 neurons
optic = fly.where(cell_class="visual")
```

## Plug it into a task

```python
from torch import nn

core = ct.nn.ConnectomeRNN(fly, input_nodes=fly.node_ids[:64], weights="trainable")
model = ct.nn.ConnectomeModel(
    core,
    encoder=nn.Sequential(nn.Flatten(), nn.Linear(784, core.num_input_nodes), nn.Tanh()),
    decoder=nn.Linear(core.num_output_nodes, 10),
)

logits = model(images, steps=6)
```

`examples/04_mnist_connectome.py` is exactly this, and reaches 90.8% on MNIST in
one epoch.
