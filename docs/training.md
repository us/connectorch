# Training networks

## The update rule

For state `h` over `N` neurons and a batch of `B`:

```
m_t     = A @ h_t + u_t
h_{t+1} = (1 - leak) * h_t + leak * activation(m_t + bias)
```

`A[target, source] = edge_weight`, nonzero only where the connectome has an edge.
`u_t` is the external drive injected at the input nodes. `leak=1.0` replaces the
state each step; smaller values give each neuron a memory of its own past.

## Signatures

```python
y              = model(x, steps=10)                                  # doctest: +SKIP
y, state       = model(x, steps=10, return_state=True)               # doctest: +SKIP
y, state       = model(x, state=previous, steps=4, return_state=True)  # doctest: +SKIP
```

`x` is `[batch, num_input_nodes]` for a drive held constant across steps, or
`[batch, steps, num_input_nodes]` for a time-varying one. `y` is always the full
trajectory, `[batch, steps, num_output_nodes]`; take `y[:, -1]` yourself if that is
what you want.

The module is **stateless between calls**. Nothing is carried over implicitly, ever.
Continuing a run is explicit:

```python
import torch
import connectorch as ct

brain = ct.datasets.malecns_sample()
model = ct.nn.ConnectomeRNN(brain, weights="trainable", leak=0.5)
x1 = x2 = torch.randn(4, brain.num_nodes)

first, state = model(x1, steps=3, return_state=True)
second, _    = model(x2, state=state, steps=2, return_state=True)
```

## The invariant, tested

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

before = model.edge_index.clone()
for _ in range(50):
    optimizer.zero_grad()
    model(x1, steps=4).square().mean().backward()
    optimizer.step()
assert torch.equal(before, model.edge_index)    # always
```

With `weights="trainable"`, `edge_weight` is an `nn.Parameter` and `edge_index` is
a buffer. With any fixed strategy, `edge_weight` is a buffer too and
`model.parameters()` is empty: the connectome is frozen and only whatever you wrap
around it learns.

## Keeping it stable

Recurrent networks explode. Before training a large one, look:

```python
model.diagnostics()   # doctest: +SKIP
# {'nodes': 164587, 'edges': 25563197, 'backend': 'scatter',
#  'max_in_degree': 11526, 'max_abs_row_sum': 193.1, ...}
```

`max_abs_row_sum` is the quantity to watch: it bounds how much the state can grow
in one step. If it is large, use `normalized_synapse_count`, a smaller `leak`, or a
saturating activation like `tanh`. ConnecTorch never clips your state behind your
back.
