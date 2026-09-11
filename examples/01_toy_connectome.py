"""Build a connectome by hand and run it. Five lines of real work.

python examples/01_toy_connectome.py
"""

import torch

import connectorch as ct

brain = ct.Connectome.from_edges(
    source=["A", "B", "C", "C"],
    target=["B", "C", "A", "B"],
    weight=[1.0, 0.5, 0.2, 0.8],
)

model = ct.nn.ConnectomeRNN(brain, weights="trainable")

x = torch.randn(8, brain.num_nodes)
y = model(x, steps=5)

loss = y.square().mean()
loss.backward()

print(brain)
print(model)
print(f"\noutput shape: {tuple(y.shape)}  [batch, steps, output nodes]")
print(f"gradient on each of the {brain.num_edges} synapses:")
print(model.edge_weight.grad)
