"""Train a connectome-constrained network, and prove the wiring never changed.

The task needs memory: a pulse arrives either in the first or the second half of
the sequence, and the network has to say which. Only the synapse weights are
learned. No optimizer step can add or remove a connection.

    python examples/02_train_connectome.py
"""

import numpy as np
import torch
from torch import nn

import connectorch as ct

SEED, STEPS = 0, 8
torch.manual_seed(SEED)
generator = torch.Generator().manual_seed(SEED)

brain = ct.datasets.random_sparse(num_nodes=60, num_edges=400, seed=SEED)
core = ct.nn.ConnectomeRNN(
    brain,
    input_nodes=np.arange(8),
    output_nodes=np.arange(52, 60),
    weights="trainable",
    leak=0.5,
)
model = ct.nn.ConnectomeModel(core, decoder=nn.Linear(core.num_output_nodes, 2))


def make_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, 2, (n,), generator=generator)
    x = torch.zeros(n, STEPS, core.num_input_nodes)
    for i, label in enumerate(labels):
        low, high = (0, STEPS // 2) if label == 0 else (STEPS // 2, STEPS)
        x[i, int(torch.randint(low, high, (1,), generator=generator)), :] = 1.0
    return x, labels


train_x, train_y = make_batch(256)
test_x, test_y = make_batch(128)

wiring_before = core.edge_index.clone()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

print(f"{brain}\n")
print(f"{'epoch':>6}  {'loss':>8}  {'val acc':>8}")
for epoch in range(61):
    if epoch % 15 == 0:
        with torch.no_grad():
            accuracy = (model(test_x).argmax(1) == test_y).float().mean()
            print(f"{epoch:>6}  {criterion(model(train_x), train_y):>8.4f}  {accuracy:>8.3f}")
    optimizer.zero_grad()
    criterion(model(train_x), train_y).backward()
    optimizer.step()

print(f"\nparameters: {ct.nn.count_parameters(model)}")
print(f"topology unchanged after training: {torch.equal(wiring_before, core.edge_index)}")
