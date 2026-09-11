"""Proof that a connectome-constrained network actually learns, not just differentiates.

The task is deliberately trivial for a recurrent model and impossible for a
feedforward one: a pulse is delivered to the input neurons either in the first half
or the second half of the sequence, and the network must say which. Solving it
requires the state to persist across steps, which means the recurrence has to work.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from connectorch import Connectome
from connectorch.nn import ConnectomeRNN

NUM_NODES = 60
NUM_EDGES = 400
STEPS = 8
SEED = 0


def random_connectome(seed: int = SEED) -> Connectome:
    rng = np.random.default_rng(seed)
    return Connectome.from_edges(
        source=rng.integers(0, NUM_NODES, NUM_EDGES),
        target=rng.integers(0, NUM_NODES, NUM_EDGES),
        synapse_count=rng.integers(1, 40, NUM_EDGES),
        nodes={"node_id": np.arange(NUM_NODES)},
    )


def make_task(num_samples: int, num_inputs: int, generator: torch.Generator):
    """Return ``x`` of ``[n, STEPS, num_inputs]`` and labels of ``[n]``."""
    labels = torch.randint(0, 2, (num_samples,), generator=generator)
    x = torch.zeros(num_samples, STEPS, num_inputs)
    for i, label in enumerate(labels):
        low, high = (0, STEPS // 2) if label == 0 else (STEPS // 2, STEPS)
        step = int(torch.randint(low, high, (1,), generator=generator))
        x[i, step, :] = 1.0
    return x, labels


class Classifier(nn.Module):
    """Connectome core plus a linear readout of the final step."""

    def __init__(self, core: ConnectomeRNN) -> None:
        super().__init__()
        self.core = core
        self.readout = nn.Linear(core.num_output_nodes, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.core(x)[:, -1, :])


@pytest.mark.parametrize("backend", ["scatter", "dense"])
def test_a_connectome_constrained_network_learns(backend: str) -> None:
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    brain = random_connectome()
    core = ConnectomeRNN(
        brain,
        input_nodes=np.arange(8),
        output_nodes=np.arange(NUM_NODES - 8, NUM_NODES),
        weights="trainable",
        initializer="normalized_synapse_count",
        activation="tanh",
        leak=0.5,
        backend=backend,
    )
    model = Classifier(core)

    train_x, train_y = make_task(256, core.num_input_nodes, generator)
    test_x, test_y = make_task(128, core.num_input_nodes, generator)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(train_x), train_y).item()

    for _ in range(60):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(train_x), train_y).item()
        accuracy = (model(test_x).argmax(1) == test_y).float().mean().item()

    assert final_loss < 0.3 * initial_loss, f"{initial_loss=} {final_loss=}"
    assert accuracy > 0.9, f"held-out accuracy {accuracy:.3f}"


def test_the_topology_is_still_the_connectome_after_training() -> None:
    """Training the task above must leave the wiring diagram untouched."""
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    brain = random_connectome()
    core = ConnectomeRNN(brain, weights="trainable", leak=0.5)
    model = Classifier(core)
    before = core.edge_index.clone()

    x, y = make_task(64, core.num_input_nodes, generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(20):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()

    assert torch.equal(before, core.edge_index)
    assert core.edge_index.shape[1] == brain.num_edges


def test_fixed_weights_do_not_train_but_a_readout_still_can() -> None:
    """weights='synapse_count' leaves the connectome frozen; only the readout learns."""
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    brain = random_connectome()
    core = ConnectomeRNN(brain, weights="synapse_count", leak=0.5)
    model = Classifier(core)
    frozen = core.edge_weight.clone()

    x, y = make_task(64, core.num_input_nodes, generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(10):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()

    assert torch.equal(frozen, core.edge_weight)
    assert all(not name.startswith("core.") for name, _ in model.named_parameters())
