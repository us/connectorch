"""Tests for what training is allowed to do to a connection's weight.

The default parameterisation keeps the topology and nothing else: after enough
updates a pair joined by two synapses can outweigh a pair joined by two hundred,
and a neuron can excite half its targets while inhibiting the other half, which no
neuron does. These tests pin down the alternative that does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from connectorch import Connectome, ConnectorchError
from connectorch.nn import BiologicalWeights, ConnectomeModel, ConnectomeRNN
from connectorch.transforms import DROSOPHILA_POLARITY, infer_signs


@pytest.fixture
def brain() -> Connectome:
    """A small graph with cell types and transmitters, as a real loader would give."""
    rng = np.random.default_rng(0)
    num_nodes, num_edges = 40, 300
    types = np.array([f"T{i % 5}" for i in range(num_nodes)])
    transmitters = rng.choice(
        ["acetylcholine", "gaba", "glutamate"], size=num_nodes, p=[0.6, 0.2, 0.2]
    )
    graph = Connectome.from_edges(
        source=rng.integers(0, num_nodes, num_edges),
        target=rng.integers(0, num_nodes, num_edges),
        synapse_count=rng.integers(1, 200, num_edges),
        nodes={
            "node_id": np.arange(num_nodes),
            "cell_type": types,
            "neurotransmitter": transmitters,
        },
    )
    return infer_signs(graph, DROSOPHILA_POLARITY)


def train(core: ConnectomeRNN, steps: int = 120, lr: float = 0.02) -> None:
    model = ConnectomeModel(
        core,
        encoder=nn.Linear(8, core.num_input_nodes),
        decoder=nn.Linear(core.num_output_nodes, 4),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = torch.randn(32, 8), torch.randint(0, 4, (32,))
    for _ in range(steps):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x, steps=3), y).backward()
        optimizer.step()


def mixed_sign_neurons(core: ConnectomeRNN) -> int:
    """Neurons that both excite and inhibit. A real neuron never does."""
    weights = core.edge_weight.detach().numpy()
    source = core.edge_index[0].numpy()
    count = 0
    for node in np.unique(source):
        outgoing = weights[source == node]
        if outgoing.size > 1 and (outgoing > 0).any() and (outgoing < 0).any():
            count += 1
    return count


# ----------------------------------------------------------------------
# Dale's principle
# ----------------------------------------------------------------------


def test_free_weights_break_dales_principle(brain: Connectome) -> None:
    """The baseline this exists to replace: recorded, so the difference is visible."""
    torch.manual_seed(0)
    core = ConnectomeRNN(brain, weights="trainable", leak=0.4)
    assert mixed_sign_neurons(core) == 0
    train(core)
    assert mixed_sign_neurons(core) > 0, "free training would have to be constrained"


def test_dale_holds_through_training(brain: Connectome) -> None:
    torch.manual_seed(0)
    core = ConnectomeRNN(brain, weights=BiologicalWeights(brain, dale=True), leak=0.4)
    train(core)
    assert mixed_sign_neurons(core) == 0


def test_dale_uses_the_sign_column_not_the_initial_values(brain: Connectome) -> None:
    core = ConnectomeRNN(brain, weights=BiologicalWeights(brain, dale=True))
    expected = np.where(brain.edge_attribute("sign") == 0, 1, brain.edge_attribute("sign"))
    assert np.array_equal(np.sign(core.edge_weight.detach().numpy()), expected)


def test_dale_without_signs_says_how_to_get_them() -> None:
    bare = Connectome.from_edges([0, 1], [1, 2], synapse_count=[3, 4])
    with pytest.raises(ConnectorchError, match="infer_signs"):
        BiologicalWeights(bare, dale=True)


# ----------------------------------------------------------------------
# the biological prior survives training
# ----------------------------------------------------------------------


def test_the_gain_stays_inside_its_bounds(brain: Connectome) -> None:
    """Learning may modulate what was measured; it may not overrule it."""
    torch.manual_seed(0)
    weights = BiologicalWeights(brain, gain_bounds=(0.5, 2.0), dale=True)
    core = ConnectomeRNN(brain, weights=weights, leak=0.4)
    prior = core.edge_weight.detach().abs().clone()

    train(core, steps=200, lr=0.2)

    ratio = (core.edge_weight.detach().abs() / prior).numpy()
    assert ratio.min() >= 0.5 - 1e-6
    assert ratio.max() <= 2.0 + 1e-6


def test_it_starts_as_the_measured_connectome(brain: Connectome) -> None:
    """Gain 1.0 at initialisation, so the model begins where the biology is."""
    weights = BiologicalWeights(brain, dale=True)
    assert torch.allclose(weights.gain(), torch.ones(brain.num_edges), atol=1e-6)


def test_free_weights_do_not_preserve_the_prior(brain: Connectome) -> None:
    """Contrast: the same training, unconstrained, walks far away from the counts."""
    torch.manual_seed(0)
    core = ConnectomeRNN(brain, weights="trainable", leak=0.4)
    prior = core.edge_weight.detach().abs().clone()
    train(core, steps=200, lr=0.2)
    ratio = (core.edge_weight.detach().abs() / prior).numpy()
    assert ratio.max() > 2.0 or ratio.min() < 0.5


def test_bounds_must_contain_one(brain: Connectome) -> None:
    with pytest.raises(ConnectorchError, match="exclude 1.0"):
        BiologicalWeights(brain, gain_bounds=(1.5, 3.0))
    with pytest.raises(ConnectorchError, match="0 < low < high"):
        BiologicalWeights(brain, gain_bounds=(2.0, 0.5))


def test_an_unbounded_gain_is_possible_but_opt_in(brain: Connectome) -> None:
    weights = BiologicalWeights(brain, gain_bounds=None, dale=True)
    assert weights.describe()["gain_bounds"] is None
    assert torch.allclose(weights.gain(), torch.ones(brain.num_edges), atol=1e-6)


# ----------------------------------------------------------------------
# cell-type parameter sharing
# ----------------------------------------------------------------------


def test_sharing_by_cell_type_reduces_the_parameter_count(brain: Connectome) -> None:
    per_edge = BiologicalWeights(brain)
    shared = BiologicalWeights(brain, share_by="cell_type")
    assert per_edge.num_groups == brain.num_edges
    assert shared.num_groups < brain.num_edges
    assert shared.num_groups <= 5 * 5, "five cell types can make at most 25 pairs"


def test_connections_between_the_same_type_pair_move_together(brain: Connectome) -> None:
    """That is what sharing means, and it is checkable: one gradient step, one value."""
    weights = BiologicalWeights(brain, share_by="cell_type", dale=True)
    core = ConnectomeRNN(brain, weights=weights, leak=0.4)
    train(core, steps=20)

    gains = weights.gain().detach().numpy()
    groups = weights.group_index.numpy()
    for group in np.unique(groups):
        within = gains[groups == group]
        assert np.allclose(within, within[0]), "a shared group must hold one gain"


def test_sharing_needs_a_column_that_exists(brain: Connectome) -> None:
    with pytest.raises(ConnectorchError, match="no node column"):
        BiologicalWeights(brain, share_by="nonexistent")


def test_unlabelled_neurons_are_not_lumped_into_one_group() -> None:
    """An empty cell type is missing data, not a cell type shared by everyone."""
    graph = Connectome.from_edges(
        source=[0, 1, 2],
        target=[1, 2, 3],
        synapse_count=[5, 6, 7],
        nodes={"node_id": [0, 1, 2, 3], "cell_type": ["", "", "", ""]},
    )
    weights = BiologicalWeights(graph, share_by="cell_type")
    assert weights.num_groups == graph.num_edges


# ----------------------------------------------------------------------
# integration with the runtime
# ----------------------------------------------------------------------


def test_gradients_reach_the_gain(brain: Connectome) -> None:
    weights = BiologicalWeights(brain, share_by="cell_type", dale=True)
    core = ConnectomeRNN(brain, weights=weights, leak=0.4)
    core(torch.randn(4, brain.num_nodes), steps=3).square().mean().backward()
    assert weights.raw_gain.grad is not None
    assert torch.isfinite(weights.raw_gain.grad).all()
    assert torch.count_nonzero(weights.raw_gain.grad) > 0


def test_the_topology_still_cannot_change(brain: Connectome) -> None:
    weights = BiologicalWeights(brain, share_by="cell_type", dale=True)
    core = ConnectomeRNN(brain, weights=weights, leak=0.4)
    before = core.edge_index.clone()
    train(core)
    assert torch.equal(before, core.edge_index)


def test_a_weight_module_from_another_connectome_is_refused(brain: Connectome) -> None:
    other = Connectome.from_edges([0, 1], [1, 2], synapse_count=[1, 2])
    with pytest.raises(ConnectorchError, match="same connectome"):
        ConnectomeRNN(brain, weights=BiologicalWeights(other))


def test_the_model_reports_what_it_is_doing(brain: Connectome) -> None:
    core = ConnectomeRNN(brain, weights=BiologicalWeights(brain, share_by="cell_type", dale=True))
    described = core.weights.describe()
    assert described["kind"] == "biological"
    assert described["dale"] is True
    assert described["share_by"] == "cell_type"
    assert "biological" in repr(core)
