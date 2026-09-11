"""Tests for the encoder/core/decoder wrappers and the activation registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from connectorch import Connectome
from connectorch.nn import (
    ACTIVATIONS,
    ConnectomeModel,
    ConnectomeRNN,
    InputProjection,
    count_parameters,
    resolve_activation,
)


@pytest.fixture
def core() -> ConnectomeRNN:
    rng = np.random.default_rng(0)
    brain = Connectome.from_edges(
        source=rng.integers(0, 16, 60),
        target=rng.integers(0, 16, 60),
        synapse_count=rng.integers(1, 20, 60),
        nodes={"node_id": np.arange(16)},
    )
    return ConnectomeRNN(
        brain, input_nodes=np.arange(4), output_nodes=np.arange(12, 16), weights="trainable"
    )


def test_input_projection_maps_features_onto_input_neurons(core: ConnectomeRNN) -> None:
    encoder = InputProjection(784, core)
    assert encoder(torch.randn(3, 784)).shape == (3, core.num_input_nodes)


@pytest.mark.parametrize("readout,expected", [("last", (3, 2)), ("mean", (3, 2))])
def test_readouts_produce_task_outputs(core: ConnectomeRNN, readout: str, expected) -> None:
    model = ConnectomeModel(
        core,
        encoder=nn.Linear(10, core.num_input_nodes),
        decoder=nn.Linear(core.num_output_nodes, 2),
        readout=readout,
    )
    assert model(torch.randn(3, 10), steps=4).shape == expected


def test_readout_all_keeps_the_trajectory(core: ConnectomeRNN) -> None:
    model = ConnectomeModel(core, decoder=nn.Linear(core.num_output_nodes, 2), readout="all")
    assert model(torch.randn(3, core.num_input_nodes), steps=4).shape == (3, 4, 2)


def test_readout_last_is_the_final_step(core: ConnectomeRNN) -> None:
    """'last' must be the end of the trajectory, not the start or an average."""
    model = ConnectomeModel(core, readout="last")
    x = torch.randn(3, core.num_input_nodes)
    assert torch.equal(model(x, steps=5), core(x, steps=5)[:, -1, :])


def test_readout_mean_is_the_mean_over_steps(core: ConnectomeRNN) -> None:
    model = ConnectomeModel(core, readout="mean")
    x = torch.randn(3, core.num_input_nodes)
    assert torch.allclose(model(x, steps=5), core(x, steps=5).mean(dim=1))


def test_an_unknown_readout_names_the_valid_ones(core: ConnectomeRNN) -> None:
    with pytest.raises(ValueError, match="'last', 'mean' or 'all'"):
        ConnectomeModel(core, readout="magic")


def test_a_bare_model_passes_straight_through(core: ConnectomeRNN) -> None:
    model = ConnectomeModel(core)
    assert model(torch.randn(2, core.num_input_nodes), steps=3).shape == (
        2,
        core.num_output_nodes,
    )


def test_gradients_reach_the_connectome_through_the_wrappers(core: ConnectomeRNN) -> None:
    model = ConnectomeModel(
        core,
        encoder=nn.Linear(10, core.num_input_nodes),
        decoder=nn.Linear(core.num_output_nodes, 2),
    )
    model(torch.randn(3, 10), steps=3).sum().backward()
    assert core.edge_weight.grad is not None
    assert torch.isfinite(core.edge_weight.grad).all()


def test_count_parameters_splits_the_budget(core: ConnectomeRNN) -> None:
    decoder = nn.Linear(core.num_output_nodes, 2)
    counts = count_parameters(ConnectomeModel(core, decoder=decoder))
    assert counts["connectome"] == core.num_edges
    assert counts["other"] == sum(p.numel() for p in decoder.parameters())
    assert counts["total"] == counts["connectome"] + counts["other"]


def test_count_parameters_ignores_frozen_weights() -> None:
    brain = Connectome.from_edges([0, 1], [1, 2], synapse_count=[3, 4])
    frozen = ConnectomeRNN(brain, weights="synapse_count")
    assert count_parameters(ConnectomeModel(frozen))["connectome"] == 0


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_every_activation_runs_and_differentiates(name: str) -> None:
    brain = Connectome.from_edges([0, 1], [1, 2], weight=[0.5, 0.5])
    model = ConnectomeRNN(brain, weights="trainable", activation=name)
    model(torch.randn(2, 3), steps=2).sum().backward()
    assert torch.isfinite(model.edge_weight.grad).all()


def test_a_callable_activation_is_accepted() -> None:
    brain = Connectome.from_edges([0, 1], [1, 2], weight=[0.5, 0.5])
    model = ConnectomeRNN(brain, weights="trainable", activation=torch.nn.functional.elu)
    assert model.activation_name == "custom"
    assert model(torch.randn(2, 3), steps=2).shape == (2, 2, 3)


def test_an_unknown_activation_lists_the_real_ones() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        resolve_activation("wobble")


def test_leak_outside_its_range_is_refused() -> None:
    brain = Connectome.from_edges([0], [1], weight=[1.0])
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="leak must be in"):
            ConnectomeRNN(brain, weights="weight", leak=bad)


def test_an_edgeless_connectome_cannot_become_a_model() -> None:
    from connectorch import ConnectorchError

    brain = Connectome.from_edges([0], [1]).filter_edges(min_weight=None, drop_self_loops=True)
    empty = brain.subgraph([0])
    with pytest.raises(ConnectorchError, match="no edges"):
        ConnectomeRNN(empty, weights="binary")


def test_bias_is_a_parameter_only_when_asked_for() -> None:
    brain = Connectome.from_edges([0, 1], [1, 2], weight=[1.0, 1.0])
    assert ConnectomeRNN(brain, weights="weight", bias=False).bias is None
    biased = ConnectomeRNN(brain, weights="weight", bias=True)
    assert biased.bias.shape == (3,)
    biased(torch.randn(2, 3), steps=2).sum().backward()
    assert biased.bias.grad is not None
