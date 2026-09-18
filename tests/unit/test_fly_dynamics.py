"""Fly-pure dynamics: threshold-linear output, per-type leak, synaptic delays."""

from __future__ import annotations

import io

import pytest
import torch

import connectorch as ct
from connectorch.nn import SparseExpander, k_wta
from connectorch.nn.dynamics import ACTIVATIONS


def tiny() -> ct.Connectome:
    return ct.Connectome.from_edges(
        source=["L1a", "L1b", "Mi1"],
        target=["Mi1", "Mi1", "T4"],
        weight=[1.0, 1.0, 1.0],
        nodes={
            "node_id": ["L1a", "L1b", "Mi1", "T4"],
            "cell_type": ["L1", "L1", "Mi1", "T4a"],
        },
    )


def test_threshold_linear_is_silent_below_threshold() -> None:
    f = ACTIVATIONS["threshold_linear"]
    x = torch.tensor([-2.0, 0.5, 1.0, 1.5, 3.0])
    assert torch.equal(f(x), torch.tensor([0.0, 0.0, 0.0, 0.5, 2.0]))


def test_leak_by_gives_fast_and_slow_neurons() -> None:
    brain = tiny()
    fast = ct.nn.ConnectomeRNN(
        brain,
        weights="binary",
        activation="identity",
        leak=0.5,
        leak_by={"L1": 1.0},
        backend="dense",
    )
    assert fast.leak_vec is not None
    assert fast.leak_vec.shape == (4, 1)
    # L1 rows take leak 1.0, everyone else the global 0.5.
    assert torch.equal(fast.leak_vec.squeeze(-1), torch.tensor([1.0, 1.0, 0.5, 0.5]))
    plain = ct.nn.ConnectomeRNN(brain, weights="binary", backend="dense")
    assert plain.leak_vec is None
    assert plain.max_delay == 0


def test_leak_by_rejects_unknown_column_and_bad_values() -> None:
    brain = tiny()
    with pytest.raises(Exception, match="no node column"):
        ct.nn.ConnectomeRNN(brain, weights="binary", leak_by={"L1": 0.5}, leak_column="region")
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        ct.nn.ConnectomeRNN(brain, weights="binary", leak_by={"L1": 0.0})
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        ct.nn.ConnectomeRNN(brain, weights="binary", leak_by={"L1": 1.5})


def test_delay_zero_matches_no_delay_bitwise() -> None:
    brain = tiny()
    torch.manual_seed(0)
    a = ct.nn.ConnectomeRNN(
        brain,
        weights="trainable",
        initializer="binary",
        activation="tanh",
        leak=0.4,
        backend="dense",
    )
    torch.manual_seed(0)
    b = ct.nn.ConnectomeRNN(
        brain,
        weights="trainable",
        initializer="binary",
        activation="tanh",
        leak=0.4,
        delay_by={"Mi1": 0},
        backend="dense",
    )
    x = torch.randn(2, 5, 4)
    assert torch.equal(a(x), b(x))


def test_delay_holds_back_the_slow_source() -> None:
    # Mi1 -> T4 delayed by one step: T4 at step 0 sees nothing from Mi1,
    # at step 1 it sees what Mi1 was at step 0.
    brain = tiny()
    model = ct.nn.ConnectomeRNN(
        brain,
        weights="binary",
        activation="identity",
        leak=1.0,
        delay_by={"Mi1": 1},
        backend="dense",
    )
    assert model.max_delay == 1
    assert model.delay_values == [0, 1]
    x = torch.zeros(1, 4, 4)
    x[0, 0, 2] = 1.0  # drive Mi1 (input order = all nodes) at step 0 only
    y = model(x)
    t4 = y[0, :, 3]
    # Drive becomes Mi1 state after one step; the delay holds it one more:
    # T4 sees nothing until step 2.
    assert t4[0].item() == pytest.approx(0.0)
    assert t4[1].item() == pytest.approx(0.0)
    assert t4[2].item() == pytest.approx(1.0)


def test_delay_rejects_bad_values_and_columns() -> None:
    brain = tiny()
    with pytest.raises(Exception, match="no node column"):
        ct.nn.ConnectomeRNN(brain, weights="binary", delay_by={"Mi1": 1}, delay_column="region")
    with pytest.raises(ValueError, match="integer in \\[0, 4\\]"):
        ct.nn.ConnectomeRNN(brain, weights="binary", delay_by={"Mi1": 5})
    with pytest.raises(ValueError, match="integer in \\[0, 4\\]"):
        ct.nn.ConnectomeRNN(brain, weights="binary", delay_by={"Mi1": -1})


def test_delayed_model_round_trips_through_state_dict() -> None:
    brain = tiny()
    model = ct.nn.ConnectomeRNN(
        brain,
        weights="binary",
        leak_by={"L1": 1.0},
        delay_by={"Mi1": 1},
        backend="dense",
    )
    x = torch.randn(2, 4, 4)
    before = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = ct.nn.ConnectomeRNN(
        brain,
        weights="binary",
        leak_by={"L1": 1.0},
        delay_by={"Mi1": 1},
        backend="dense",
    )
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(before, clone(x))
    assert clone.max_delay == 1
    assert torch.equal(clone.leak_vec, model.leak_vec)


def test_k_wta_keeps_exactly_k_per_row() -> None:
    x = torch.tensor([[3.0, 1.0, 2.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    y = k_wta(x, 2)
    assert int((y[0] != 0).sum()) == 2
    assert y[0, 0].item() == 3.0 and y[0, 2].item() == 2.0
    with pytest.raises(ValueError, match="at least 1"):
        k_wta(x, 0)


def test_sparse_expander_is_fixed_expansion() -> None:
    mod = SparseExpander(50, 2000, fan_in=6, k=100, seed=0)
    assert sum(p.numel() for p in mod.parameters() if p.requires_grad) == 0
    assert int(mod.projection.sum()) == 2000 * 6
    x = torch.randn(4, 50)
    y = mod(x)
    assert y.shape == (4, 2000)
    assert bool(((y != 0).sum(dim=1) == 100).all())
    again = SparseExpander(50, 2000, fan_in=6, k=100, seed=0)
    assert torch.equal(mod.projection, again.projection)
    with pytest.raises(ValueError, match="fan_in"):
        SparseExpander(50, 10, fan_in=51)


def test_delayed_gradients_flow_to_weights_and_inputs() -> None:
    brain = tiny()
    model = ct.nn.ConnectomeRNN(
        brain,
        weights="trainable",
        initializer="binary",
        activation="tanh",
        leak=0.5,
        delay_by={"Mi1": 1},
        backend="scatter",
    )
    x = torch.randn(2, 3, 4, requires_grad=True)
    model(x).square().mean().backward()
    assert model.edge_weight.grad is not None
    assert torch.isfinite(model.edge_weight.grad).all()
    assert x.grad is not None
