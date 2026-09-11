"""Tests for ConnectomeRNN: semantics, autograd, topology invariance, device handling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connectorch import Connectome
from connectorch.nn import ConnectomeRNN

BACKENDS = ["dense", "sparse_mm", "scatter"]
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def chain() -> Connectome:
    """A --2--> B --3--> C."""
    return Connectome.from_edges(source=["A", "B"], target=["B", "C"], weight=[2.0, 3.0])


@pytest.fixture
def small() -> Connectome:
    rng = np.random.default_rng(0)
    return Connectome.from_edges(
        source=rng.integers(0, 12, 40),
        target=rng.integers(0, 12, 40),
        weight=rng.normal(size=40),
        nodes={"node_id": np.arange(12)},
    )


# ----------------------------------------------------------------------
# analytic recurrence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_analytic_chain_recurrence(chain: Connectome, backend: str) -> None:
    """With identity activation and leak=1, h=[1,0,0] gives B=2 then C=6."""
    model = ConnectomeRNN(chain, weights="weight", activation="identity", leak=1.0, backend=backend)
    y = model(torch.zeros(1, 3), steps=2, state=torch.tensor([[1.0, 0.0, 0.0]]))
    assert y[0, 0].tolist() == [0.0, 2.0, 0.0]
    assert y[0, 1].tolist() == [0.0, 0.0, 6.0]


def test_reciprocal_pair_oscillates(chain: Connectome) -> None:
    """A <-> B with weight 2 each way returns a 4x scaled state after two steps."""
    brain = Connectome.from_edges(source=["A", "B"], target=["B", "A"], weight=[2.0, 2.0])
    model = ConnectomeRNN(brain, weights="weight", activation="identity", leak=1.0)
    y = model(torch.zeros(1, 2), steps=2, state=torch.tensor([[1.0, 0.0]]))
    assert y[0, 0].tolist() == [0.0, 2.0]
    assert y[0, 1].tolist() == [4.0, 0.0]


def test_leak_mixes_the_previous_state(chain: Connectome) -> None:
    model = ConnectomeRNN(
        chain, weights="weight", activation="identity", leak=0.25, backend="dense"
    )
    state = torch.tensor([[1.0, 0.0, 0.0]])
    y = model(torch.zeros(1, 3), steps=1, state=state)
    # h_A = 0.75 * 1 + 0.25 * 0 ; h_B = 0.75 * 0 + 0.25 * 2
    assert y[0, 0].tolist() == pytest.approx([0.75, 0.5, 0.0])


def test_input_is_injected_at_the_named_nodes() -> None:
    brain = Connectome.from_edges(source=["A", "B"], target=["B", "C"], weight=[2.0, 3.0])
    model = ConnectomeRNN(
        brain,
        input_nodes=["A"],
        output_nodes=["B", "C"],
        weights="weight",
        activation="identity",
        leak=1.0,
    )
    y = model(torch.tensor([[1.0]]), steps=2)
    assert model.num_input_nodes == 1
    assert model.num_output_nodes == 2
    # step 1: A gets drive 1, B gets 0. step 2: B = 2*A(=1) = 2, plus A's new drive.
    assert y[0, 0].tolist() == [0.0, 0.0]
    assert y[0, 1].tolist() == [2.0, 0.0]


# ----------------------------------------------------------------------
# autograd and the headline invariant
# ----------------------------------------------------------------------


def test_trainable_weights_receive_finite_gradients(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="trainable", initializer="weight")
    model(torch.randn(4, small.num_nodes), steps=3).square().mean().backward()
    grad = model.edge_weight.grad
    assert grad is not None
    assert grad.shape == (small.num_edges,)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def test_fixed_weights_are_not_parameters(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="weight")
    assert list(model.parameters()) == []
    assert "edge_weight" in dict(model.named_buffers())
    assert not model.edge_weight.requires_grad


def test_gradcheck_on_a_tiny_double_precision_model() -> None:
    brain = Connectome.from_edges(
        source=[0, 1, 2, 2], target=[1, 2, 0, 1], weight=[0.3, -0.2, 0.1, 0.4]
    )
    model = ConnectomeRNN(
        brain, weights="trainable", initializer="weight", dtype=torch.float64, leak=0.5
    )
    x = torch.randn(2, 3, dtype=torch.float64)

    def run(weight: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, {"edge_weight": weight}, (x,), {"steps": 3})

    assert torch.autograd.gradcheck(run, (model.edge_weight.detach().requires_grad_(True),))


@pytest.mark.parametrize("backend", ["scatter", "dense"])
def test_training_never_changes_the_topology(small: Connectome, backend: str) -> None:
    """The headline property: optimizer steps may change weights, never connectivity."""
    model = ConnectomeRNN(small, weights="trainable", initializer="weight", backend=backend)
    original_edges = model.edge_index.clone()
    original_weights = model.edge_weight.detach().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    x = torch.randn(4, small.num_nodes)
    for _ in range(50):
        optimizer.zero_grad()
        model(x, steps=3).square().mean().backward()
        optimizer.step()

    assert torch.equal(original_edges, model.edge_index)
    assert model.edge_weight.numel() == original_weights.numel()
    assert not torch.allclose(model.edge_weight.detach(), original_weights)


def test_training_cannot_create_an_edge(small: Connectome) -> None:
    """Probe what the model actually propagates, not just what edge_index says.

    Reading edge_index alone would pass even if the runtime were secretly sending
    messages along connections that are not in the connectome, so the pattern here
    is measured by driving one neuron at a time and seeing where the signal lands.
    """
    model = ConnectomeRNN(
        small, weights="trainable", initializer="weight", activation="identity", leak=1.0
    )

    def pattern() -> torch.Tensor:
        """Nonzero at [target, source] wherever a message actually flows."""
        with torch.no_grad():
            impulses = torch.eye(small.num_nodes)
            responses = model(
                torch.zeros(small.num_nodes, small.num_nodes), steps=1, state=impulses
            )[:, 0, :]
        return (responses.t().abs() > 0).float()

    declared = torch.zeros(small.num_nodes, small.num_nodes)
    declared[model.edge_index[1], model.edge_index[0]] = 1.0
    assert torch.equal(pattern(), declared), "the runtime propagates along edges it declares"

    before = pattern()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    x = torch.randn(4, small.num_nodes)
    for _ in range(20):
        optimizer.zero_grad()
        model(x, steps=2).square().mean().backward()
        optimizer.step()
    assert torch.equal(before, pattern())


# ----------------------------------------------------------------------
# batching, sequences, state
# ----------------------------------------------------------------------


def test_batched_matches_one_at_a_time(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="weight")
    x = torch.randn(6, small.num_nodes)
    batched = model(x, steps=4)
    separate = torch.cat([model(x[i : i + 1], steps=4) for i in range(6)])
    assert torch.allclose(batched, separate, atol=1e-6)


def test_sequence_input_is_consumed_one_slice_per_step() -> None:
    brain = Connectome.from_edges(source=["A"], target=["B"], weight=[1.0])
    model = ConnectomeRNN(
        brain,
        input_nodes=["A"],
        output_nodes=["A"],
        weights="weight",
        activation="identity",
        leak=1.0,
    )
    x = torch.tensor([[[1.0], [2.0], [3.0]]])  # [batch=1, steps=3, inputs=1]
    y = model(x)
    assert y.shape == (1, 3, 1)
    assert y[0, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_steps_must_match_a_sequence_length() -> None:
    brain = Connectome.from_edges(source=[0], target=[1], weight=[1.0])
    model = ConnectomeRNN(brain, weights="weight")
    with pytest.raises(ValueError, match="does not match the sequence length"):
        model(torch.zeros(1, 3, 2), steps=5)


def test_steps_is_required_for_a_constant_drive() -> None:
    brain = Connectome.from_edges(source=[0], target=[1], weight=[1.0])
    model = ConnectomeRNN(brain, weights="weight")
    with pytest.raises(ValueError, match="steps is required"):
        model(torch.zeros(1, 2))


def test_state_continuation_matches_one_long_run(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="weight")
    x = torch.randn(3, small.num_nodes)
    first, state = model(x, steps=3, return_state=True)
    second, _ = model(x, steps=2, state=state, return_state=True)
    whole = model(x, steps=5)
    assert torch.allclose(torch.cat([first, second], dim=1), whole, atol=1e-6)


def test_module_keeps_no_hidden_state(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="weight")
    x = torch.randn(2, small.num_nodes)
    assert torch.equal(model(x, steps=3), model(x, steps=3))


def test_wrong_input_width_is_reported_clearly(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="weight")
    with pytest.raises(ValueError, match="input columns"):
        model(torch.zeros(2, small.num_nodes + 1), steps=1)


# ----------------------------------------------------------------------
# PyTorch semantics
# ----------------------------------------------------------------------


def test_state_dict_round_trip_preserves_topology_and_weights(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="trainable", initializer="weight")
    state = model.state_dict()
    assert "edge_index" in state, "topology must be in the state dict for reproducibility"

    clone = ConnectomeRNN(small, weights="trainable", initializer="binary")
    clone.load_state_dict(state)
    assert torch.equal(clone.edge_weight, model.edge_weight)
    assert torch.equal(clone.edge_index, model.edge_index)


def test_dtype_conversion_moves_every_tensor(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="trainable", initializer="weight").to(torch.float64)
    assert model.edge_weight.dtype is torch.float64
    assert model.edge_index.dtype is torch.int64, "indices must stay integral"
    y = model(torch.randn(2, small.num_nodes, dtype=torch.float64), steps=2)
    assert y.dtype is torch.float64


def test_diagnostics_reports_the_numbers_that_predict_blowup(small: Connectome) -> None:
    report = ConnectomeRNN(small, weights="weight").diagnostics()
    assert report["nodes"] == small.num_nodes
    assert report["edges"] == small.num_edges
    assert report["max_in_degree"] >= 1
    assert np.isfinite(report["max_abs_row_sum"])


def test_repr_says_what_the_model_is(small: Connectome) -> None:
    text = repr(ConnectomeRNN(small, weights="weight"))
    assert "nodes=12" in text
    assert "backend='sparse_mm'" in text


@CUDA
@pytest.mark.cuda
def test_cuda_round_trip(small: Connectome) -> None:
    model = ConnectomeRNN(small, weights="trainable", initializer="weight").cuda()
    for tensor in list(model.buffers()) + list(model.parameters()):
        assert tensor.is_cuda
    y = model(torch.randn(2, small.num_nodes, device="cuda"), steps=3)
    assert y.is_cuda
    y.square().mean().backward()
    assert model.edge_weight.grad is not None

    state = {k: v.cpu() for k, v in model.state_dict().items()}
    cpu_model = ConnectomeRNN(small, weights="trainable", initializer="binary")
    cpu_model.load_state_dict(state)
    assert not cpu_model.edge_weight.is_cuda


# ----------------------------------------------------------------------
# the other memory invariant: O(E * B * T) activations
# ----------------------------------------------------------------------


def test_activation_cost_is_reported_and_is_zero_without_gradients(small: Connectome) -> None:
    """Refusing a dense N x N means nothing if the recommended path OOMs unannounced."""
    model = ConnectomeRNN(small, weights="trainable")
    itemsize = model.edge_weight.element_size()
    assert model.activation_bytes(8, 4) == 2 * small.num_edges * 8 * 4 * itemsize
    with torch.no_grad():
        assert model.activation_bytes(8, 4) == 0

    frozen = ConnectomeRNN(small, weights="weight")
    assert frozen.activation_bytes(8, 4) == 0, "frozen weights store no activations"


def test_a_huge_backward_pass_is_flagged_before_it_is_paid(small: Connectome, monkeypatch) -> None:
    """The warning has to carry the real number and a way out, like the dense guard.

    The budget is lowered rather than the graph enlarged. Actually running a call
    that stores tens of gibibytes would have this test killed by the OOM killer on
    any ordinary machine, which is the very outcome the warning exists to prevent.
    """
    import warnings

    model = ConnectomeRNN(small, weights="trainable")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.zeros(2, small.num_nodes), steps=2)
    assert not caught, "a small call under the budget must not nag"

    monkeypatch.setattr("connectorch.nn.recurrent._ACTIVATION_BUDGET_BYTES", 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.zeros(4, small.num_nodes), steps=3)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "GiB" in text
    assert "batch 4" in text and "3 steps" in text
    assert f"{small.num_edges:,} edges" in text
    assert "torch.no_grad()" in text

    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        model(torch.zeros(4, small.num_nodes), steps=3)
    assert not caught, "nothing is stored under no_grad, so nothing to warn about"


def test_diagnostics_reports_the_activation_cost(small: Connectome) -> None:
    report = ConnectomeRNN(small, weights="trainable").diagnostics()
    assert report["activation_bytes_per_batch_step"] == 2 * small.num_edges * 4
