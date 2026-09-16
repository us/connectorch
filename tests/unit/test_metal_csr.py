"""Independent numerical, lifecycle, and native Metal tests for CSR propagation.

Every Apple GPU test is marked ``mps``. Run CPU numerical checks with
``pytest tests/unit/test_metal_csr.py -m 'not mps'``. The optimizer test is also
MPS-only so this selection does not perform parameter updates.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from connectorch import Connectome
from connectorch.backends import build_propagator
from connectorch.backends._metal_csr import TrainableCSR
from connectorch.backends.metal_csr import MetalCSRPropagator
from connectorch.exceptions import BackendError
from connectorch.nn import BiologicalWeights, ConnectomeRNN

MPS_AVAILABLE = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)
MPS = pytest.param(
    "mps",
    marks=[
        pytest.mark.mps,
        pytest.mark.skipif(not MPS_AVAILABLE, reason="needs native Apple Metal"),
    ],
)
DEVICES = ["cpu", MPS]


def csr_fixture():
    # Empty row 1, empty column 3, unsorted columns, a stored zero, and both signs.
    ptr = np.array([0, 3, 3, 4, 6], dtype=np.int64)
    col = np.array([4, 0, 2, 1, 4, 0], dtype=np.int64)
    row = np.array([0, 0, 0, 2, 3, 3], dtype=np.int64)
    values = torch.tensor([0.25, -1.5, 0.0, 0.75, -0.4, 0.6])
    return ptr, col, row, values, (4, 6)


def dense_reference(h, values, row, col, shape):
    """No sparse operations or runtime-derived indices in this oracle."""
    matrix = torch.zeros(shape, dtype=h.dtype, device=h.device)
    matrix = matrix.index_put(
        (torch.as_tensor(row, device=h.device), torch.as_tensor(col, device=h.device)), values
    )
    return h @ matrix.T


def strided_leaf(value, device):
    storage = torch.empty(
        (*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device=device
    )
    storage[..., ::2] = value.to(device)
    return storage[..., ::2].detach().requires_grad_()


def forbid_sparse_mm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native Metal execution must not call torch.sparse.mm")

    monkeypatch.setattr(torch.sparse, "mm", forbidden)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch", [1, 8, 33])
def test_forward_state_and_edge_gradients(device, batch, monkeypatch):
    ptr, col, row, values, shape = csr_fixture()
    generator = torch.Generator().manual_seed(2900 + batch)
    h = torch.randn(batch, shape[1], generator=generator).requires_grad_()
    v = values.clone().requires_grad_()
    upstream = torch.randn(batch, shape[0], generator=generator)
    expected = dense_reference(h, v, row, col, shape)
    expected_grads = torch.autograd.grad(expected, (h, v), upstream)

    actual_h, actual_v = strided_leaf(h, device), strided_leaf(v, device)
    assert not actual_h.is_contiguous() and not actual_v.is_contiguous()
    graph = TrainableCSR(ptr, col, shape, device=device)
    if device == "mps":
        forbid_sparse_mm(monkeypatch)
    actual = graph.mm(actual_h, actual_v)
    # A noncontiguous upstream tests the backward shaders' input layout too.
    actual_upstream = strided_leaf(upstream, device).detach()
    grads = torch.autograd.grad(actual, (actual_h, actual_v), actual_upstream)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    for got, wanted in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(got.cpu(), wanted, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(actual[:, 1]) == 0
    assert torch.count_nonzero(grads[0][:, 3]) == 0
    assert grads[1][2].item() != 0, "A stored zero remains a differentiable edge"
    assert graph.verify_frozen()
    if device == "mps":
        assert graph.metadata()["kernel_launch_count"] == {
            "forward": 1,
            "state_backward": 1,
            "edge_backward": 1,
        }


@pytest.mark.parametrize("device", DEVICES)
def test_long_unsorted_row_spans_multiple_simd_reductions(device, monkeypatch):
    n = 521
    col = np.random.default_rng(17).permutation(n)
    ptr, row, shape = np.array([0, n, n, n]), np.zeros(n, dtype=np.int64), (3, n)
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(8, n, generator=generator, dtype=torch.float64).requires_grad_()
    values = torch.linspace(-0.9, 0.8, n, dtype=torch.float64).requires_grad_()
    upstream = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    expected = dense_reference(h, values, row, col, shape)
    expected_grads = torch.autograd.grad(expected, (h, values), upstream)
    graph = TrainableCSR(ptr, col, shape, device=device)
    x = h.detach().float().to(device).requires_grad_()
    w = values.detach().float().to(device).requires_grad_()
    if device == "mps":
        forbid_sparse_mm(monkeypatch)
    got = graph.mm(x, w)
    grads = torch.autograd.grad(got, (x, w), upstream.float().to(device))
    torch.testing.assert_close(got.cpu().double(), expected, rtol=3e-5, atol=2e-5)
    for actual, reference in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(actual.cpu().double(), reference, rtol=3e-5, atol=2e-6)


def test_cpu_double_gradcheck():
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape)
    h = torch.randn(2, shape[1], dtype=torch.float64, requires_grad=True)
    v = values.double().requires_grad_()
    assert torch.autograd.gradcheck(graph.mm, (h, v), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty_batch,empty_edges", [(True, False), (False, True), (True, True)])
def test_empty_batch_and_edges_have_zero_gradients(device, empty_batch, empty_edges):
    ptr, col, _, values, shape = csr_fixture()
    if empty_edges:
        ptr, col, values = np.zeros(5, dtype=np.int64), np.array([], dtype=np.int64), values[:0]
    graph = TrainableCSR(ptr, col, shape, device=device)
    h = torch.ones((0 if empty_batch else 3, shape[1]), device=device, requires_grad=True)
    v = values.to(device).detach().requires_grad_()
    y = graph.mm(h, v)
    dh, dv = torch.autograd.grad(y.sum(), (h, v))
    assert y.shape == (h.shape[0], shape[0])
    assert torch.count_nonzero(y) == torch.count_nonzero(dh) == torch.count_nonzero(dv) == 0
    assert graph.verify_frozen()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("which", ["state", "edges", "both"])
def test_selective_gradients_only_launch_needed_kernels(device, which):
    ptr, col, row, values, shape = csr_fixture()
    h = torch.linspace(-0.4, 0.8, 18).reshape(3, 6)
    x = h.to(device).detach().requires_grad_(which != "edges")
    v = values.to(device).detach().requires_grad_(which != "state")
    graph = TrainableCSR(ptr, col, shape, device=device)
    graph.mm(x, v).square().sum().backward()
    assert (x.grad is not None) == (which != "edges")
    assert (v.grad is not None) == (which != "state")
    hx, vx = h.clone().requires_grad_(), values.clone().requires_grad_()
    dense_reference(hx, vx, row, col, shape).square().sum().backward()
    if x.grad is not None:
        torch.testing.assert_close(x.grad.cpu(), hx.grad, rtol=2e-5, atol=2e-6)
    if v.grad is not None:
        torch.testing.assert_close(v.grad.cpu(), vx.grad, rtol=2e-5, atol=2e-6)
    if device == "mps":
        assert graph.metadata()["kernel_launch_count"] == {
            "forward": 1,
            "state_backward": int(which != "edges"),
            "edge_backward": int(which != "state"),
        }


def test_higher_order_gradients_are_explicitly_rejected():
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape)
    h = torch.ones(2, shape[1], requires_grad=True)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(graph.mm(h, values).sum(), h, create_graph=True)


@pytest.mark.parametrize("device", DEVICES)
def test_backward_saves_state_and_values_without_edge_batch_tensor(device):
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape, device=device)
    h = torch.ones(7, 6, device=device, requires_grad=True)
    v = values.to(device).requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = graph.mm(h, v)
    assert saved == [(7, 6), (6,)]
    assert graph.metadata()["edge_batch_intermediate"] is False
    output.sum().backward()


def test_topology_is_copied_and_versioned():
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape)
    h = torch.ones(2, shape[1])
    expected = graph.mm(h, values)
    ptr[:] = 0
    col[:] = 0
    torch.testing.assert_close(graph.mm(h, values), expected, rtol=0, atol=0)
    graph._buffers["transpose_edge"][0] = 1
    with pytest.raises(RuntimeError, match="cache was mutated"):
        graph.mm(h, values)


def test_explicit_audit_detects_data_writes_without_version_increment():
    ptr, col, _, _, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape)
    graph._buffers["col"].data[0] = 5
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        graph.verify_frozen()


@pytest.mark.parametrize(
    "ptr,col,shape,match",
    [
        ([0, 2, 2], [1, 1], (2, 2), "Duplicate"),
        ([0, 1, 2], [1.0, 0.0], (2, 2), "integer array"),
        ([0, 3, 2], [1, 0], (2, 2), "inconsistent"),
        ([0, 1], [0], (2, 2), "inconsistent"),
        ([1, 1, 2], [1, 0], (2, 2), "inconsistent"),
        ([0, 1, 2], [2, 0], (2, 2), "out of bounds"),
        ([0, 1, 2], [-1, 0], (2, 2), "nonnegative"),
        ([0, 0], [], (0, 2), "positive integer"),
        ([0, 1, 2], [1, 0], (2.0, 2), "positive integer"),
        ([0, 1, 2], [1, 0], (True, 2), "positive integer"),
    ],
)
def test_malformed_csr_is_a_backend_error(ptr, col, shape, match):
    with pytest.raises(BackendError, match=match):
        TrainableCSR(ptr, col, shape)


@pytest.mark.parametrize(
    "case", ["half", "mismatched_dtype", "state_width", "value_length", "value_rank"]
)
def test_bad_numerical_inputs_are_backend_errors(case):
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape)
    h = torch.ones(2, 6)
    if case == "half":
        h, values = h.half(), values.half()
    elif case == "mismatched_dtype":
        values = values.double()
    elif case == "state_width":
        h = h[:, :5]
    elif case == "value_length":
        values = values[:5]
    else:
        values = values.reshape(2, 3)
    with pytest.raises(BackendError):
        graph.mm(h, values)


def test_mps_requires_compile_shader_without_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.delattr(torch.mps, "compile_shader", raising=False)
    ptr, col, _, _, shape = csr_fixture()
    with pytest.raises(BackendError, match="compile_shader.*no CPU fallback"):
        TrainableCSR(ptr, col, shape, device="mps")


def small_connectome(alternative=False):
    ids = np.array([10, 20, 30, 40, 50])
    source = np.array([0, 3, 1, 0, 2, 4])
    if alternative:
        source = np.array([1, 4, 0, 1, 3, 2])
    target = np.array([0, 0, 2, 3, 3, 4])
    return Connectome.from_edges(
        source=ids[source],
        target=ids[target],
        weight=[0.4, -0.6, 0.0, 0.8, -0.3, 0.1],
        nodes={"node_id": ids, "cell_type": ["A", "A", "B", "C", "C"]},
    )


def make_model(backend, device="cpu", bounded=False, alternative=False):
    brain = small_connectome(alternative)
    weights = (
        BiologicalWeights(
            brain, prior="weight", gain_bounds=(0.9, 1.1), share_by="cell_type", dale=False
        )
        if bounded
        else "trainable"
    )
    return ConnectomeRNN(
        brain,
        weights=weights,
        initializer="weight",
        input_nodes=[10, 40],
        output_nodes=[30, 10, 50],
        activation="tanh",
        leak=0.7,
        bias=True,
        backend=backend,
        device=device,
    )


def model_inputs():
    generator = torch.Generator().manual_seed(260915)
    initial = torch.randn(3, 5, generator=generator) * 0.25
    drive = torch.randn(3, 4, 2, generator=generator) * 0.3
    coefficient = torch.randn(3, 4, 3, generator=generator)
    return initial, drive, coefficient


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bounded", [False, True])
def test_stock_recurrent_model_forward_and_all_gradients(device, bounded, monkeypatch):
    reference = make_model("dense", bounded=bounded)
    actual = make_model("metal_csr", device, bounded)
    actual.load_state_dict(reference.state_dict(), strict=True)
    initial, drive, coefficient = model_inputs()
    s = initial.clone().requires_grad_()
    x = drive.clone().requires_grad_()
    expected, expected_state = reference(x, state=s, return_state=True)
    ((expected * coefficient).sum() + expected_state.square().sum()).backward()
    ss = initial.to(device).detach().requires_grad_()
    xx = drive.to(device).detach().requires_grad_()
    if device == "mps":
        forbid_sparse_mm(monkeypatch)
    y, state = actual(xx, state=ss, return_state=True)
    ((y * coefficient.to(device)).sum() + state.square().sum()).backward()
    for got, want in ((y, expected), (state, expected_state), (xx.grad, x.grad), (ss.grad, s.grad)):
        torch.testing.assert_close(got.cpu(), want, rtol=5e-5, atol=3e-6)
    ref_params = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0, name
        torch.testing.assert_close(
            parameter.grad.cpu(), ref_params[name].grad, rtol=5e-5, atol=3e-6
        )
    assert actual.propagator.verify_frozen()
    if device == "mps":
        assert actual.propagator.metadata()["kernel_launch_count"] == {
            "forward": 4,
            "state_backward": 4,
            "edge_backward": 4,
        }


def test_metal_edge_batch_memory_estimate_does_not_claim_zero_total_memory():
    model = make_model("metal_csr", bounded=True)
    assert model.activation_bytes(8, 4) == 0
    assert model.diagnostics()["activation_bytes_per_batch_step"] == 0
    assert model.propagator.saves_edge_batch_activations is False
    scatter = make_model("scatter", bounded=True)
    assert scatter.activation_bytes(8, 4) == 2 * scatter.num_edges * 8 * 4 * 4
    assert scatter.diagnostics()["activation_bytes_per_batch_step"] == 2 * scatter.num_edges * 4


@pytest.mark.parametrize("device", DEVICES)
def test_state_loading_rebuilds_graph_and_is_strictly_portable(device):
    initial, drive, _ = model_inputs()
    first = make_model("metal_csr", device)
    second = make_model("metal_csr", device, alternative=True)
    x, state = drive.to(device), initial.to(device)
    with torch.no_grad():
        before = first(x, state=state)
        expected = second(x, state=state)
    assert not torch.allclose(before, expected)
    cached = first.propagator._runtime
    checkpoint = copy.deepcopy(second.state_dict())
    assert not any(name.startswith("propagator.") for name in checkpoint)
    first.load_state_dict(checkpoint, strict=True)
    assert first.propagator._runtime is None
    with torch.no_grad():
        got = first(x, state=state)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert first.propagator._runtime is not cached
    assert first.propagator.verify_frozen()
    for backend in ("dense", "scatter", "sparse_mm"):
        other = make_model(backend)
        other.load_state_dict(checkpoint, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(
                other(drive, state=initial), expected.cpu(), rtol=3e-5, atol=2e-6
            )
        first.load_state_dict(other.state_dict(), strict=True)
        assert first.propagator._runtime is None
        with torch.no_grad():
            torch.testing.assert_close(first(x, state=state), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_to_invalidates_cache_including_same_device_and_inference_mode(device):
    initial, drive, _ = model_inputs()
    model = make_model("metal_csr")
    with torch.no_grad():
        expected = model(drive, state=initial)
    old = model.propagator._runtime
    # Whole-module moves must create ordinary parameters/input indices for later
    # training. Only the backend's derived topology cache promises inference-safe
    # construction and device migration.
    model.to(device)
    assert model.propagator._runtime is None
    with torch.inference_mode():
        model.propagator.to(device)
        assert model.propagator._runtime is None
        got = model(drive.to(device), state=initial.to(device))
    assert model.propagator._runtime is not old
    assert model.propagator._runtime.device.type == device
    assert not model.propagator.edge_index.is_inference()
    torch.testing.assert_close(got.cpu(), expected, rtol=3e-5, atol=2e-6)
    # Use new ordinary inputs, not inference tensors, for the subsequent backward.
    model(drive.to(device), state=initial.to(device)).sum().backward()
    assert model.weights.weight.grad is not None
    model.to("cpu")
    assert model.propagator._runtime is None
    with torch.no_grad():
        torch.testing.assert_close(model(drive, state=initial), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_runtime_constructed_in_inference_mode_can_later_backpropagate(device):
    ptr, col, row, values, shape = csr_fixture()
    with torch.inference_mode():
        graph = TrainableCSR(ptr, col, shape, device=device)
        graph.mm(torch.ones(2, 6, device=device), values.to(device))
    assert all(not tensor.is_inference() for tensor in graph._buffers.values())
    h = torch.ones(2, 6, device=device, requires_grad=True)
    v = values.to(device).detach().requires_grad_()
    y = graph.mm(h, v)
    dh, dv = torch.autograd.grad(y.sum(), (h, v))
    reference_h = h.detach().cpu().requires_grad_()
    reference_v = values.clone().requires_grad_()
    reference = dense_reference(reference_h, reference_v, row, col, shape)
    grads = torch.autograd.grad(reference.sum(), (reference_h, reference_v))
    torch.testing.assert_close(dh.cpu(), grads[0])
    torch.testing.assert_close(dv.cpu(), grads[1])
    assert graph.verify_frozen()


@pytest.mark.parametrize("device", DEVICES)
def test_propagator_construction_and_rebuild_inside_inference_mode(device):
    first, second = small_connectome(), small_connectome(True)
    weights = torch.as_tensor(first.edge_attribute("weight"), dtype=torch.float32, device=device)
    h = torch.arange(10, dtype=torch.float32, device=device).reshape(5, 2) / 10
    with torch.inference_mode():
        propagator = MetalCSRPropagator(torch.tensor(first.edge_index), 5)
        propagator(h.cpu(), weights.cpu())
        propagator.rebuild(torch.tensor(second.edge_index))
        assert propagator._runtime is None
        # This is a real CPU-to-MPS topology transfer for the native test case.
        propagator.to(device)
        propagator(h, weights)
    assert not propagator.edge_index.is_inference()
    x, v = h.clone().requires_grad_(), weights.clone().requires_grad_()
    got = propagator(x, v)
    dx, dv = torch.autograd.grad(got.sum(), (x, v))
    source, target = second.edge_index
    ref_x, ref_v = h.cpu().clone().requires_grad_(), weights.cpu().clone().requires_grad_()
    expected = dense_reference(ref_x.T, ref_v, target, source, (5, 5)).T
    grads = torch.autograd.grad(expected.sum(), (ref_x, ref_v))
    torch.testing.assert_close(got.cpu(), expected)
    torch.testing.assert_close(dx.cpu(), grads[0])
    torch.testing.assert_close(dv.cpu(), grads[1])
    assert propagator.verify_frozen()


def test_explicit_rebuild_is_required_for_endpoint_mutation():
    brain = small_connectome()
    propagator = MetalCSRPropagator(torch.tensor(brain.edge_index), brain.num_nodes)
    h = torch.ones(5, 2)
    values = torch.tensor(brain.edge_attribute("weight"), dtype=torch.float32)
    propagator(h, values)
    propagator.edge_index.data[0, 0] = 1
    with pytest.raises(RuntimeError, match="endpoints differ"):
        propagator.verify_frozen()
    other = small_connectome(True)
    propagator.rebuild(torch.tensor(other.edge_index))
    source, target = other.edge_index
    torch.testing.assert_close(
        propagator(h, values), dense_reference(h.T, values, target, source, (5, 5)).T
    )
    assert propagator.verify_frozen()


@pytest.mark.parametrize(
    "edges,nodes,match",
    [
        ([[0, 0], [1, 1]], 3, "parallel edges"),
        ([[0, 1], [2, 1]], 3, "sorted by target"),
        ([[0, 3], [1, 2]], 3, "outside the node range"),
        ([[0, -1], [1, 2]], 3, "outside the node range"),
        ([[0.0, 1.0], [1.0, 2.0]], 3, "integer endpoints"),
        ([[0, 1], [1, 2]], True, "positive integer node count"),
        ([[0, 1], [1, 2]], 3.0, "positive integer node count"),
        ([[0, 1]], 3, "edge_index"),
    ],
)
def test_propagator_rejects_bad_topology(edges, nodes, match):
    with pytest.raises(BackendError, match=match):
        MetalCSRPropagator(torch.tensor(edges), nodes)


def test_rejected_rebuild_preserves_existing_working_runtime():
    brain = small_connectome()
    propagator = build_propagator("metal_csr", torch.tensor(brain.edge_index), 5, trainable=True)
    h, values = torch.ones(5, 2), torch.ones(brain.num_edges)
    expected = propagator(h, values)
    old = propagator._runtime
    with pytest.raises(BackendError, match="parallel edges"):
        propagator.rebuild(torch.tensor([[0, 0], [1, 1]]))
    assert propagator._runtime is old
    torch.testing.assert_close(propagator(h, values), expected, rtol=0, atol=0)


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="needs native Apple Metal")
def test_mps_rejects_mixed_device_and_half_inputs():
    ptr, col, _, values, shape = csr_fixture()
    graph = TrainableCSR(ptr, col, shape, device="mps")
    with pytest.raises(BackendError, match="resident"):
        graph.mm(torch.ones(2, 6), values.to("mps"))
    with pytest.raises(BackendError, match="float32"):
        graph.mm(torch.ones(2, 6, device="mps", dtype=torch.float16), values.to("mps").half())


@pytest.mark.mps
@pytest.mark.skipif(not MPS_AVAILABLE, reason="needs native Apple Metal")
def test_biological_weights_optimizer_matches_dense_and_moves_gains(monkeypatch):
    """Run on macm3 only: four actual updates through the stock weight module."""
    reference = make_model("dense", bounded=True)
    actual = make_model("metal_csr", "mps", bounded=True)
    actual.load_state_dict(reference.state_dict(), strict=True)
    initial, drive, coefficient = model_inputs()
    prior = actual.edge_weight.detach().cpu().clone()
    before = actual.weights.raw_gain.detach().cpu().clone()
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.015)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=0.015)
    forbid_sparse_mm(monkeypatch)
    for _ in range(4):
        reference_optimizer.zero_grad(set_to_none=True)
        actual_optimizer.zero_grad(set_to_none=True)
        expected, expected_state = reference(drive, state=initial, return_state=True)
        got, state = actual(drive.to("mps"), state=initial.to("mps"), return_state=True)
        loss = (expected - coefficient).square().mean() + expected_state.square().mean()
        actual_loss = (got - coefficient.to("mps")).square().mean() + state.square().mean()
        loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(actual_loss.cpu(), loss, rtol=5e-5, atol=3e-6)
        for (name, parameter), (ref_name, ref_parameter) in zip(
            actual.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
            torch.testing.assert_close(
                parameter.grad.cpu(), ref_parameter.grad, rtol=1e-4, atol=3e-6
            )
        actual_optimizer.step()
        reference_optimizer.step()
        for parameter, ref_parameter in zip(
            actual.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(parameter.cpu(), ref_parameter, rtol=1e-4, atol=3e-6)
    assert not torch.equal(actual.weights.raw_gain.detach().cpu(), before)
    after = actual.edge_weight.detach().cpu()
    assert not torch.equal(after, prior)
    assert torch.equal(after.sign(), prior.sign())
    nonzero = prior != 0
    ratios = after[nonzero] / prior[nonzero]
    assert torch.all((ratios >= 0.9) & (ratios <= 1.1))
    assert torch.count_nonzero(after[~nonzero]) == 0
    assert actual.propagator.verify_frozen()
