"""A frozen numerical fixture, so the runtime's semantics cannot drift unnoticed.

If this test fails, the network computes something different from what it used to.
That is either a bug you just introduced, or a deliberate change, in which case
regenerate the fixture with::

    python -m pytest tests/regression/test_golden_output.py --regenerate-golden

and say what changed in CHANGELOG.md. Never regenerate to make a red test green
without understanding why it went red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from connectorch import Connectome
from connectorch.nn import ConnectomeRNN

GOLDEN = Path(__file__).parent / "golden_output.json"
SEED = 20260911


def fixture_connectome() -> Connectome:
    """A fixed 12-node graph. Never change this without regenerating the golden file."""
    rng = np.random.default_rng(SEED)
    return Connectome.from_edges(
        source=rng.integers(0, 12, 40),
        target=rng.integers(0, 12, 40),
        synapse_count=rng.integers(1, 30, 40),
        nodes={"node_id": np.arange(12)},
    )


def fixture_model() -> ConnectomeRNN:
    torch.manual_seed(SEED)
    return ConnectomeRNN(
        fixture_connectome(),
        weights="normalized_synapse_count",
        activation="tanh",
        leak=0.4,
        dtype=torch.float64,
    )


def fixture_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn(2, 12, generator=generator, dtype=torch.float64)


def compute() -> dict:
    model = fixture_model()
    output = model(fixture_input(), steps=3)
    brain = fixture_connectome()
    return {
        "fingerprint": brain.fingerprint(),
        "num_nodes": brain.num_nodes,
        "num_edges": brain.num_edges,
        "edge_weight": model.edge_weight.tolist(),
        "output": output.tolist(),
    }


def test_golden_output_is_unchanged(request) -> None:
    current = compute()

    if request.config.getoption("--regenerate-golden", default=False):  # pragma: no cover
        GOLDEN.write_text(json.dumps(current, indent=2))
        pytest.skip("golden fixture regenerated")

    if not GOLDEN.exists():  # pragma: no cover - first run only
        GOLDEN.write_text(json.dumps(current, indent=2))
        pytest.skip("golden fixture created")

    expected = json.loads(GOLDEN.read_text())
    assert current["fingerprint"] == expected["fingerprint"], "the graph itself changed"
    assert current["num_edges"] == expected["num_edges"]
    assert np.allclose(current["edge_weight"], expected["edge_weight"], atol=1e-12), (
        "weight initialisation changed"
    )
    assert np.allclose(current["output"], expected["output"], atol=1e-12), (
        "the recurrence computes something different from what it used to"
    )


def test_the_fixture_is_deterministic_within_a_run() -> None:
    assert compute()["output"] == compute()["output"]
