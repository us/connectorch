"""The absolute memory invariant, enforced as a test.

Accidentally densifying a 166,691 x 166,691 adjacency costs 103.5 GiB and takes the
process with it. The runtime is written so this cannot happen, and this test makes
sure it stays that way as the code changes: no production module may densify.

``backends/dense.py`` is the one exception. It exists as a correctness oracle, it
refuses graphs above a gibibyte at construction, and it says so in its own module
docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "connectorch"

#: Modules allowed to build a dense adjacency, with the reason.
ALLOWED = {"backends/dense.py": "the reference backend, which refuses large graphs"}

#: Call names that densify an adjacency.
FORBIDDEN_CALLS = {"to_dense", "todense", "to_sparse_dense"}


def production_modules() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*.py")
        if path.relative_to(SOURCE_ROOT).as_posix() not in ALLOWED
    )


@pytest.mark.parametrize("path", production_modules(), ids=lambda p: p.name)
def test_no_production_module_densifies(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = [
        f"{path.name}:{node.lineno} calls .{node.func.attr}()"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in FORBIDDEN_CALLS
    ]
    assert not offenders, (
        f"{offenders} would materialise a dense adjacency. Only {sorted(ALLOWED)} may do that."
    )


def test_the_allowed_module_still_exists() -> None:
    """If dense.py is renamed, this test must be updated rather than quietly passing."""
    for relative in ALLOWED:
        assert (SOURCE_ROOT / relative).exists(), (
            f"{relative} is on the densification allowlist but no longer exists; "
            "update ALLOWED in this test."
        )


def test_a_large_graph_never_allocates_n_squared() -> None:
    """Compile a graph too large to densify and run a step. It must simply work."""
    import numpy as np
    import torch

    from connectorch import Connectome
    from connectorch.nn import ConnectomeRNN

    num_nodes, num_edges = 60_000, 200_000
    rng = np.random.default_rng(0)
    brain = Connectome.from_edges(
        source=rng.integers(0, num_nodes, num_edges),
        target=rng.integers(0, num_nodes, num_edges),
        synapse_count=rng.integers(1, 20, num_edges),
        nodes={"node_id": np.arange(num_nodes)},
    )
    dense_gib = num_nodes * num_nodes * 4 / 2**30
    assert dense_gib > 10, "the fixture must be big enough for densification to be fatal"

    model = ConnectomeRNN(brain, weights="trainable", output_nodes=np.arange(64))
    assert model.backend == "scatter"
    y = model(torch.zeros(2, brain.num_nodes), steps=2)
    y.square().mean().backward()
    assert model.edge_weight.grad is not None
    assert torch.isfinite(model.edge_weight.grad).all()
