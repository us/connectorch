"""Tests for the topology-matched controls.

A control that quietly differs in size or degree from the graph it controls for
would make every comparison built on it meaningless, so those properties are
asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectorch import Connectome
from connectorch.transforms import (
    collapse_ei,
    degree_preserving_rewire,
    random_topology,
    shuffle_edge_weights,
    shuffle_signs,
)


@pytest.fixture
def brain() -> Connectome:
    rng = np.random.default_rng(0)
    return Connectome.from_edges(
        source=rng.integers(0, 80, 600),
        target=rng.integers(0, 80, 600),
        synapse_count=rng.integers(1, 40, 600),
        nodes={"node_id": np.arange(80)},
    )


def degrees(connectome: Connectome) -> tuple[np.ndarray, np.ndarray]:
    source, target = connectome.edge_index
    n = connectome.num_nodes
    return np.bincount(source, minlength=n), np.bincount(target, minlength=n)


def test_random_topology_matches_the_edge_count_exactly(brain: Connectome) -> None:
    """Drawing endpoints independently produces duplicates the IR then merges."""
    control = random_topology(brain, seed=0)
    assert control.num_edges == brain.num_edges
    assert control.num_nodes == brain.num_nodes


def test_random_topology_actually_rewires(brain: Connectome) -> None:
    control = random_topology(brain, seed=0)
    assert not np.array_equal(control.edge_index, brain.edge_index)


def test_degree_preserving_rewire_preserves_both_degrees(brain: Connectome) -> None:
    control = degree_preserving_rewire(brain, seed=0)
    assert control.num_edges == brain.num_edges
    out_before, in_before = degrees(brain)
    out_after, in_after = degrees(control)
    assert np.array_equal(out_before, out_after)
    assert np.array_equal(in_before, in_after)
    assert not np.array_equal(control.edge_index, brain.edge_index)


def test_shuffle_edge_weights_keeps_the_wiring(brain: Connectome) -> None:
    control = shuffle_edge_weights(brain, seed=0)
    assert np.array_equal(control.edge_index, brain.edge_index)
    assert not np.array_equal(
        control.edge_attribute("synapse_count"), brain.edge_attribute("synapse_count")
    )
    assert (
        control.edge_attribute("synapse_count").sum() == brain.edge_attribute("synapse_count").sum()
    )


@pytest.mark.parametrize(
    "transform", [random_topology, degree_preserving_rewire, shuffle_edge_weights]
)
def test_controls_are_labelled_as_controls(transform, brain: Connectome) -> None:
    """A control must never be mistakable for biological data downstream."""
    record = transform(brain, seed=0).provenance["history"][-1]
    assert record["control_for"] == brain.fingerprint()
    assert "not biological data" in record["note"]


def signed_brain() -> Connectome:
    rng = np.random.default_rng(1)
    n = 40
    source = rng.integers(0, n, 200)
    target = rng.integers(0, n, 200)
    base = Connectome.from_edges(
        source=source,
        target=target,
        synapse_count=rng.integers(1, 40, 200),
        nodes={"node_id": np.arange(n)},
    )
    columns = {name: base.edge_attribute(name) for name in base.edge_columns}
    columns["sign"] = rng.choice(np.array([-1, 1], dtype=np.int8), size=base.num_edges).astype(
        np.int8
    )
    return Connectome(
        nodes={"node_id": base.node_ids},
        edges={
            "source": base.node_ids[base.edge_index[0]],
            "target": base.node_ids[base.edge_index[1]],
            **columns,
        },
        provenance=dict(base.provenance),
    )


def test_shuffle_signs_permutes_only_signs() -> None:
    brain = signed_brain()
    control = shuffle_signs(brain, seed=0)
    assert np.array_equal(control.edge_index, brain.edge_index)
    assert sorted(control.edge_attribute("sign").tolist()) == sorted(
        brain.edge_attribute("sign").tolist()
    )
    assert not np.array_equal(control.edge_attribute("sign"), brain.edge_attribute("sign"))
    assert np.array_equal(
        control.edge_attribute("synapse_count"), brain.edge_attribute("synapse_count")
    )


def test_collapse_ei_removes_all_inhibition() -> None:
    brain = signed_brain()
    control = collapse_ei(brain)
    assert np.array_equal(control.edge_index, brain.edge_index)
    assert set(np.unique(control.edge_attribute("sign")).tolist()) == {1}
    assert np.array_equal(
        control.edge_attribute("synapse_count"), brain.edge_attribute("synapse_count")
    )


@pytest.mark.parametrize(
    "transform", [random_topology, degree_preserving_rewire, shuffle_edge_weights]
)
def test_controls_are_reproducible(transform, brain: Connectome) -> None:
    assert transform(brain, seed=3).fingerprint() == transform(brain, seed=3).fingerprint()
    assert transform(brain, seed=3).fingerprint() != transform(brain, seed=4).fingerprint()
