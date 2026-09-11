"""Tests for the Connectome intermediate representation."""

from __future__ import annotations

import numpy as np
import pytest

from connectorch import Connectome, ConnectomeValidationError, ConnectorchError


def test_indexing_is_sorted_and_deterministic() -> None:
    brain = Connectome.from_edges(source=[100, 5000, 7], target=[5000, 7, 100])
    assert brain.node_ids.tolist() == [7, 100, 5000]
    assert brain.index_of(5000) == 2
    assert brain.id_of(0) == 7


def test_indexing_is_independent_of_input_row_order() -> None:
    a = Connectome.from_edges(source=[100, 5000, 7], target=[5000, 7, 100], synapse_count=[3, 4, 5])
    b = Connectome.from_edges(source=[7, 100, 5000], target=[100, 5000, 7], synapse_count=[5, 3, 4])
    assert a.node_ids.tolist() == b.node_ids.tolist()
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_when_a_weight_changes() -> None:
    a = Connectome.from_edges(source=[0, 1], target=[1, 0], synapse_count=[1, 2])
    b = Connectome.from_edges(source=[0, 1], target=[1, 0], synapse_count=[1, 3])
    assert a.fingerprint() != b.fingerprint()


def test_edge_with_unknown_endpoint_is_rejected_with_a_useful_message() -> None:
    with pytest.raises(ConnectomeValidationError) as excinfo:
        Connectome.from_edges(source=[1], target=[50038], nodes={"node_id": [1, 2]})
    message = str(excinfo.value)
    assert "edge 0" in message
    assert "50038" in message
    assert "np.int64" not in message, "numpy repr leaked into a user-facing message"


def test_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(ConnectomeValidationError, match="duplicate node ids"):
        Connectome.from_edges(source=[1], target=[2], nodes={"node_id": [1, 2, 2]})


def test_float_node_ids_are_rejected() -> None:
    with pytest.raises(ConnectomeValidationError, match="precision"):
        Connectome.from_edges(source=np.array([1.0]), target=np.array([2.0]))


def test_negative_synapse_count_is_rejected() -> None:
    with pytest.raises(ConnectomeValidationError, match="negative"):
        Connectome.from_edges(source=[0], target=[1], synapse_count=[-4])


def test_self_loops_survive_construction() -> None:
    brain = Connectome.from_edges(source=[0, 1], target=[0, 1])
    assert brain.num_edges == 2
    source, target = brain.edge_index
    assert (source == target).all()


def test_self_loops_are_only_dropped_when_asked() -> None:
    brain = Connectome.from_edges(source=[0, 1, 0], target=[0, 1, 1])
    assert brain.filter_edges().num_edges == 3
    assert brain.filter_edges(drop_self_loops=True).num_edges == 1


def test_parallel_edges_are_aggregated_and_recorded() -> None:
    brain = Connectome.from_edges(source=[0, 0, 1], target=[1, 1, 0], synapse_count=[2, 3, 7])
    assert brain.num_edges == 2
    assert brain.provenance["parallel_edges_merged"] == 1
    source, target = brain.edge_index
    merged = brain.edge_attribute("synapse_count")[(source == 0) & (target == 1)]
    assert merged.tolist() == [5]


def test_parallel_edges_can_be_kept() -> None:
    brain = Connectome.from_edges(
        source=[0, 0], target=[1, 1], synapse_count=[2, 3], aggregate_parallel_edges=False
    )
    assert brain.num_edges == 2


def test_edges_are_sorted_by_target_then_source() -> None:
    rng = np.random.default_rng(0)
    source = rng.integers(0, 20, 200)
    target = rng.integers(0, 20, 200)
    brain = Connectome.from_edges(source=source, target=target)
    src, dst = brain.edge_index
    key = dst.astype(np.int64) * 20 + src
    assert (np.diff(key) > 0).all(), "canonical edge order must be strictly increasing"


def test_where_filters_on_node_attributes() -> None:
    brain = Connectome(
        nodes={"node_id": [1, 2, 3], "region": ["ME_R", "ME_L", "ME_R"]},
        edges={"source": [1, 2], "target": [2, 3]},
    )
    assert brain.where(region="ME_R").tolist() == [1, 3]
    assert brain.where(region=["ME_R", "ME_L"]).tolist() == [1, 2, 3]


def test_subgraph_reindexes_and_records_what_was_dropped() -> None:
    brain = Connectome.from_edges(source=[1, 2, 3], target=[2, 3, 1])
    sub = brain.subgraph([1, 2])
    assert sub.num_nodes == 2
    assert sub.num_edges == 1
    assert sub.node_ids.tolist() == [1, 2]
    assert sub.provenance["history"][-1]["edges_dropped"] == 2


def test_filter_edges_records_provenance() -> None:
    brain = Connectome.from_edges(source=[0, 1], target=[1, 0], synapse_count=[1, 9])
    kept = brain.filter_edges(min_synapses=5)
    assert kept.num_edges == 1
    assert kept.num_nodes == 2, "filtering edges must not remove nodes"
    assert kept.provenance["history"][-1]["min_synapses"] == 5


def test_save_load_round_trip_preserves_everything(tmp_path) -> None:
    brain = Connectome(
        nodes={"node_id": [7, 100, 5000], "cell_class": ["a", "b", "c"]},
        edges={"source": [7, 100], "target": [100, 5000], "synapse_count": [3, 4]},
        provenance={"dataset": "test:v1", "license": "CC-BY"},
    )
    path = brain.save(tmp_path / "brain.ct")
    loaded = Connectome.load(path)

    assert loaded.node_ids.tolist() == brain.node_ids.tolist()
    assert loaded.edge_index.tolist() == brain.edge_index.tolist()
    assert loaded.edge_attribute("synapse_count").tolist() == [3, 4]
    assert loaded.nodes.column("cell_class").to_pylist() == ["a", "b", "c"]
    assert loaded.provenance["dataset"] == "test:v1"
    assert loaded.fingerprint() == brain.fingerprint()


def test_load_detects_tampering(tmp_path) -> None:
    brain = Connectome.from_edges(source=[0, 1], target=[1, 0], synapse_count=[1, 2])
    path = brain.save(tmp_path / "brain.ct")
    tampered = Connectome.from_edges(source=[0, 1], target=[1, 0], synapse_count=[1, 99])
    import pyarrow.parquet as pq

    pq.write_table(tampered.edges, path / "edges.parquet")
    with pytest.raises(ConnectorchError, match="fingerprint"):
        Connectome.load(path)


def test_no_pickle_on_disk(tmp_path) -> None:
    brain = Connectome.from_edges(source=[0], target=[1])
    path = brain.save(tmp_path / "brain.ct")
    names = sorted(p.name for p in path.iterdir())
    assert names == ["edges.parquet", "metadata.json", "nodes.parquet"]
