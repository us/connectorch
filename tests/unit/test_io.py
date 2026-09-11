"""Tests for the file-format and NetworkX adapters."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from connectorch import Connectome, ConnectorchError
from connectorch.io import connectome_from_table, from_networkx, read_edge_table, to_networkx

networkx = pytest.importorskip("networkx", reason="optional dependency")


@pytest.fixture
def edge_table() -> pa.Table:
    return pa.table(
        {
            "body_pre": [10, 20, 30],
            "body_post": [20, 30, 10],
            "weight": [5, 6, 7],
            "ignored": ["a", "b", "c"],
        }
    )


def test_reads_csv_parquet_and_feather(tmp_path, edge_table: pa.Table) -> None:
    import pyarrow.csv as pa_csv
    import pyarrow.feather as feather

    pq.write_table(edge_table, tmp_path / "edges.parquet")
    feather.write_feather(edge_table, tmp_path / "edges.feather")
    pa_csv.write_csv(edge_table, tmp_path / "edges.csv")

    for name in ("edges.parquet", "edges.feather", "edges.csv"):
        table = read_edge_table(tmp_path / name)
        assert table.column("body_pre").to_pylist() == [10, 20, 30]


def test_column_projection_skips_the_rest(tmp_path, edge_table: pa.Table) -> None:
    pq.write_table(edge_table, tmp_path / "edges.parquet")
    table = read_edge_table(tmp_path / "edges.parquet", columns=["body_pre", "body_post"])
    assert table.column_names == ["body_pre", "body_post"]


def test_unknown_suffix_asks_for_the_format(tmp_path) -> None:
    (tmp_path / "edges.bin").write_bytes(b"")
    with pytest.raises(ConnectorchError, match="pass format="):
        read_edge_table(tmp_path / "edges.bin")


def test_connectome_from_table_renames_columns(edge_table: pa.Table) -> None:
    brain = connectome_from_table(
        edge_table,
        source_column="body_pre",
        target_column="body_post",
        attributes={"weight": "synapse_count"},
    )
    assert brain.num_nodes == 3
    assert brain.edge_columns == ("synapse_count",)
    assert "ignored" not in brain.edge_columns


def test_missing_column_names_what_is_available(edge_table: pa.Table) -> None:
    with pytest.raises(ConnectorchError, match="available"):
        connectome_from_table(edge_table, source_column="nope", target_column="body_post")


def test_networkx_round_trip() -> None:
    graph = networkx.DiGraph()
    graph.add_edge(1, 2, weight=3.0)
    graph.add_edge(2, 3, weight=4.0)
    graph.nodes[1]["region"] = "ME_R"
    graph.nodes[2]["region"] = "ME_L"
    graph.nodes[3]["region"] = "ME_R"

    brain = from_networkx(graph, node_attributes=["region"])
    assert brain.num_nodes == 3
    assert brain.num_edges == 2
    assert brain.where(region="ME_R").tolist() == [1, 3]

    back = to_networkx(brain)
    assert set(back.edges) == set(graph.edges)
    assert back[1][2]["weight"] == 3.0


def test_undirected_graphs_are_refused() -> None:
    graph = networkx.Graph()
    graph.add_edge(1, 2)
    with pytest.raises(ConnectorchError, match="directed"):
        from_networkx(graph)


def test_empty_graph_is_refused() -> None:
    with pytest.raises(ConnectorchError, match="no edges"):
        from_networkx(networkx.DiGraph())


def test_save_reload_survives_a_round_trip_through_arrow(tmp_path) -> None:
    """Reloading from parquet must not change any identifier or attribute."""
    brain = Connectome(
        nodes={"node_id": np.array([7, 100, 5000]), "cell_class": ["a", "b", "c"]},
        edges={"source": [7, 100], "target": [100, 5000], "synapse_count": [3, 4]},
    )
    reloaded = Connectome.load(brain.save(tmp_path / "b.ct"))
    assert reloaded.fingerprint() == brain.fingerprint()
    assert reloaded.where(cell_class="b").tolist() == [100]
