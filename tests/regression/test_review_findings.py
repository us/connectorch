"""Regression tests for defects that produced a wrong answer without complaining.

Every test here failed before its fix. They are kept together, rather than spread
across the unit suite, so that the class of mistake this library has already made
once is visible in one place and cannot quietly return.
"""

from __future__ import annotations

import functools
import http.server
import threading

import numpy as np
import pytest
import torch

from connectorch import Connectome, ConnectomeValidationError, ConnectorchError
from connectorch.nn import ConnectomeRNN

BACKENDS = ["dense", "sparse_mm", "scatter"]


def test_string_ids_survive_a_save_and_reload(tmp_path) -> None:
    """Object arrays hash their pointers, so the fingerprint changed every process."""
    brain = Connectome.from_edges(source=["alpha"], target=["beta"], weight=[1.0])
    before = brain.fingerprint()
    loaded = Connectome.load(brain.save(tmp_path / "s.ct"))
    assert loaded.fingerprint() == before
    assert loaded.node_ids.tolist() == ["alpha", "beta"]


def test_string_fingerprint_is_stable_across_construction_routes() -> None:
    from_lists = Connectome.from_edges(source=["a"], target=["b"])
    from_arrays = Connectome.from_edges(
        source=np.array(["a"], dtype=object), target=np.array(["b"], dtype=object)
    )
    assert from_lists.fingerprint() == from_arrays.fingerprint()


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_state_dict_rebuilds_the_backend_layout(backend: str) -> None:
    """A loaded checkpoint changes edge_index; derived layouts must follow it."""
    nodes = {"node_id": [0, 1, 2]}
    source = Connectome.from_edges(source=[0], target=[1], weight=[2.0], nodes=nodes)
    target = Connectome.from_edges(source=[0], target=[2], weight=[2.0], nodes=nodes)

    donor = ConnectomeRNN(
        source, weights="weight", activation="identity", leak=1.0, backend=backend
    )
    receiver = ConnectomeRNN(
        target, weights="weight", activation="identity", leak=1.0, backend=backend
    )
    receiver.load_state_dict(donor.state_dict())

    out = receiver(torch.zeros(1, 3), steps=1, state=torch.tensor([[1.0, 0.0, 0.0]]))
    assert out[0, 0].tolist() == [0.0, 2.0, 0.0], "propagated along the old topology"


def test_aggregating_narrow_integers_does_not_wrap() -> None:
    """int8 100 + 100 used to aggregate to -56."""
    brain = Connectome.from_edges(
        source=[0, 0], target=[1, 1], synapse_count=np.array([100, 100], dtype=np.int8)
    )
    assert brain.edge_attribute("synapse_count").tolist() == [200]


def test_aggregating_large_integers_keeps_every_digit() -> None:
    """Summing through float64 lost the last digits above 2**53."""
    brain = Connectome.from_edges(
        source=[0, 0], target=[1, 1], synapse_count=np.array([2**53, 1], dtype=np.int64)
    )
    assert int(brain.edge_attribute("synapse_count")[0]) == 2**53 + 1


def test_mixed_integer_and_string_ids_are_rejected() -> None:
    """numpy turns [1, '1'] into ['1', '1'], merging two different neurons."""
    with pytest.raises(ConnectomeValidationError, match="mixes types"):
        Connectome.from_edges(source=[1, "1"], target=["x", "x"], synapse_count=[2, 3])


def test_node_columns_must_match_the_node_count() -> None:
    """Extra annotation values used to be dropped without a word."""
    with pytest.raises(ConnectomeValidationError, match="node column 'label'"):
        Connectome(
            nodes={"node_id": [0, 1], "label": ["a", "b", "lost"]},
            edges={"source": [0], "target": [1]},
        )


def test_reserved_edge_column_names_are_rejected() -> None:
    """An attribute called source_index used to overwrite the real endpoints on export."""
    with pytest.raises(ConnectomeValidationError, match="reserved"):
        Connectome.from_edges([0], [1], source_index=[1], target_index=[0])


def test_unsigned_endpoints_map_to_the_right_node() -> None:
    """uint64 against int64 promoted both to float64, collapsing ids above 2**53."""
    ids = np.array([2**53, 2**53 + 1], dtype=np.int64)
    brain = Connectome.from_edges(
        np.array([2**53 + 1], dtype=np.uint64),
        np.array([2**53], dtype=np.int64),
        nodes={"node_id": ids},
    )
    assert brain.edge_index.tolist() == [[1], [0]]


def test_index_of_rejects_a_float_that_names_no_node() -> None:
    """float(2**53) used to match the id 2**53+1 after promotion."""
    brain = Connectome.from_edges([2**53 + 1], [2**53 + 1])
    with pytest.raises(KeyError):
        brain.index_of(float(2**53))
    with pytest.raises(KeyError):
        brain.index_of(1.5)


def test_object_attributes_do_not_reintroduce_row_order_dependence() -> None:
    """Object columns were skipped as sort tie-breakers, so aggregation drifted."""
    forward = Connectome.from_edges(
        source=np.array([0, 0], dtype=object),
        target=np.array([1, 1], dtype=object),
        label=np.array(["z", "a"], dtype=object),
    )
    reversed_rows = Connectome.from_edges(
        source=np.array([0, 0], dtype=object),
        target=np.array([1, 1], dtype=object),
        label=np.array(["a", "z"], dtype=object),
    )
    assert (
        forward.edge_attribute("label").tolist() == reversed_rows.edge_attribute("label").tolist()
    )


def test_an_inhibitory_edge_inhibits() -> None:
    """sign=-1 was validated and then ignored, turning inhibition into excitation."""
    brain = Connectome.from_edges([0], [1], weight=[2.0], sign=[-1])
    model = ConnectomeRNN(brain, weights="weight", activation="identity", leak=1.0)
    out = model(torch.zeros(1, 2), steps=1, state=torch.tensor([[1.0, 0.0]]))
    assert out[0, 0].tolist() == [0.0, -2.0]


def test_unknown_sign_stays_excitatory() -> None:
    brain = Connectome.from_edges([0], [1], weight=[2.0], sign=[0])
    model = ConnectomeRNN(brain, weights="weight", activation="identity", leak=1.0)
    out = model(torch.zeros(1, 2), steps=1, state=torch.tensor([[1.0, 0.0]]))
    assert out[0, 0].tolist() == [0.0, 2.0]


def test_a_truncated_download_reports_the_size_it_got(tmp_path) -> None:
    """The size check stat-ed the partial file after deleting it, raising the wrong error."""
    from connectorch.datasets._cache import ensure_file

    served = tmp_path / "served"
    served.mkdir()
    (served / "f.bin").write_bytes(b"abc")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ConnectorchError, match="gave 3 bytes but 5"):
            ensure_file(
                f"http://127.0.0.1:{server.server_port}/f.bin",
                cache_dir=tmp_path / "cache",
                expected_size=5,
                download=True,
                progress=False,
            )
    finally:
        server.shutdown()


# ----------------------------------------------------------------------
# identifiers, hashing and numeric columns
# ----------------------------------------------------------------------


def test_float_group_sums_do_not_cancel() -> None:
    """Differencing a global prefix sum lost small weights next to large ones."""
    brain = Connectome.from_edges([0, 0, 1], [0, 0, 1], weight=[1e16, 1e16, 1.0])
    assert brain.edge_attribute("weight").tolist() == [2e16, 1.0]


def test_unsigned_counts_do_not_go_negative_when_aggregated() -> None:
    """Widening uint64 to int64 wrapped a large count past the sign bit."""
    brain = Connectome.from_edges(
        [0, 0], [1, 1], synapse_count=np.array([2**63, 1], dtype=np.uint64)
    )
    assert int(brain.edge_attribute("synapse_count")[0]) == 2**63 + 1


def test_a_single_edge_group_keeps_its_own_value() -> None:
    brain = Connectome.from_edges([0, 1], [1, 2], synapse_count=[7, 9])
    assert brain.edge_attribute("synapse_count").tolist() == [7, 9]


def test_controls_keep_the_input_aggregation_policy() -> None:
    """A control built from a graph that kept parallel edges must keep them too."""
    from connectorch.transforms import degree_preserving_rewire, shuffle_edge_weights

    brain = Connectome.from_edges([0, 0], [1, 1], weight=[1.0, 2.0], aggregate_parallel_edges=False)
    assert degree_preserving_rewire(brain, seed=0).num_edges == 2
    assert shuffle_edge_weights(brain, seed=0).num_edges == 2


def test_networkx_import_keeps_isolated_nodes() -> None:
    """A neuron with no reconstructed connections is still part of the connectome."""
    networkx = pytest.importorskip("networkx")
    from connectorch.io import from_networkx

    graph = networkx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)
    assert from_networkx(graph).num_nodes == 3


def test_networkx_export_keeps_parallel_edges() -> None:
    """A DiGraph holds one edge per pair, so exporting to one silently dropped weights."""
    networkx = pytest.importorskip("networkx")
    from connectorch.io import to_networkx

    brain = Connectome.from_edges([0, 0], [1, 1], weight=[2.0, 3.0], aggregate_parallel_edges=False)
    exported = to_networkx(brain)
    assert isinstance(exported, networkx.MultiDiGraph)
    assert sorted(d["weight"] for _, _, d in exported.edges(data=True)) == [2.0, 3.0]

    simple = to_networkx(Connectome.from_edges([0, 1], [1, 2], weight=[2.0, 3.0]))
    assert isinstance(simple, networkx.DiGraph)
    assert not isinstance(simple, networkx.MultiDiGraph)


def test_null_object_ids_are_rejected_not_stringified() -> None:
    """astype(str) turned None into the perfectly valid node id 'None'."""
    with pytest.raises(ConnectomeValidationError, match="null identifiers"):
        Connectome.from_edges(
            np.array([None, "None"], dtype=object),
            np.array(["x", "x"], dtype=object),
            weight=[2.0, 3.0],
        )


def test_renaming_an_attribute_onto_an_endpoint_is_rejected() -> None:
    """Renaming a column to 'source' silently rewrote the graph's topology."""
    import pyarrow as pa

    from connectorch.io import connectome_from_table

    table = pa.table({"pre": [10], "post": [20], "w": [30]})
    with pytest.raises(ConnectorchError, match="would be renamed"):
        connectome_from_table(
            table, source_column="pre", target_column="post", attributes={"w": "source"}
        )


def test_index_of_never_matches_through_an_in_band_sentinel() -> None:
    """The 'missing' sentinel was int64 min, which is a perfectly valid node id."""
    brain = Connectome.from_edges([-(2**63)], [-(2**63)])
    for absent in (1.5, float("nan"), float("inf")):
        with pytest.raises(KeyError):
            brain.index_of(absent)
    assert brain.index_of(-(2**63)) == 0


def test_string_fingerprints_do_not_collide_across_a_separator() -> None:
    """Joining ids on a NUL let an id containing a NUL impersonate two ids."""
    a = Connectome.from_edges(["a"], ["b\x00c"])
    b = Connectome.from_edges(["a\x00b"], ["c"])
    assert a.fingerprint() != b.fingerprint()


def test_shared_parameters_are_counted_once() -> None:
    """Two cores sharing one weight tensor made 'other' go negative."""
    from torch import nn

    from connectorch.nn import count_parameters

    brain = Connectome.from_edges([0], [1], weight=[1.0])
    first = ConnectomeRNN(brain, weights="trainable", initializer="weight")
    second = ConnectomeRNN(brain, weights="trainable", initializer="weight")
    second.edge_weight = first.edge_weight

    counts = count_parameters(nn.ModuleList([first, second]))
    assert counts == {"total": 1, "connectome": 1, "other": 0}


# ----------------------------------------------------------------------
# download integrity and fingerprint coverage
# ----------------------------------------------------------------------


def test_fingerprint_distinguishes_large_synapse_counts() -> None:
    """Hashing counts through float64 made 2**53 and 2**53+1 identical."""
    a = Connectome.from_edges([0], [1], synapse_count=[2**53])
    b = Connectome.from_edges([0], [1], synapse_count=[2**53 + 1])
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_distinguishes_string_ids_from_integer_ids() -> None:
    """A graph of empty strings hashed the same as a graph of zeros."""
    strings = Connectome.from_edges([""], [""])
    integers = Connectome.from_edges([0], [0])
    assert strings.fingerprint() != integers.fingerprint()


def test_cli_validate_checks_attribute_columns(tmp_path, capsys) -> None:
    """validate dropped every non-endpoint column, so it called bad tables valid."""
    from connectorch.cli.main import main

    path = tmp_path / "bad.csv"
    path.write_text("source,target,synapse_count\n0,1,-1\n")
    assert main(["validate", str(path)]) == 1
    assert "negative" in capsys.readouterr().err

    good = tmp_path / "good.csv"
    good.write_text("source,target,synapse_count\n0,1,5\n")
    assert main(["validate", str(good)]) == 0


# ----------------------------------------------------------------------
# initialisation and dtype validation
# ----------------------------------------------------------------------


def test_the_readme_quickstart_runs_as_written() -> None:
    """The default initializer demanded synapse counts a weight-only graph lacks."""
    brain = Connectome.from_edges(
        source=["A", "B", "C", "C"],
        target=["B", "C", "A", "B"],
        weight=[1.0, 0.5, 0.2, 0.8],
    )
    model = ConnectomeRNN(brain, weights="trainable", activation="tanh", leak=0.5)
    assert model.initializer == "weight"
    y = model(torch.randn(8, brain.num_nodes), steps=5)
    assert y.shape == (8, 5, 3)


def test_auto_initializer_picks_what_the_graph_has() -> None:
    counts = Connectome.from_edges(["A"], ["B"], synapse_count=[4])
    weights = Connectome.from_edges(["A"], ["B"], weight=[1.0])
    bare = Connectome.from_edges(["A"], ["B"])
    assert ConnectomeRNN(counts, weights="trainable").initializer == "normalized_synapse_count"
    assert ConnectomeRNN(weights, weights="trainable").initializer == "weight"
    assert ConnectomeRNN(bare, weights="trainable").initializer == "binary"


def test_flipping_a_sign_changes_the_fingerprint() -> None:
    """sign drives the runtime, so a graph with flipped signs is a different graph."""
    excitatory = Connectome.from_edges([0], [1], weight=[2.0], sign=[1])
    inhibitory = Connectome.from_edges([0], [1], weight=[2.0], sign=[-1])
    assert excitatory.fingerprint() != inhibitory.fingerprint()


def test_object_dtype_attributes_survive_a_round_trip(tmp_path) -> None:
    """Parquet returns object ints as int64, so the reloaded graph failed its own check."""
    brain = Connectome.from_edges([0], [1], synapse_count=np.array([1], dtype=object))
    reloaded = Connectome.load(brain.save(tmp_path / "o.ct"))
    assert reloaded.fingerprint() == brain.fingerprint()


def test_non_numeric_synapse_counts_are_a_validation_error_not_a_typeerror() -> None:
    with pytest.raises(ConnectomeValidationError, match="not a number"):
        Connectome.from_edges([0], [1], synapse_count=np.array(["oops"]))


# ----------------------------------------------------------------------
# lossless conversion and column typing
# ----------------------------------------------------------------------


def test_object_columns_mixing_large_ints_and_floats_are_rejected() -> None:
    """Converting a mixed object column to float64 rounded 2**53+1 away silently."""
    with pytest.raises(ConnectomeValidationError, match="mixes integers and floats"):
        Connectome.from_edges(
            [0, 1], [1, 2], synapse_count=np.array([2**53 + 1, 1.0], dtype=object)
        )


def test_object_columns_of_large_unsigned_ints_survive() -> None:
    """Forcing object ints into int64 raised OverflowError above 2**63."""
    brain = Connectome.from_edges(
        [0], [1], synapse_count=np.array([np.uint64(2**63)], dtype=object)
    )
    assert int(brain.edge_attribute("synapse_count")[0]) == 2**63


def test_object_columns_of_plain_ints_become_int64() -> None:
    brain = Connectome.from_edges([0], [1], synapse_count=np.array([5], dtype=object))
    assert brain.edge_attribute("synapse_count").dtype == np.int64


def test_a_non_numeric_weight_column_is_rejected_at_construction() -> None:
    """A weight of 'oops' was accepted, then blew up inside model construction."""
    with pytest.raises(ConnectomeValidationError, match="not a number"):
        Connectome.from_edges([0], [1], weight=np.array(["oops"]))
    with pytest.raises(ConnectomeValidationError, match="not a number"):
        Connectome.from_edges([0], [1], weight=np.array([None], dtype=object))


def test_the_repr_names_the_initializer_it_resolved() -> None:
    """The docs promise the auto choice is visible; it has to actually be there."""
    model = ConnectomeRNN(Connectome.from_edges([0], [1]), weights="trainable")
    assert "initializer='binary'" in repr(model)


# ----------------------------------------------------------------------
# aggregation of booleans, complex and tab-separated input
# ----------------------------------------------------------------------


def test_boolean_weights_are_summed_across_parallel_edges() -> None:
    """Two edges both marked present are two connections; the merge kept one."""
    brain = Connectome.from_edges([0, 0], [1, 1], weight=np.array([True, True]))
    assert brain.edge_attribute("weight").tolist() == [2]


def test_complex_object_columns_are_rejected() -> None:
    """np.number let complex through, and the cast dropped the imaginary part."""
    with pytest.raises(ConnectomeValidationError, match="complex"):
        Connectome.from_edges([0], [1], weight=np.array([complex(1, 2)], dtype=object))


def test_tsv_files_are_parsed_with_tabs(tmp_path) -> None:
    """.tsv was recognised but then parsed with commas, collapsing to one column."""
    from connectorch.io import read_edge_table

    path = tmp_path / "edges.tsv"
    path.write_text("source\ttarget\tsynapse_count\n0\t1\t5\n")
    assert read_edge_table(path).column_names == ["source", "target", "synapse_count"]
