"""Every validation path, and the quality of the message it produces.

A validation error is the main way this library talks to someone whose data is not
what they thought it was. These tests check that each one says which row is wrong
and what to do, not just that it fired.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectorch import Connectome, ConnectomeValidationError, ConnectorchError
from connectorch.io import from_neuprint


def message_for(**kwargs) -> str:
    with pytest.raises(ConnectomeValidationError) as excinfo:
        Connectome.from_edges(**kwargs)
    return str(excinfo.value)


def test_empty_connectome() -> None:
    assert "no nodes" in message_for(source=[], target=[])


def test_length_mismatch_between_endpoints() -> None:
    text = message_for(source=[0, 1], target=[1])
    assert "same length" in text
    assert "2" in text and "1" in text


def test_edge_column_length_mismatch() -> None:
    text = message_for(source=[0, 1], target=[1, 2], synapse_count=[1])
    assert "1 values but there are 2 edges" in text


def test_nan_endpoints_name_the_rows() -> None:
    text = message_for(source=np.array([0.0, np.nan]), target=np.array([1.0, 1.0]))
    assert "NaN" in text or "precision" in text


def test_non_integral_synapse_count() -> None:
    assert "whole number" in message_for(source=[0], target=[1], synapse_count=[1.5])


def test_integral_float_synapse_counts_are_allowed() -> None:
    brain = Connectome.from_edges(source=[0], target=[1], synapse_count=[3.0])
    assert brain.edge_attribute("synapse_count").tolist() == [3.0]


def test_invalid_sign_values() -> None:
    text = message_for(source=[0], target=[1], sign=[7])
    assert "-1, 0 (unknown) or +1" in text
    assert "7" in text


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_valid_sign_values_pass(sign: int) -> None:
    assert Connectome.from_edges(source=[0], target=[1], sign=[sign]).num_edges == 1


def test_two_dimensional_input() -> None:
    with pytest.raises(ConnectomeValidationError, match="1-D"):
        Connectome.from_edges(source=np.zeros((2, 2)), target=np.zeros((2, 2)))


def test_an_unusable_type_says_so() -> None:
    with pytest.raises(ConnectomeValidationError, match="cannot interpret"):
        Connectome.from_edges(source=object(), target=[1])


def test_a_node_table_without_ids() -> None:
    with pytest.raises(ConnectomeValidationError, match="no 'node_id' column"):
        Connectome(nodes={"label": ["a"]}, edges={"source": [0], "target": [1]})


def test_an_edge_table_without_endpoints() -> None:
    with pytest.raises(ConnectomeValidationError, match="'source' and 'target'"):
        Connectome(edges={"weight": [1.0]})


def test_a_table_of_the_wrong_type() -> None:
    with pytest.raises(ConnectomeValidationError, match="pyarrow Table"):
        Connectome(edges=42)


def test_many_bad_rows_are_summarised_not_dumped() -> None:
    """An error listing ten thousand rows is not an error message."""
    text = message_for(source=list(range(20)), target=[99] * 20, nodes={"node_id": list(range(20))})
    assert "20 edge(s) have unknown endpoints" in text
    assert text.count("\n") < 6


def test_neuprint_without_a_token_says_where_to_get_one(monkeypatch) -> None:
    pytest.importorskip("neuprint", reason="optional dependency not installed")
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    with pytest.raises(ConnectorchError, match="account page"):
        from_neuprint(dataset="male-cns:v1.0")


def test_neuprint_without_the_package_says_how_to_install_it(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "neuprint":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ConnectorchError, match=r"connectorch\[neuprint\]"):
        from_neuprint(dataset="male-cns:v1.0")
