"""Tests for the command line and the dataset registry.

The CLI is the first thing someone runs to find out whether the library can read
their file, so its exit codes have to mean what they say.
"""

from __future__ import annotations

import json

import pytest

from connectorch import Connectome, ConnectorchError
from connectorch.cli.main import main
from connectorch.datasets import registry


def test_info_describes_a_saved_connectome(tmp_path, capsys) -> None:
    brain = Connectome(
        nodes={"node_id": [1, 2, 3], "cell_class": ["a", "b", "c"]},
        edges={"source": [1, 2], "target": [2, 3], "synapse_count": [4, 5]},
        provenance={"dataset": "test:v1", "license": "CC-BY"},
    )
    path = brain.save(tmp_path / "b.ct")
    assert main(["info", str(path)]) == 0

    out = capsys.readouterr().out
    assert "test:v1" in out
    assert "Nodes:       3" in out
    assert "CC-BY" in out
    assert brain.fingerprint() in out


def test_info_on_something_that_is_not_a_connectome(tmp_path, capsys) -> None:
    (tmp_path / "empty").mkdir()
    assert main(["info", str(tmp_path / "empty")]) == 1
    assert "not a ConnecTorch dataset" in capsys.readouterr().err


def test_validate_accepts_a_good_table_and_reports_merges(tmp_path, capsys) -> None:
    path = tmp_path / "edges.csv"
    path.write_text("source,target,synapse_count\n0,1,5\n0,1,3\n1,2,2\n")
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 edges" in out
    assert "merged 1" in out


def test_validate_rejects_an_unknown_endpoint(tmp_path, capsys) -> None:
    path = tmp_path / "edges.csv"
    path.write_text("source,target\n0,1\n")
    assert main(["validate", str(path), "--source-column", "nope"]) == 1
    assert "available" in capsys.readouterr().err


def test_datasets_list_and_info(capsys) -> None:
    assert main(["datasets", "list"]) == 0
    assert "male-cns" in capsys.readouterr().out

    assert main(["datasets", "info", "male-cns"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["license"] == "CC-BY-4.0"
    assert record["neuprint_dataset"] == "male-cns:v1.0"
    assert "loader" not in record, "the registry must not leak callables into JSON"


def test_datasets_info_on_an_unknown_name(capsys) -> None:
    assert main(["datasets", "info", "nope"]) == 1
    assert "available" in capsys.readouterr().err


def test_downloading_something_that_is_bundled_is_refused(capsys) -> None:
    assert main(["datasets", "download", "male-cns-sample"]) == 1
    assert "nothing to download" in capsys.readouterr().err


def test_registry_loads_the_bundled_sample() -> None:
    brain = registry.get("male-cns-sample")
    assert brain.num_nodes == 100
    assert brain.provenance["license"] == "CC-BY-4.0"


def test_registry_rejects_an_unknown_dataset() -> None:
    with pytest.raises(ConnectorchError, match="available"):
        registry.get("nope")
    with pytest.raises(ConnectorchError, match="available"):
        registry.info("nope")


def test_registry_lists_every_dataset_with_a_license() -> None:
    """Attribution is a legal obligation, not a nice-to-have."""
    for name in registry.list_datasets():
        record = registry.info(name)
        assert record["license"]
        assert record["citation"]
