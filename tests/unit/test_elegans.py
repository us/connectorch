"""The C. elegans loader, tested offline against a synthetic archive.

Only ``elegans()`` touches the network; ``_assemble`` parses and filters,
so every behaviour below runs without downloading anything.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from connectorch import ConnectorchError
from connectorch.datasets.elegans import _assemble, elegans

NODES = [
    ("SENSORY NEURONS", "ADAL"),
    ("INTERNEURONS", "AIBL"),
    ("MOTOR NEURONS", "DB1"),
    ("PHARYNX", "I1L"),
    ("SEX-SPECIFIC CELLS", "CEMDL"),
]

EDGES = [
    (0, 1, 4),  # sensory -> inter, strong
    (1, 2, 2),  # inter -> motor
    (0, 3, 7),  # sensory -> pharynx (dropped unless included)
    (4, 1, 1),  # sex-specific -> inter, weak
    (3, 3, 3),  # pharynx self-loop
]


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.csv.zip"
    with zipfile.ZipFile(path, "w") as zf:
        node_lines = ["# index, node_type, node_subtype, name, _pos"]
        for i, (kind, name) in enumerate(NODES):
            node_lines.append(f'{i},{kind},,{name},"array([0.0, 0.0])"')
        zf.writestr("nodes.csv", "\n".join(node_lines) + "\n")
        edge_lines = ["# source, target, synapses"]
        for s, t, w in EDGES:
            edge_lines.append(f"{s},{t},{w}")
        zf.writestr("edges.csv", "\n".join(edge_lines) + "\n")
    return path


def test_pharynx_excluded_by_default(archive: Path) -> None:
    worm = _assemble(archive, "hermaphrodite_chemical_synapse", False, 1)
    assert worm.num_nodes == 4
    names = worm.nodes.column("cell_name").to_pylist()
    assert "I1L" not in names
    assert worm.num_edges == 3  # sensory->pharynx and pharynx loop dropped


def test_include_pharynx_keeps_everything(archive: Path) -> None:
    worm = _assemble(archive, "hermaphrodite_chemical_synapse", True, 1)
    assert worm.num_nodes == 5
    assert worm.num_edges == 5


def test_min_synapses_filters_weak_pairs(archive: Path) -> None:
    worm = _assemble(archive, "hermaphrodite_chemical_synapse", False, 2)
    assert worm.num_edges == 2  # weights 4 and 2 survive
    assert sorted(worm.edge_attribute("synapse_count").tolist()) == [2, 4]


def test_annotations_survive_on_the_kept_nodes(archive: Path) -> None:
    worm = _assemble(archive, "hermaphrodite_chemical_synapse", False, 1)
    by_id = dict(
        zip(
            worm.nodes.column("node_id").to_pylist(),
            zip(
                worm.nodes.column("cell_name").to_pylist(),
                worm.nodes.column("cell_class").to_pylist(),
                strict=True,
            ),
            strict=True,
        )
    )
    assert by_id[0] == ("ADAL", "SENSORY NEURONS")
    assert by_id[4] == ("CEMDL", "SEX-SPECIFIC CELLS")


def test_provenance_names_the_source_table(archive: Path) -> None:
    worm = _assemble(archive, "male_chemical_synapse", False, 1)
    assert worm.provenance["table"] == "male_chemical_synapse"
    assert "networks.skewed.de" in worm.provenance["source"]
    assert "Cook" in worm.provenance["citation"]


def test_unknown_table_and_bad_threshold_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConnectorchError, match="unknown elegans table"):
        elegans(table="nope", download=False)
    with pytest.raises(ConnectorchError, match="min_synapses"):
        elegans(min_synapses=0, download=False)


def test_where_selects_by_cell_class(archive: Path) -> None:
    worm = _assemble(archive, "hermaphrodite_chemical_synapse", False, 1)
    assert len(worm.where(cell_class="MOTOR NEURONS")) == 1
