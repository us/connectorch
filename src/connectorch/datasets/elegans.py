"""The *C. elegans* whole-animal connectome (Cook et al. 2019).

The second species in the library, and the proof that nothing in the
pipeline is fly-specific: 302 neurons, a few thousand connections, the
whole graph in kilobytes. It loads on a laptop in milliseconds, which
makes it the right connectome for fast iteration and for tests that
need real biology without a 508 MB download.

Data
----
Machine-readable CSVs from the Netzschleuder mirror of the Cook et al.
2019 reconstructions (upstream: https://wormwiring.org/pages/adjacency.html)::

    https://networks.skewed.de/net/celegans_2019/files/

The ``synapse`` tables carry EM-scored synapse counts per cell pair, with
no extrapolated connections. Node properties carry the cell name and its
broad class (sensory, inter, motor).

License
-------
ConnecTorch's MIT license covers the code only. The data stays under its
own terms; see ``THIRD_PARTY_DATA.md``.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ConnectorchError
from ..ir import Connectome
from ._cache import default_cache_dir, ensure_file

__all__ = ["elegans", "TABLES", "DATASET_INFO", "NEURON_CLASSES"]

BASE_URL = "https://networks.skewed.de/net/celegans_2019/files"

#: Table name -> (zip filename, verified byte size).
TABLES: dict[str, tuple[str, int]] = {
    "hermaphrodite_chemical_synapse": (
        "hermaphrodite_chemical_synapse.csv.zip",
        22318,
    ),
    "male_chemical_synapse": ("male_chemical_synapse.csv.zip", 22054),
}

DATASET_INFO: dict[str, Any] = {
    "name": "c-elegans",
    "version": "cook-2019",
    "description": (
        "Whole-animal C. elegans chemical connectome: 302 neurons, "
        "EM-scored synapse counts, no extrapolated connections"
    ),
    "license": "data: Cook et al. 2019, see THIRD_PARTY_DATA.md",
    "citation": (
        "Cook, S. J. et al. Whole-animal connectomes of both Caenorhabditis "
        "elegans sexes. Nature 571, 63-71 (2019). "
        "https://doi.org/10.1038/s41586-019-1352-7"
    ),
    "homepage": "https://wormwiring.org/pages/adjacency.html",
}

#: Node classes that are neurons. The pharyngeal nervous system is a
#: separate, largely autonomous circuit and is excluded by default.
NEURON_CLASSES = frozenset(
    {"SENSORY NEURONS", "INTERNEURONS", "MOTOR NEURONS", "SEX-SPECIFIC CELLS"}
)


def elegans(
    table: str = "hermaphrodite_chemical_synapse",
    *,
    include_pharynx: bool = False,
    min_synapses: int = 1,
    download: bool | None = None,
    cache_dir: Path | str | None = None,
) -> Connectome:
    """Load the *C. elegans* chemical connectome as a :class:`Connectome`.

    Parameters
    ----------
    table:
        Which reconstruction table: hermaphrodite or male chemical synapses.
    include_pharynx:
        Keep the 57 pharyngeal cells. By default only neurons are kept;
        pharyngeal edges are dropped with them.
    min_synapses:
        Drop pairs below this many scored synapses.
    download:
        ``True`` to fetch the ~22 KB archive, ``False`` to refuse,
        ``None`` (default) to raise a message naming the size and URL first.
    cache_dir:
        Override for the dataset cache.

    Examples
    --------
    >>> worm = elegans(download=True)                       # doctest: +SKIP
    >>> worm.where(cell_class="MOTOR NEURONS")[:3]          # doctest: +SKIP
    """
    if table not in TABLES:
        raise ConnectorchError(f"unknown elegans table {table!r}; available: {sorted(TABLES)}.")
    if min_synapses < 1:
        raise ConnectorchError(f"min_synapses must be at least 1, got {min_synapses}.")
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache = cache / "c-elegans-cook-2019"

    filename, size = TABLES[table]
    archive = ensure_file(
        f"{BASE_URL}/{filename}", cache_dir=cache, expected_size=size, download=download
    )
    return _assemble(archive, table, include_pharynx, min_synapses)


def _assemble(archive: Path, table: str, include_pharynx: bool, min_synapses: int) -> Connectome:
    """Build a :class:`Connectome` from a downloaded Netzschleuder archive.

    Split out so the parsing and filtering is testable offline against a
    synthetic archive; only :func:`elegans` touches the network.
    """
    filename = TABLES[table][0]
    names, classes = _read_nodes(archive)
    keep_class = np.array(
        [c in NEURON_CLASSES or (include_pharynx and c == "PHARYNX") for c in classes]
    )
    source, target, counts = _read_edges(archive)

    if min_synapses > 1:
        keep = counts >= min_synapses
        source, target, counts = source[keep], target[keep], counts[keep]
    keep_edge = keep_class[source] & keep_class[target]
    source, target, counts = source[keep_edge], target[keep_edge], counts[keep_edge]

    node_ids = np.flatnonzero(keep_class)
    provenance = {
        "dataset": "c-elegans:cook-2019",
        "table": table,
        "source": f"{BASE_URL}/{filename}",
        "license": DATASET_INFO["license"],
        "citation": DATASET_INFO["citation"],
        "filters": {
            "include_pharynx": include_pharynx,
            "min_synapses": int(min_synapses),
        },
    }
    return Connectome(
        nodes={
            "node_id": node_ids,
            "cell_name": np.array(names)[node_ids],
            "cell_class": np.array(classes)[node_ids],
        },
        edges={"source": source, "target": target, "synapse_count": counts},
        provenance=provenance,
    )


def _read_nodes(archive: Path) -> tuple[list[str], list[str]]:
    """Cell names and classes, in node-index order."""
    with zipfile.ZipFile(archive) as zf, zf.open("nodes.csv") as fh:
        lines = [
            line
            for line in (row.decode() for row in fh)
            if line.strip() and not line.startswith("#")
        ]
    names, classes = [], []
    for row in csv.reader(lines):
        names.append(row[3])
        classes.append(row[1])
    return names, classes


def _read_edges(archive: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source/target indices and scored synapse counts."""
    with zipfile.ZipFile(archive) as zf, zf.open("edges.csv") as fh:
        lines = [
            line
            for line in (row.decode() for row in fh)
            if line.strip() and not line.startswith("#")
        ]
    source, target, counts = [], [], []
    for row in csv.reader(lines):
        source.append(int(row[0]))
        target.append(int(row[1]))
        counts.append(int(row[2]))
    return (
        np.array(source, dtype=np.int64),
        np.array(target, dtype=np.int64),
        np.array(counts, dtype=np.int64),
    )
