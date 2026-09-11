"""The Janelia/Google/Cambridge Male CNS Drosophila connectome, v1.0.

The flagship dataset: a whole male Drosophila central nervous system reconstructed
at synapse resolution. ConnecTorch reads the published flat-connectome tables
directly rather than making millions of API calls.

Data
----
Files live in a public Google Cloud Storage bucket and need no credentials::

    https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

The connectivity table has three columns: ``body_pre``, ``body_post`` and
``weight`` (a synapse count). The annotation table is keyed by ``bodyId`` and
carries cell type, class and side.

License
-------
MaleCNS v1.0 is CC-BY. ConnecTorch's MIT license covers the code only; it does not
relicense the data. See ``THIRD_PARTY_DATA.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ConnectorchError
from ..io.tabular import read_edge_table
from ..ir import Connectome
from ._cache import default_cache_dir, ensure_file

__all__ = ["malecns", "malecns_sample", "VARIANTS", "DATASET_INFO"]

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

#: Connectivity table variants, with their verified byte sizes.
VARIANTS: dict[str, tuple[str, int]] = {
    "traced-only": (
        "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
        508_025_642,
    ),
    "significant-only": (
        "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        502_169_298,
    ),
    "full": ("connectome-weights-male-cns-v1.0-minconf-0.5.feather", 1_051_241_946),
}

ANNOTATIONS = ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14_483_314)
NEUROTRANSMITTERS = ("body-neurotransmitters-male-cns-v1.0.feather", 43_282_834)

#: Annotation columns kept by default, renamed so ``where(cell_class=...)`` is
#: valid Python. The source file has 36 columns; carrying all of them for every
#: body wastes memory a training run has better uses for.
DEFAULT_ANNOTATIONS: dict[str, str] = {
    "type": "cell_type",
    "class": "cell_class",
    "subclass": "subclass",
    "superclass": "superclass",
    "somaSide": "side",
    "instance": "instance",
    "status": "status",
}

DATASET_INFO: dict[str, Any] = {
    "name": "male-cns",
    "version": "v1.0",
    "description": "Male Drosophila central nervous system, synapse-resolution connectome",
    "license": "CC-BY-4.0",
    "citation": (
        "Janelia FlyEM, Google Research and the University of Cambridge, "
        "Male CNS connectome v1.0. https://neuprint.janelia.org (male-cns:v1.0)"
    ),
    "homepage": "https://neuprint.janelia.org",
    "neuprint_dataset": "male-cns:v1.0",
    "variants": {name: {"file": f, "bytes": n} for name, (f, n) in VARIANTS.items()},
}


def malecns(
    *,
    variant: str = "traced-only",
    cache_dir: str | Path | None = None,
    min_synapses: int = 1,
    annotations: bool = True,
    annotation_columns: dict[str, str] | None = None,
    download: bool | None = None,
) -> Connectome:
    """Load MaleCNS v1.0 as a :class:`~connectorch.ir.Connectome`.

    Parameters
    ----------
    variant:
        Which published connectivity table to use. ``"traced-only"`` (508 MiB,
        the default) keeps connections between traced bodies;
        ``"significant-only"`` (502 MiB) keeps statistically significant ones;
        ``"full"`` (1.0 GiB) keeps everything at confidence >= 0.5.
    cache_dir:
        Where downloads are kept. Defaults to ``$CONNECTORCH_CACHE`` or
        ``~/.cache/connectorch``.
    min_synapses:
        Drop connections below this synapse count. Recorded in provenance.
    annotations:
        Attach cell type, class and side metadata from the annotation table.
    annotation_columns:
        ``{source column: IR column}`` overriding :data:`DEFAULT_ANNOTATIONS`.
        The full 36-column list is in this module's docstring reference.
    download:
        ``True`` to fetch missing files, ``False`` to refuse, ``None`` (default)
        to raise a message naming the size and URL first. Nothing here downloads
        half a gigabyte without being told to.

    Returns
    -------
    Connectome
        Provenance records the dataset, variant, source URL, license and filters.

    Examples
    --------
    >>> brain = malecns(download=True)                      # doctest: +SKIP
    >>> brain.where(cell_class="descending neuron")[:3]     # doctest: +SKIP
    """
    if variant not in VARIANTS:
        raise ConnectorchError(
            f"unknown MaleCNS variant {variant!r}; available: {sorted(VARIANTS)}."
        )
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache = cache / "male-cns-v1.0"

    filename, size = VARIANTS[variant]
    edge_path = ensure_file(
        f"{BASE_URL}/{filename}", cache_dir=cache, expected_size=size, download=download
    )

    table = read_edge_table(edge_path, columns=["body_pre", "body_post", "weight"])
    source = table.column("body_pre").to_numpy()
    target = table.column("body_post").to_numpy()
    counts = table.column("weight").to_numpy()
    del table

    if min_synapses > 1:
        keep = counts >= min_synapses
        source, target, counts = source[keep], target[keep], counts[keep]

    node_columns: dict[str, Any] | None = None
    if annotations:
        node_ids = np.unique(np.concatenate([source, target]))
        annotation_path = ensure_file(
            f"{BASE_URL}/{ANNOTATIONS[0]}",
            cache_dir=cache,
            expected_size=ANNOTATIONS[1],
            download=download,
        )
        node_columns = _align_annotations(
            annotation_path, node_ids, annotation_columns or DEFAULT_ANNOTATIONS
        )

    provenance = {
        "dataset": "male-cns:v1.0",
        "variant": variant,
        "source": f"{BASE_URL}/{filename}",
        "license": DATASET_INFO["license"],
        "citation": DATASET_INFO["citation"],
        "filters": {"min_confidence": 0.5, "min_synapses": int(min_synapses)},
    }
    return Connectome(
        nodes=node_columns,
        edges={"source": source, "target": target, "synapse_count": counts},
        provenance=provenance,
    )


def _align_annotations(
    path: Path, node_ids: np.ndarray, columns: dict[str, str]
) -> dict[str, np.ndarray]:
    """Left-join the annotation table onto the node set that the edges define.

    Bodies present in the connectivity table but absent from the annotations get
    an empty string rather than being dropped: an unannotated neuron is still a
    neuron, and silently losing it would change the graph.
    """
    table = read_edge_table(path, columns=["bodyId", *columns])
    body = table.column("bodyId").to_numpy()

    position = np.searchsorted(node_ids, body)
    in_range = position < node_ids.size
    matched = np.zeros(body.size, dtype=bool)
    matched[in_range] = node_ids[position[in_range]] == body[in_range]

    aligned: dict[str, np.ndarray] = {"node_id": node_ids}
    for original, renamed in columns.items():
        values = table.column(original).to_pylist()
        out = np.full(node_ids.size, "", dtype=object)
        out[position[matched]] = [
            "" if values[i] is None else str(values[i]) for i in np.flatnonzero(matched)
        ]
        aligned[renamed] = out.astype(str)
    return aligned


def malecns_sample() -> Connectome:
    """Load the small MaleCNS subset bundled with ConnecTorch. No network access.

    A deterministic subgraph of the real dataset, small enough to ship in the
    wheel and to run in continuous integration. It carries the same CC-BY
    attribution as the full dataset; see ``THIRD_PARTY_DATA.md``.
    """
    path = Path(__file__).parent / "data" / "malecns_sample.ct"
    if not path.exists():
        raise ConnectorchError(
            f"the bundled MaleCNS sample is missing from {path}. "
            "Reinstall connectorch, or build it with "
            "`python -m connectorch.datasets.build_sample`."
        )
    return Connectome.load(path)
