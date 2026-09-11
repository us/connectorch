"""neuPrint adapter.

neuPrint serves most published Drosophila connectomes, including MaleCNS. This
adapter is for interactive work on subsets of a dataset: pull the neurons matching
a query and the connections between them.

For a whole connectome, use :func:`connectorch.datasets.malecns` instead. Asking
neuPrint for 25 million connections one REST call at a time is slow for you and
rude to a shared public server; the published bulk files exist for exactly that.

``neuprint-python`` is an optional dependency::

    pip install 'connectorch[neuprint]'

A token is required and is read from the ``NEUPRINT_TOKEN`` environment variable
by default. It is never logged, never written into provenance, and never stored.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..exceptions import ConnectorchError
from ..ir import Connectome

__all__ = ["from_neuprint"]

DEFAULT_SERVER = "https://neuprint.janelia.org"


def _require_neuprint():  # type: ignore[no-untyped-def]
    try:
        import neuprint
    except ImportError:
        raise ConnectorchError(
            "this needs neuprint-python, which is an optional dependency. "
            "Install it with: pip install 'connectorch[neuprint]'"
        ) from None
    return neuprint


def from_neuprint(
    *,
    dataset: str,
    criteria: Any = None,
    server: str = DEFAULT_SERVER,
    token: str | None = None,
    min_synapses: int = 1,
    client: Any = None,
) -> Connectome:
    """Fetch a connectome subset from a neuPrint server.

    Parameters
    ----------
    dataset:
        neuPrint dataset identifier, e.g. ``"male-cns:v1.0"``.
    criteria:
        A ``neuprint.NeuronCriteria`` selecting the neurons to fetch, e.g.
        ``NeuronCriteria(type="DNp01")``. ``None`` fetches every neuron, which on
        a whole-CNS dataset is a great deal of traffic; prefer the bulk loader.
    server:
        neuPrint server URL.
    token:
        Authentication token. Defaults to ``$NEUPRINT_TOKEN``. Get one from your
        neuPrint account page.
    min_synapses:
        Drop connections below this synapse count, server-side where possible.
    client:
        An existing ``neuprint.Client`` to reuse instead of creating one.

    Returns
    -------
    Connectome
        With ``synapse_count`` edges and whatever neuron metadata neuPrint
        returned, renamed so ``where(cell_type=...)`` is valid Python.

    Notes
    -----
    Uses ``fetch_adjacencies(..., omit_rois=True)``, which returns one row per
    body-to-body connection rather than one row per connection per brain region.
    Without it a single pair can come back dozens of times.
    """
    neuprint = _require_neuprint()

    token = token or os.environ.get("NEUPRINT_TOKEN")
    if client is None and not token:
        raise ConnectorchError(
            "no neuPrint token. Set the NEUPRINT_TOKEN environment variable, or "
            "pass token=..., or pass an existing client=. Tokens are available "
            f"from your account page at {server}."
        )
    if client is None:
        client = neuprint.Client(server, dataset=dataset, token=token)

    neurons, connections = neuprint.fetch_adjacencies(
        sources=criteria,
        targets=criteria,
        min_total_weight=min_synapses,
        omit_rois=True,
        client=client,
    )
    if len(connections) == 0:
        raise ConnectorchError(
            "neuPrint returned no connections for that query. Widen the criteria "
            "or lower min_synapses."
        )

    node_columns: dict[str, Any] = {"node_id": neurons["bodyId"].to_numpy()}
    for original, renamed in (("type", "cell_type"), ("instance", "instance")):
        if original in neurons.columns:
            node_columns[renamed] = neurons[original].fillna("").astype(str).to_numpy()

    return Connectome(
        nodes=node_columns,
        edges={
            "source": connections["bodyId_pre"].to_numpy(),
            "target": connections["bodyId_post"].to_numpy(),
            "synapse_count": connections["weight"].to_numpy().astype(np.int64),
        },
        provenance={
            "dataset": dataset,
            "source": f"neuprint:{server}",
            "criteria": repr(criteria) if criteria is not None else None,
            "filters": {"min_synapses": int(min_synapses)},
        },
    )
