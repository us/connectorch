"""Turning neurotransmitter predictions into connection polarity.

A connectome says which neuron contacts which. It does not say whether that
contact excites or inhibits. Transmitter predictions get you closer, but not all
the way: polarity is a property of the **receptor** on the postsynaptic side, and
the same transmitter can do opposite things at different receptors.

So nothing here happens by default. You pass a mapping, explicitly, and what you
chose is written into the connectome's provenance alongside the result.

What the mapping buys you is Dale's principle: a neuron releases the same
transmitter at all of its outputs, so its sign belongs to the *neuron*, not to
each connection separately. Applying it here, once, is what lets the runtime hold
that constraint through training.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import ConnectorchError
from ..ir import Connectome
from ..ir.schema import SIGN

__all__ = ["DROSOPHILA_POLARITY", "infer_signs"]

#: A polarity mapping for adult *Drosophila*, for use with `infer_signs`. It is a
#: named constant rather than a default, because it is a modelling choice and a
#: simplification, and you should have to type it.
#:
#: - **acetylcholine → +1.** The principal fast excitatory transmitter of the
#:   insect CNS, acting on nicotinic receptors.
#: - **GABA → -1.** Fast inhibition through Rdl chloride channels.
#: - **glutamate → -1.** In *Drosophila*, unlike in vertebrates, glutamate is
#:   largely inhibitory: it gates the GluClalpha chloride channel. See Liu &
#:   Wilson, *Glutamate is an inhibitory neurotransmitter in the Drosophila
#:   olfactory system*, PNAS 2013 (doi:10.1073/pnas.1220560110), and
#:   Molina-Obando et al., eLife 2019 (doi:10.7554/eLife.49373). Treating it as
#:   excitatory, as a vertebrate intuition would, inverts a large part of the
#:   network.
#: - **histamine → -1.** Photoreceptor output onto lamina neurons through
#:   histamine-gated chloride channels.
#: - **dopamine, octopamine, serotonin → 0.** These act mostly through G
#:   protein-coupled receptors on slower, modulatory timescales. A fast signed
#:   weight is the wrong object for them; ``0`` marks them unknown rather than
#:   pretending otherwise. Modelling them properly is neuromodulation, not a sign.
DROSOPHILA_POLARITY: dict[str, int] = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
}


def infer_signs(
    connectome: Connectome,
    mapping: dict[str, int],
    *,
    column: str = "neurotransmitter",
    strategy: str = "presynaptic_transmitter",
) -> Connectome:
    """Give every connection a sign, from the transmitter of the neuron that makes it.

    Parameters
    ----------
    connectome:
        Must carry a node column naming each neuron's transmitter.
    mapping:
        ``{transmitter: -1 | 0 | +1}``. Required, with no default: see
        :data:`DROSOPHILA_POLARITY` for one you can pass. A transmitter absent
        from the mapping is an error, not a silent zero.
    column:
        Node column holding the transmitter name. Compared case-insensitively.
    strategy:
        Recorded in provenance. Only ``"presynaptic_transmitter"`` exists today.

    Returns
    -------
    Connectome
        A copy carrying a ``sign`` edge column, with the mapping and the coverage
        recorded in ``provenance["sign_inference"]``.

    Notes
    -----
    This is an approximation and is labelled as one wherever it is recorded. The
    postsynaptic receptor is what actually decides polarity, and the transmitter
    labels are themselves predictions from electron microscopy images with their
    own error rate.
    """
    if column not in connectome.node_columns:
        raise ConnectorchError(
            f"no node column {column!r} to read transmitters from; available: "
            f"{list(connectome.node_columns)}. Load the dataset with "
            "neurotransmitters=True, or attach the column yourself."
        )
    invalid = {k: v for k, v in mapping.items() if v not in (-1, 0, 1)}
    if invalid:
        raise ConnectorchError(f"polarity must be -1, 0 or +1; got {invalid}.")

    transmitters = np.asarray(connectome.nodes.column(column).to_pylist(), dtype=object)
    normalised = np.array([str(v or "").strip().lower() for v in transmitters])
    lookup = {str(k).strip().lower(): int(v) for k, v in mapping.items()}

    unknown = sorted({t for t in np.unique(normalised) if t and t not in lookup})
    if unknown:
        raise ConnectorchError(
            f"transmitters {unknown[:8]} appear in {column!r} but not in the "
            "mapping. Add them, so that the polarity of every neuron is a choice "
            "you made rather than a default you did not see."
        )

    node_sign = np.array([lookup.get(t, 0) for t in normalised], dtype=np.int8)
    source_index = connectome.edge_index[0]
    edge_sign = node_sign[source_index]

    signed = int((edge_sign != 0).sum())
    record = dict(connectome.provenance)
    record["sign_inference"] = {
        "strategy": strategy,
        "column": column,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "connections_signed": signed,
        "connections_unknown": int(edge_sign.size - signed),
        "note": (
            "an approximation: polarity is set by the postsynaptic receptor, and "
            "the transmitter labels are themselves predictions"
        ),
    }

    columns = {name: connectome.edge_attribute(name) for name in connectome.edge_columns}
    columns[SIGN] = edge_sign
    node_columns = {
        name: connectome.nodes.column(name).to_numpy(zero_copy_only=False)
        for name in connectome.node_columns
    }
    return Connectome(
        nodes={"node_id": connectome.node_ids, **node_columns},
        edges={
            "source": connectome.node_ids[source_index],
            "target": connectome.node_ids[connectome.edge_index[1]],
            **columns,
        },
        aggregate_parallel_edges=bool(connectome.provenance.get("aggregate_parallel_edges", True)),
        provenance=record,
    )
