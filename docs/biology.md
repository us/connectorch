# Keeping the biology in the loop

The default runtime gives every connection its own free parameter. That preserves
the *topology* and nothing else, and it is worth being precise about what gets
lost.

Train a connectome that way and measure how far each weight moved from the
synapse count it started at:

| parameterisation | after training, `\|w\| / prior` | neurons that both excite and inhibit |
|---|---|---|
| `weights="trainable"` | 0.00 – 4.32 | 93 of 100 |
| `BiologicalWeights(..., dale=True)` | 0.66 – 1.53 | 0 of 100 |

The first row is a network whose wiring diagram came from a fly and whose numbers
came from nowhere: connections measured at 200 synapses driven to zero,
connections measured at 2 grown fourfold, and most neurons excitatory to some
targets and inhibitory to others, which no neuron is.

## What a connection is worth

```text
weight = sign * prior * gain
```

- **prior** is the measured synapse count, through a named strategy.
- **sign** is the presynaptic neuron's polarity, from its transmitter.
- **gain** is the only thing gradient descent touches, and it is bounded.

```python
import connectorch as ct
from connectorch.transforms import DROSOPHILA_POLARITY, infer_signs

fly = ct.datasets.malecns(download=True, neurotransmitters=True)
fly = infer_signs(fly, DROSOPHILA_POLARITY)

core = ct.nn.ConnectomeRNN(
    fly,
    weights=ct.nn.BiologicalWeights(
        fly,
        gain_bounds=(0.5, 2.0),   # learning may halve or double what was measured
        share_by="cell_type",     # one gain per pair of cell types
        dale=True,                # a neuron's polarity is fixed
    ),
)
```

The gain starts at exactly 1.0, so the model *begins* as the measured connectome
and departs from it only as far as the data pushes it, and never further than the
bound. Passing `gain_bounds=None` gives that up; it is available and it is opt-in.

## Cell-type sharing

A nervous system is not wired by hand. Neurons of a type follow a rule, and
25 million independent parameters is not that rule — it is enough freedom to
memorise the training set through a graph that happens to be biological.

`share_by="cell_type"` gives one gain to every connection running from cell type
A to cell type B. This is the mechanism FlyVis used ([Lappalainen et al. 2024](references.md#prior-work)),
generalised here to any node column: share by `cell_class` for something coarser,
by `side` to let the two hemispheres differ, by anything the dataset annotates.

## Polarity is a modelling choice, and you have to make it

Nothing infers a sign for you. `infer_signs` takes a mapping and refuses to run
without one, because polarity is set by the **postsynaptic receptor**, not by the
transmitter, and a mapping is an approximation with consequences.

[`DROSOPHILA_POLARITY`](references.md) is provided for the fly, and one entry in
it deserves attention:

```text
"glutamate": -1
```

In *Drosophila*, unlike in vertebrates, glutamate is largely **inhibitory**: it
gates the GluClalpha chloride channel (Liu & Wilson, PNAS 2013). A vertebrate
intuition applied here inverts a large fraction of the network.

Modulatory transmitters — dopamine, octopamine, serotonin — map to `0`, meaning
unknown. They act through G protein-coupled receptors on slower timescales, and a
fast signed weight is the wrong object for them. Modelling them properly is
neuromodulation, which this runtime does not yet do.

The transmitter labels are themselves predictions from electron microscopy images
([Eckstein et al. 2024](references.md#the-data)) with their own error rate. A sign
derived from them is a prediction on top of a prediction, and `provenance
["sign_inference"]` records the mapping, the coverage and that caveat.

## What this still is not

Fixing the weights is one step, not the whole distance. The runtime is still a
rate model: no spikes, no membrane conductances, no dendritic compartments, no gap
junctions, no neuromodulation, no plasticity, and a "step" that corresponds to no particular
number of milliseconds. Fixed heterogeneous synaptic delays (`delay_by`) and
per-type leak (`leak_by`) are supported; learned time constants are not. The input
and output neurons are whichever ones you select, with none of the retinotopy a
real sensory map has.

What changed is narrower and worth stating exactly: **the measured synapse counts
and the measured transmitters are now still present in the model after training,
instead of having been optimised into something else.**
