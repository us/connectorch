# Scientific caveats

Read this before claiming anything about biology.

## What a connectome is, and is not

A connectome is a **structural wiring diagram**: an estimate of which neuron
contacts which, and how many synapses were reconstructed between them. Compiling
one into a neural network does not recreate the animal's brain.

Specifically:

- **Synapse count is not synaptic efficacy.** The number of reconstructed contacts
  between two neurons correlates with influence, but receptor identity, release
  probability, vesicle pools and short-term plasticity all matter and none of them
  is in the data.
- **Neurotransmitter identity does not fix a sign.** Acetylcholine is usually
  excitatory in the fly and GABA usually inhibitory, but polarity is a property of
  the *receptor* on the postsynaptic side, not of the transmitter. The MaleCNS
  transmitter column is itself a prediction from electron microscopy images
  ([Eckstein et al. 2024](references.md#the-data)) with its own confidence scores,
  so a sign derived from it is a prediction on top of a prediction. ConnecTorch
  will use an explicit `sign` column if you provide one, and will never infer one
  for you from a transmitter label without you asking.
- **Cellular dynamics are absent.** The default runtime is a rate model. There are
  no spikes, no membrane conductances, no dendritic compartments, no gap
  junctions, no neuromodulation.
- **The reconstruction is incomplete and noisy.** Segmentation errors, missed
  synapses and unproofread bodies are real. Published datasets ship confidence
  thresholds and "traced only" variants precisely because of this: MaleCNS's
  default tables are already filtered at confidence 0.5.
- **A trained model is not the animal.** Optimising these weights for a machine
  learning objective produces an artificial model *constrained by* biology. It
  does not tell you what the fly computes.

## What you can hold onto

Two of these are addressable with what the dataset already gives you, and
[Keeping the biology](biology.md) is how: the measured synapse counts can stay in
the model as a bounded prior instead of being optimised away, and each neuron's
polarity can be held fixed so it cannot excite some targets while inhibiting
others. Neither makes the model a fly. Both stop it drifting further from one
than it needs to.

## What ConnecTorch does about it

Every weight strategy is a named, documented **numerical** choice, never a
biological claim. The library will not silently invent an interpretation of your
data. Filters, aggregation and dataset versions are recorded in `provenance`, and
`fingerprint()` identifies the exact graph a result came from.

## Controls are not optional

If you want to claim that biological wiring is a useful inductive bias, the
comparison has to be against wiring matched in every way except being biological.
Otherwise the result may only be showing that sparsity helps, or that a particular
degree distribution helps.

```python
from connectorch.transforms import (
    degree_preserving_rewire,   # same in- and out-degrees, different wiring
    random_topology,            # same node and edge counts, nothing else
    shuffle_edge_weights,       # same wiring, synapse counts permuted
)
```

At minimum, report the real connectome, a degree-preserving rewiring, and a random
graph of the same size, at a matched trainable-parameter budget, over at least
five random seeds. `connectorch.nn.count_parameters` splits the budget into the
connectome and everything else, because the encoder and decoder are part of it.

A single-seed difference between a connectome and one control is not a result.

## Sources

Every factual claim on this page is referenced on the [References](references.md)
page, including the connectome itself, the transmitter predictions, and the prior
work this library builds on.
