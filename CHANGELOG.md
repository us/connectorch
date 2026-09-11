# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/). Anything that changes the numerical
semantics of the runtime is called out explicitly, because a silent change there
would invalidate other people's results.

## [Unreleased]

### Added

- `Connectome` intermediate representation: deterministic sorted indexing,
  validation with messages that name the offending row, parallel-edge
  aggregation, provenance tracking, structural fingerprint, and a `.ct`
  directory format of parquet plus JSON with no pickle anywhere.
- `ConnectomeRNN`: a rate-based recurrent `torch.nn.Module` over a fixed
  connectome topology, with trainable or fixed edge weights.
- Three propagation backends: `scatter` (the training backend), `sparse_mm`
  (CSR, the fast forward path), and `dense` (a correctness oracle that refuses
  graphs above a gibibyte).
- `backend="auto"`, which chooses `scatter` for trainable weights and
  `sparse_mm` otherwise, from measurements in `benchmarks/`.
- MaleCNS v1.0 loader with resumable caching, and a bundled 100-neuron CC-BY
  sample that loads offline.
- CSV, Parquet, Arrow/Feather and NetworkX adapters; a `connectorch` CLI.
- Topology-matched controls (`degree_preserving_rewire`, `random_topology`,
  `shuffle_edge_weights`), because a connectome result without them is not a
  result.
- Resumable, verified downloads: a resume whose `Content-Range` disagrees with
  the local file restarts rather than appending the wrong bytes.

- `ConnectomeRNN.activation_bytes(batch, steps)` and a warning before a backward
  pass whose stored per-edge activations would exceed half the device's free
  memory. The dense `[N, N]` refusal was already loud; the `O(E x B x T)` cost of
  the path the library actually recommends was not defended at all.

- `BiologicalWeights`, a parameterisation that keeps the measured data in the
  model after training: `weight = sign * synapse_count_prior * bounded_gain`,
  with the gain optionally shared across all connections between a pair of cell
  types. Trained on the bundled sample, free weights end between 0.00x and 4.32x
  of the count they started at and leave 93 of 100 neurons both exciting and
  inhibiting their targets; this keeps them within 0.66x to 1.53x with none.
- `malecns(neurotransmitters=True)` and `transforms.infer_signs`, which turn the
  dataset's transmitter predictions into connection polarity under a mapping the
  caller supplies. `DROSOPHILA_POLARITY` is provided but never applied by
  default.
- `experiments/01_inductive_bias.py`, comparing the connectome against
  topology-matched controls and a dense RNN at a matched parameter budget.

### Changed semantics

Nothing here has been released yet, so no stored data is affected, but these
would be breaking changes after v0.1:

- `fingerprint()` now tags identifiers by type and hashes integer edge columns as
  integers. Previously a graph of empty strings hashed like a graph of zeros, and
  synapse counts of `2**53` and `2**53+1` hashed identically. The numerical
  behaviour of the runtime is unchanged: the golden regression fixture's outputs
  and weights are bit-for-bit identical across this change.
- Edge weights are produced by an `EdgeWeights` module rather than being a bare
  parameter. `model.edge_weight` still reads as before and the strategy names
  still work; a checkpoint from an earlier commit will not load, and none has
  been released.
- The `sign` edge column is now applied by the runtime. It was previously
  validated and then ignored, which turned an edge marked inhibitory into an
  excitatory one.

### Fixed

Defects that silently produced wrong numbers, each with a regression test in
`tests/regression/`:

- parallel-edge aggregation wrapped narrow integers (two int8 counts of 100 gave
  -56) and lost precision on large ones;
- `load_state_dict` left a backend propagating along the previous topology;
- string node ids hashed their own memory addresses, so a graph failed its own
  integrity check after a save and reload;
- mixed integer and string ids merged into one node;
- `uint64` edge endpoints mapped to the wrong neuron above 2**53;
- a resumed download whose `Content-Range` disagreed with the local file was
  appended anyway, and an incomplete file could be cached as complete;
- `.tsv` files were parsed with commas.
