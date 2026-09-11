# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/). Anything that changes the numerical
semantics of the runtime is called out explicitly, because a silent change there
would invalidate other people's results.

## 0.1.0 (2026-09-11)


### Features

* add a content hash beside the structural fingerprint ([4b3d77e](https://github.com/us/connectorch/commit/4b3d77ef466e6f0e72dabe769bb37efa9b7a7b0e))
* compile biological connectomes into trainable PyTorch networks ([0e3ec5e](https://github.com/us/connectorch/commit/0e3ec5e5acdce74b66ec29f9d8aed9ab5ce1acc7))
* keep the measured synapse counts and polarity through training ([51fa277](https://github.com/us/connectorch/commit/51fa27728fa54f5b564bece1c713e8ba5f55b264))
* load neurotransmitter predictions and turn them into connection polarity ([db945ce](https://github.com/us/connectorch/commit/db945cedc6045a4e13688fef835d91c8cd156b15))


### Bug Fixes

* **examples:** make the quoted benchmark reproducible ([864d496](https://github.com/us/connectorch/commit/864d49674d3287eabda3f857165cb3756f8ff93a))
* require neuprint-python 0.6.3 for the omit_rois argument ([432a54f](https://github.com/us/connectorch/commit/432a54fe2aa267fc7417a5993c9c2abbce0fc400))
* ship the bundled MaleCNS sample in the repository ([5856d1b](https://github.com/us/connectorch/commit/5856d1b977f2caa89ab57b838746f009f6a17bad))


### Documentation

* cite the published MaleCNS paper and separate its two neuron counts ([67e8530](https://github.com/us/connectorch/commit/67e85309a8864737976f34f1a2240841c9a86eed))
* **experiments:** record the result, which is negative ([4bbfb64](https://github.com/us/connectorch/commit/4bbfb645099aeb85bdbf850e1a87cc094769af87))
* explain what training does to the biology, and measure it ([8411d2a](https://github.com/us/connectorch/commit/8411d2a6832bc78090d9af4585fc9d0393264969))

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
