# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/). Anything that changes the numerical
semantics of the runtime is called out explicitly, because a silent change there
would invalidate other people's results.

## 0.1.0 (2026-09-18)


### Features

* add a content hash beside the structural fingerprint ([4b3d77e](https://github.com/us/connectorch/commit/4b3d77ef466e6f0e72dabe769bb37efa9b7a7b0e))
* add native Apple Metal CSR propagation and gradients ([7c36c84](https://github.com/us/connectorch/commit/7c36c849fd01374a68bc4c41dd54b3d76cb3dcdb))
* add native Apple Metal CSR propagation and gradients ([becfdc9](https://github.com/us/connectorch/commit/becfdc945dc43108951b6fedc870c9a6ddcdf551))
* compile biological connectomes into trainable PyTorch networks ([0e3ec5e](https://github.com/us/connectorch/commit/0e3ec5e5acdce74b66ec29f9d8aed9ab5ce1acc7))
* **datasets:** C. elegans Cook 2019 loader, second species ([2021905](https://github.com/us/connectorch/commit/20219058a486f8acb133c1338a9b1216e0748821))
* **experiments:** 2D endpoint-free flow bound with correct drive ([e3710c3](https://github.com/us/connectorch/commit/e3710c36b5463dfd75dbbc1d5d25a2f0ef10df2a))
* **experiments:** 2D flow task on the motion circuit with T4+T5 readout ([797e21c](https://github.com/us/connectorch/commit/797e21c4eeeb8cc75b9e0bdb315c6b0eb75c3a17))
* **experiments:** direction task on the motion circuit ([0f7d0ea](https://github.com/us/connectorch/commit/0f7d0eaa776057390f0d0369bec51ff6dc8b7754))
* **experiments:** endpoint-free mechanism dissection of motion wiring ([ea8b4a3](https://github.com/us/connectorch/commit/ea8b4a3ce5958a1e44dceee50b3f02e4c6d2bffb))
* **experiments:** flyhash efficiency win at matched compute ([fdacb02](https://github.com/us/connectorch/commit/fdacb02a26da74db1f8806cb430a0035f2c7af2b))
* **experiments:** mushroom-body head wins few-shot classification ([55fdf1b](https://github.com/us/connectorch/commit/55fdf1bdd280a8f8375b50c2798ac7939a1b44f1))
* **experiments:** worm chemotaxis on the real sensorimotor circuit ([c2bb1e8](https://github.com/us/connectorch/commit/c2bb1e838ee56ab67bb4d620895e5d63f43a274c))
* fly-pure dynamics and first positive selectivity result ([e437dca](https://github.com/us/connectorch/commit/e437dca42562608d7b8bb058988e50a805f0d22a))
* keep the measured synapse counts and polarity through training ([51fa277](https://github.com/us/connectorch/commit/51fa27728fa54f5b564bece1c713e8ba5f55b264))
* load neurotransmitter predictions and turn them into connection polarity ([db945ce](https://github.com/us/connectorch/commit/db945cedc6045a4e13688fef835d91c8cd156b15))


### Bug Fixes

* **examples:** make the quoted benchmark reproducible ([864d496](https://github.com/us/connectorch/commit/864d49674d3287eabda3f857165cb3756f8ff93a))
* require neuprint-python 0.6.3 for the omit_rois argument ([432a54f](https://github.com/us/connectorch/commit/432a54fe2aa267fc7417a5993c9c2abbce0fc400))
* ship the bundled MaleCNS sample in the repository ([5856d1b](https://github.com/us/connectorch/commit/5856d1b977f2caa89ab57b838746f009f6a17bad))
* **tests:** strict zip in elegans annotation test ([cd7f809](https://github.com/us/connectorch/commit/cd7f809bef5355f0a916096f3f913374b8d00229))


### Documentation

* cite the published MaleCNS paper and separate its two neuron counts ([67e8530](https://github.com/us/connectorch/commit/67e85309a8864737976f34f1a2240841c9a86eed))
* **experiments:** OOD shift results for the motion wiring ([6777fc9](https://github.com/us/connectorch/commit/6777fc9631219f012406ecacc85a8f0e9775ac5f))
* **experiments:** record the result, which is negative ([4bbfb64](https://github.com/us/connectorch/commit/4bbfb645099aeb85bdbf850e1a87cc094769af87))
* explain what training does to the biology, and measure it ([8411d2a](https://github.com/us/connectorch/commit/8411d2a6832bc78090d9af4585fc9d0393264969))
* findings writeup of the motion-circuit program ([18fd14b](https://github.com/us/connectorch/commit/18fd14beaf42a81a91db6ff9f56148218e3ddb6c))
* **findings:** record continual-learning killed probe ([87677e7](https://github.com/us/connectorch/commit/87677e737beee5891740d94164ee34cb038c7ea9))
* **findings:** record learnable-leak and ring-attractor killed probes ([3df4459](https://github.com/us/connectorch/commit/3df4459520dbd1dd7e28c2af72dda7866f4e5b4e))
* **findings:** record noise and speed-augmentation killed probes ([2c107e3](https://github.com/us/connectorch/commit/2c107e3cc70da57a4cce950d569739750d5058a1))

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
