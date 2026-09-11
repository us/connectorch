# References

Every claim this project makes about biology, about prior work, or about PyTorch's
behaviour traces to something on this page. Each entry was checked against the
publisher or the issue tracker, not quoted from memory.

## The data

**MaleCNS v1.0** — the connectome this library ships a sample of and loads in full.

> Berg, S. et al. *Sexual dimorphism in the complete connectome of the Drosophila
> male central nervous system.* bioRxiv 2025.10.09.680999 (2025).
> <https://doi.org/10.1101/2025.10.09.680999>

The paper reports **166,691 neurons** spanning brain and ventral nerve cord.
`ct.datasets.malecns()` loads 164,587 by default because the `traced-only`
connectivity table only contains bodies with traced connections; pass
`variant="full"` for the unfiltered table. Data released under **CC-BY**; the full attribution, including exactly how the
bundled sample was derived, is in `THIRD_PARTY_DATA.md` at the root of the
repository. Project page: <https://male-cns.janelia.org/>. Produced by Janelia
FlyEM, Google Research, and the Cambridge Drosophila Connectomics Group.

**Neurotransmitter predictions** — the source of the `body-neurotransmitters`
table, and the reason this library refuses to infer an edge's sign for you.

> Eckstein, N. et al. *Neurotransmitter classification from electron microscopy
> images at synaptic sites in Drosophila melanogaster.* Cell **187**, 2574–2594.e23
> (2024). <https://doi.org/10.1016/j.cell.2024.03.016>

These are *predictions of the transmitter released*, not of the sign of the
connection. Synaptic polarity depends on the receptor on the postsynaptic side, so
a transmitter label alone does not determine whether a connection excites or
inhibits. See [Scientific caveats](caveats.md).

**neuPrint** — the server behind `connectorch.io.from_neuprint`.

> Plaza, S. M., Clements, J. et al. *neuPrint: An open access tool for EM
> connectomics.* Frontiers in Neuroinformatics **16**, 896292 (2022).
> <https://doi.org/10.3389/fninf.2022.896292>

## Prior work

**FlyVis** — the strongest published evidence that connectome-constrained training
works, and the source of the gather/scatter approach this library uses.

> Lappalainen, J. K. et al. *Connectome-constrained networks predict neural
> activity across the fly visual system.* Nature **634**, 1132–1140 (2024).
> <https://doi.org/10.1038/s41586-024-07939-3> · <https://github.com/TuragaLab/flyvis>

FlyVis models one system of one species, shares weights per cell-type pair rather
than per synapse, and carries a stimulus pipeline built for visual motion.
ConnecTorch is dataset- and task-agnostic, and parameterises per connection by
default.

**Connectome Interpreter** — the nearest prior art, and a better tool than this one
if your question is about paths and effective connectivity rather than training.

> Yin, Y. et al. *The Connectome Interpreter Toolkit.* bioRxiv 2025.09.29.679410
> (2025). <https://doi.org/10.1101/2025.09.29.679410> ·
> <https://github.com/YijieYin/connectome_interpreter>

**WormVAE** — connectome-constrained latent variable modelling in *C. elegans*.

> <https://github.com/TuragaLab/wormvae>

## PyTorch behaviour this library works around

The choice of `scatter` as the training backend is forced by an open upstream bug,
not by preference. See [Sparse backends](backends.md) for the measurements.

- **Dense intermediate in `torch.sparse.mm` backward**:
  [pytorch/pytorch#41128](https://github.com/pytorch/pytorch/issues/41128),
  [#49683](https://github.com/pytorch/pytorch/issues/49683). The gradient returned
  is sparse-typed, but a dense `[N, N]` is materialised on the way there.
- **No deterministic CUDA kernel for `index_add_`**:
  [pytorch/pytorch#108569](https://github.com/pytorch/pytorch/issues/108569).
  `torch.use_deterministic_algorithms(True)` raises rather than falling back.
- **`torch.compile` graph breaks on index ops inside backward**:
  [pytorch/pytorch#114686](https://github.com/pytorch/pytorch/issues/114686).
  This is why `torch.compile` support is not claimed anywhere in these docs.
- **`torch.sparse` is a beta API**:
  <https://docs.pytorch.org/docs/main/sparse.html>
- **Native scatter over `torch_scatter`**: PyTorch Geometric is itself migrating
  off `torch_scatter` onto native ops
  ([pyg-team/pytorch_geometric#5867](https://github.com/pyg-team/pytorch_geometric/issues/5867)),
  which is why ConnecTorch does not take it as a dependency.

## How to cite ConnecTorch

If ConnecTorch is useful to you, cite the software (see `CITATION.cff`) **and,
separately, the connectome you used**. The dataset authors did the hard part.
