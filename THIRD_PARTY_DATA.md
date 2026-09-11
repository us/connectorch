# Third-party data

ConnecTorch's source code is MIT licensed. That license covers the code only. It
does **not** relicense any biological dataset this project loads, links to, or
bundles. Each dataset keeps the license its authors gave it, and using ConnecTorch
to read a dataset does not change your obligations under that license.

## MaleCNS v1.0

**Bundled in this repository and in the published wheel:**
`src/connectorch/datasets/data/malecns_sample.ct` (100 neurons, 3,630 connections).

| | |
|---|---|
| Dataset | Male CNS connectome, version 1.0 (`male-cns:v1.0`) |
| Authors | Janelia Research Campus FlyEM project, Google Research, and the University of Cambridge |
| Homepage | https://male-cns.janelia.org/ |
| Paper | Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell **189**, 5504–5526.e15 (2026). https://doi.org/10.1016/j.cell.2026.08.015 |
| Source files | GCS bucket `flyem-male-cns`, prefix `v1.0/connectome-data/flat-connectome/`, readable over plain HTTPS without credentials |
| License | Creative Commons Attribution (CC-BY 4.0) |

**Modifications made to produce the bundled sample.** The full dataset was read
from `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` and
`body-annotations-male-cns-v1.0-minconf-0.5.feather`. Connections below 5 synapses
were dropped. A 100-body subgraph was then selected deterministically, by growing a
set greedily from the highest-degree body inside a pool of the 4,000
highest-degree bodies, always adding the candidate with the most connections into
the set, with ties broken by ascending body id. Body ids, cell types and synapse
counts are unchanged from the published data; nothing was synthesised. The full
selection procedure is recorded in the sample's own provenance
(`connectome.provenance["sample"]`) and its source is
`src/connectorch/datasets/build_sample.py`.

**If you use this data, cite the original authors, not ConnecTorch:**

> Berg, S. et al. Sexual dimorphism in the complete Drosophila male central
> nervous system connectome. Cell 189, 5504-5526.e15 (2026).
> doi:10.1016/j.cell.2026.08.015
> Data: MaleCNS v1.0, Janelia FlyEM / Google Research / Cambridge Drosophila
> Connectomics Group. https://male-cns.janelia.org/

Full datasets downloaded at runtime by `connectorch.datasets.malecns()` are
likewise CC-BY and are never redistributed by this project; they are fetched from
the publishers' own bucket on the user's machine.

## Everything else

`connectorch.datasets.toy` generates synthetic graphs with numpy. They are not
biological data and carry no third-party rights.

No code was copied from FlyVis, WormVAE, connectome-interpreter, btorch, or any
other project. Those projects informed the design through their public papers and
documented APIs; the implementation here is independent.
