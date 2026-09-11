"""Build the small MaleCNS fixture that ships inside the wheel.

Run once on a machine with the full dataset::

    python -m connectorch.datasets.build_sample --download

The selection is deterministic: the 100 bodies with the highest total degree,
ties broken by body id, then the induced subgraph. That gives a densely connected
sample with real cell types, which is a more useful fixture than a random sample of
mostly disconnected neurons.

The output keeps MaleCNS's CC-BY attribution in its provenance. See
``THIRD_PARTY_DATA.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..ir import Connectome
from .malecns import DATASET_INFO, malecns

__all__ = ["build_sample"]

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "malecns_sample.ct"


def build_sample(
    num_nodes: int = 100,
    *,
    variant: str = "traced-only",
    min_synapses: int = 5,
    pool_size: int = 4000,
    download: bool | None = None,
    output: Path = DEFAULT_OUTPUT,
) -> Connectome:
    """Select a deterministic subgraph of MaleCNS and write it to ``output``."""
    brain = malecns(variant=variant, min_synapses=min_synapses, download=download)
    print(f"full: {brain.num_nodes:,} nodes, {brain.num_edges:,} edges")

    chosen = _densest_subset(brain, num_nodes, pool_size=pool_size)

    sample = brain.subgraph(chosen)
    sample.provenance.update(
        {
            "dataset": DATASET_INFO["neuprint_dataset"],
            "license": DATASET_INFO["license"],
            "citation": DATASET_INFO["citation"],
            "sample": {
                "selection": (
                    "greedy densest subgraph over the highest-degree "
                    f"{pool_size} bodies, ties by ascending body id"
                ),
                "num_nodes": int(num_nodes),
                "variant": variant,
                "min_synapses": int(min_synapses),
            },
        }
    )
    print(f"sample: {sample.num_nodes:,} nodes, {sample.num_edges:,} edges")
    print(f"fingerprint: {sample.fingerprint()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    sample.save(output)
    total = sum(p.stat().st_size for p in output.rglob("*"))
    print(f"wrote {output} ({total / 1024:,.0f} KiB)")
    return sample


def _densest_subset(brain: Connectome, num_nodes: int, *, pool_size: int) -> np.ndarray:
    """Pick a densely interconnected set of bodies, deterministically.

    Taking the globally highest-degree neurons gives a nearly edgeless sample:
    the busiest cells sit in different parts of the nervous system and do not
    talk to each other. So this grows a set greedily instead, always adding the
    candidate with the most connections into the set so far, which yields a
    sample that is an actual circuit.

    The search runs inside a pool of the highest-degree bodies, so it stays cheap
    on a 25-million-edge graph. Every tie is broken by ascending body id, so the
    result never depends on a sort implementation.
    """
    source, target = brain.edge_index
    degree = np.bincount(source, minlength=brain.num_nodes) + np.bincount(
        target, minlength=brain.num_nodes
    )
    pool = np.lexsort((brain.node_ids, -degree))[:pool_size]

    in_pool = np.zeros(brain.num_nodes, dtype=bool)
    in_pool[pool] = True
    keep = in_pool[source] & in_pool[target]
    remap = np.full(brain.num_nodes, -1, dtype=np.int64)
    remap[pool] = np.arange(pool.size)
    pool_source, pool_target = remap[source[keep]], remap[target[keep]]

    pool_degree = np.bincount(pool_source, minlength=pool.size) + np.bincount(
        pool_target, minlength=pool.size
    )
    pool_ids = brain.node_ids[pool]

    selected = np.zeros(pool.size, dtype=bool)
    links = np.zeros(pool.size, dtype=np.int64)
    chosen: list[int] = []

    current = int(np.lexsort((pool_ids, -pool_degree))[0])
    for _ in range(min(num_nodes, pool.size)):
        selected[current] = True
        chosen.append(current)
        links += np.bincount(pool_target[pool_source == current], minlength=pool.size)
        links += np.bincount(pool_source[pool_target == current], minlength=pool.size)
        scores = np.where(selected, -1, links)
        current = int(np.lexsort((pool_ids, -pool_degree, -scores))[0])

    return pool_ids[np.array(chosen)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-nodes", type=int, default=100)
    parser.add_argument("--variant", default="traced-only")
    parser.add_argument("--min-synapses", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_sample(
        args.num_nodes,
        variant=args.variant,
        min_synapses=args.min_synapses,
        pool_size=args.pool_size,
        download=True if args.download else None,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
