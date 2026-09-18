"""FlyHash check: sparse expansion wins the efficiency frontier.

Dasgupta, Stevens, Navlakha (Science 2017) showed the fly mushroom-body
motif (sparse binary expansion + winner-take-all) beats dense random
projection for similarity search AT MATCHED COMPUTE. This checks that
``connectorch.nn.SparseExpander`` reproduces that pattern on cached MNIST:

- ``fly``: 784 -> 2000, fan_in 6, top-100 active (~12k ops/query)
- ``lsh_matched``: dense sign projection 784 -> 15 (~12k ops/query)
- ``lsh_big``: dense sign projection 784 -> 2000 (~1.5M ops/query, reference
  only: 100x the compute, expected to win on accuracy and to lose on cost)

Metric is mean top-10 overlap against cosine ground truth over 200 cached
MNIST test queries (no download). 5 seeds.

    python experiments/08_flyhash.py --seeds 5

Writes one JSON line per run, and a markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from connectorch.nn import SparseExpander

ARMS = ["fly", "lsh_matched", "lsh_big"]

OPS = {"fly": 12_000, "lsh_matched": 11_760, "lsh_big": 1_568_000}


def load_mnist(n_db: int = 2000, n_q: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    from connectorch.datasets._cache import default_cache_dir

    ds = datasets.MNIST(root=str(default_cache_dir() / "mnist"), train=False, download=False)
    X = ds.data[:n_db].float().reshape(n_db, -1) / 255.0
    Q = ds.data[n_db : n_db + n_q].float().reshape(n_q, -1) / 255.0
    mean = X.mean()
    return X - mean, Q - mean


def topk_hit(order: torch.Tensor, truth: torch.Tensor, k: int = 10) -> float:
    hits = 0
    for i in range(order.shape[0]):
        hits += len(set(truth[i, :k].tolist()) & set(order[i, :k].tolist()))
    return hits / order.shape[0]


def run_one(X: torch.Tensor, Q: torch.Tensor, truth: torch.Tensor, arm: str, seed: int) -> dict:
    start = time.perf_counter()
    with torch.no_grad():
        if arm == "fly":
            mod = SparseExpander(784, 2000, fan_in=6, k=100, seed=seed)
            order = (mod(Q).float() @ mod(X).float().T).argsort(descending=True)
        else:
            length = 15 if arm == "lsh_matched" else 2000
            gen = torch.Generator().manual_seed(seed)
            P = torch.randn(784, length, generator=gen)
            order = (((Q @ P) > 0).float() @ ((X @ P) > 0).float().T).argsort(descending=True)
    return {
        "arm": arm,
        "seed": seed,
        "top10_hit": round(topk_hit(order, truth), 4),
        "ops_per_query": OPS[arm],
        "seconds": round(time.perf_counter() - start, 1),
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| arm | mean top-10 overlap | ops/query |",
        "|---|---|---|",
    ]
    for arm in ARMS:
        vals = [r["top10_hit"] for r in rows if r["arm"] == arm]
        lines.append(f"| `{arm}` | {np.mean(vals):.2f} ± {np.std(vals):.2f} | {OPS[arm]:,} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="experiments/results_flyhash.jsonl")
    args = parser.parse_args()

    X, Q = load_mnist()
    with torch.no_grad():
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-9)
        Qn = Q / (Q.norm(dim=1, keepdim=True) + 1e-9)
        truth = (Qn @ Xn.T).argsort(descending=True)

    rows = []
    for seed in range(args.seeds):
        for arm in ARMS:
            row = run_one(X, Q, truth, arm, seed)
            rows.append(row)
            print(f"{arm:12s} seed {seed} hit {row['top10_hit']:.2f}", flush=True)

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    table = (
        "# Does sparse expansion win at matched compute?\n\n"
        f"MNIST test, 200 queries vs 2000 db, cosine ground truth, {args.seeds} seeds.\n\n"
        + summarise(rows)
    )
    out.with_suffix(".md").write_text(table)
    print(f"\n{table}\nwrote {len(rows)} runs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
