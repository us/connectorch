"""Measure the propagation backends so ``backend="auto"`` is a fact, not an opinion.

Run::

    python benchmarks/sparse_backends.py --out results.jsonl

Reports forward time, forward+backward time, peak memory and steps per second for
every backend at several graph sizes and batch sizes, on CPU and on CUDA when it is
available. Nothing in this repository may claim a performance number that this
script has not produced.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from connectorch.backends import build_propagator
from connectorch.datasets import random_sparse
from connectorch.exceptions import ConnectorchError

SIZES = [(1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000)]
BATCHES = [1, 32]
BACKENDS = ["dense", "sparse_mm", "scatter"]

#: Skip the dense backend above this many nodes; it allocates N^2.
DENSE_NODE_LIMIT = 4_000


def make_graph(num_nodes: int, num_edges: int, seed: int = 0) -> torch.Tensor:
    """Return a ``[2, E]`` edge_index exactly as the IR would produce it.

    Going through ``Connectome`` rather than raw random indices matters: the IR
    aggregates parallel edges, and a benchmark on an edge set the library would
    never build measures something the library never runs.
    """
    brain = random_sparse(num_nodes, num_edges, seed=seed)
    return torch.as_tensor(brain.edge_index, dtype=torch.int64)


def timed(fn, *, device: str, warmup: int = 3, iters: int = 10) -> float:
    """Median-ish wall time per call in milliseconds, with the device synchronised."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def peak_memory_gib(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def reset_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_one(backend: str, num_nodes: int, num_edges: int, batch: int, device: str) -> dict:
    edge_index = make_graph(num_nodes, num_edges).to(device)
    weight = torch.rand(edge_index.shape[1], device=device)
    state = torch.randn(num_nodes, batch, device=device)
    propagator = build_propagator(backend, edge_index, num_nodes, trainable=False).to(device)

    reset_memory(device)
    forward_ms = timed(lambda: propagator(state, weight), device=device)
    forward_peak = peak_memory_gib(device)

    def forward_backward() -> None:
        w = weight.clone().requires_grad_(True)
        propagator(state, w).sum().backward()

    reset_memory(device)
    try:
        both_ms = timed(forward_backward, device=device, warmup=2, iters=5)
        both_peak = peak_memory_gib(device)
    except ConnectorchError as error:
        both_ms, both_peak = float("nan"), float("nan")
        refused = str(error).splitlines()[0]
        print(f"    {backend} refused forward+backward: {refused}")
    except (torch.OutOfMemoryError, RuntimeError) as error:
        both_ms, both_peak = float("nan"), float("nan")
        print(f"    {backend} forward+backward failed: {type(error).__name__}: {error}")

    return {
        "backend": backend,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "batch": batch,
        "device": device,
        "forward_ms": round(forward_ms, 4),
        "forward_backward_ms": round(both_ms, 4),
        "forward_peak_gib": round(forward_peak, 4),
        "forward_backward_peak_gib": round(both_peak, 4),
        "steps_per_second": round(1000 / forward_ms, 1) if forward_ms else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results.jsonl"))
    parser.add_argument("--devices", nargs="*", default=None)
    args = parser.parse_args()

    devices = args.devices or (["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    environment = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(environment))

    rows = []
    for device in devices:
        for num_nodes, num_edges in SIZES:
            for batch in BATCHES:
                for backend in BACKENDS:
                    if backend == "dense" and num_nodes > DENSE_NODE_LIMIT:
                        continue
                    row = run_one(backend, num_nodes, num_edges, batch, device)
                    row["environment"] = environment
                    rows.append(row)
                    print(
                        f"{device:5} N={num_nodes:>7,} B={batch:>3} {backend:10} "
                        f"fwd {row['forward_ms']:8.3f} ms  "
                        f"f+b {row['forward_backward_ms']:9.3f} ms  "
                        f"peak {row['forward_backward_peak_gib']:7.3f} GiB"
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    table = args.out.with_suffix(".md")
    table.write_text(render_markdown(rows, environment))
    print(f"\nwrote {len(rows)} rows to {args.out} and a table to {table}")
    return 0


def render_markdown(rows: list[dict], environment: dict) -> str:
    """Render the measurements as a table a human can read without jq."""
    lines = [
        "# Backend benchmarks",
        "",
        "Generated by `python benchmarks/sparse_backends.py`. Every performance",
        "claim in this project traces back to a file like this one.",
        "",
        f"- torch: `{environment['torch']}`",
        f"- platform: `{environment['platform']}`",
        f"- CUDA device: `{environment['cuda_device'] or 'none'}`",
        "",
        "`f+b` is forward plus backward. `nan` means the configuration was",
        "refused or failed; the reason is printed when the benchmark runs.",
        "",
    ]
    for device in dict.fromkeys(row["device"] for row in rows):
        lines += [
            f"## {device}",
            "",
            "| nodes | edges | batch | backend | forward (ms) | f+b (ms) | f+b peak (GiB) |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in (r for r in rows if r["device"] == device):
            peak = row["forward_backward_peak_gib"]
            lines.append(
                f"| {row['num_nodes']:,} | {row['num_edges']:,} | {row['batch']} "
                f"| `{row['backend']}` | {row['forward_ms']:.3f} "
                f"| {row['forward_backward_ms']:.3f} "
                f"| {'n/a' if peak != peak else f'{peak:.3f}'} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
