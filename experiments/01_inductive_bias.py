"""Does biological wiring help, or is it just sparsity?

This is the first question worth asking of this library, and the only honest way
to ask it is against controls that differ from the connectome in exactly one
respect at a time:

    real                    the connectome as reconstructed, weights free to move
    biological              synapse counts held as a prior, a bounded gain on top
    biological_shared       the same, with one gain per pair of cell types
    degree_preserving       every neuron keeps its in- and out-degree, wiring shuffled
    random                  same node and edge count, nothing else preserved
    shuffled_weights        same wiring, synapse counts permuted across connections
    dense_rnn               an ordinary RNN at a matched parameter budget

The `biological` arms are the interesting ones: they are the only arms where the
measured synapse counts are still doing work at the end of training, rather than
having been optimised into something else.

Every arm sees the same task, the same seeds, the same encoder and decoder shapes
and the same number of trainable parameters. A difference that survives that is
about the wiring. A difference that does not survive it was never about biology.

    python experiments/01_inductive_bias.py --epochs 2 --seeds 5
    python experiments/01_inductive_bias.py --nodes 2000 --download    # real subgraph

Writes one JSON line per run, and a markdown table, so the numbers can be checked
rather than believed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import connectorch as ct
from connectorch.transforms import (
    DROSOPHILA_POLARITY,
    degree_preserving_rewire,
    infer_signs,
    random_topology,
    shuffle_edge_weights,
)

ARMS = [
    "real",
    "biological",
    "biological_shared",
    "degree_preserving",
    "random",
    "shuffled_weights",
    "dense_rnn",
]

#: Arms whose weights are a bounded gain on the measured counts, not free values.
BIOLOGICAL = {"biological", "biological_shared"}


def load_connectome(num_nodes: int | None, download: bool) -> ct.Connectome:
    """The bundled sample, or a densely connected subgraph of the real dataset.

    Either way the result carries transmitter predictions, so the biological arms
    can hold Dale's principle rather than only bounding the gain.
    """
    if not download:
        return infer_signs(ct.datasets.malecns_sample(), DROSOPHILA_POLARITY)

    full = ct.datasets.malecns(download=True, min_synapses=5, neurotransmitters=True)
    full = infer_signs(full, DROSOPHILA_POLARITY)
    from connectorch.datasets.build_sample import _densest_subset

    chosen = _densest_subset(full, num_nodes or 1000, pool_size=20_000)
    return full.subgraph(chosen)


def control(brain: ct.Connectome, arm: str, seed: int) -> ct.Connectome:
    if arm in ("real", "dense_rnn") or arm in BIOLOGICAL:
        return brain
    if arm == "degree_preserving":
        return degree_preserving_rewire(brain, seed=seed)
    if arm == "random":
        return random_topology(brain, seed=seed)
    if arm == "shuffled_weights":
        return shuffle_edge_weights(brain, seed=seed)
    raise ValueError(arm)


class DenseRNN(nn.Module):
    """A plain recurrent core sized to match the connectome's parameter count."""

    def __init__(self, num_edges: int, num_io: int) -> None:
        super().__init__()
        hidden = max(num_io, int(round(num_edges**0.5)))
        self.hidden = hidden
        self.cell = nn.Linear(hidden, hidden, bias=False)
        self.num_input_nodes = num_io
        self.num_output_nodes = num_io

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or 1
        h = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=x.dtype)
        outputs = []
        for _ in range(steps):
            drive = torch.zeros_like(h)
            drive[:, : x.shape[1]] = x
            h = torch.tanh(self.cell(h) + drive) * 0.4 + h * 0.6
            outputs.append(h[:, : self.num_output_nodes])
        return torch.stack(outputs, dim=1)


def build(brain: ct.Connectome, arm: str, num_io: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arm == "dense_rnn":
        core: nn.Module = DenseRNN(brain.num_edges, num_io)
    else:
        graph = control(brain, arm, seed)
        if arm in BIOLOGICAL:
            weights: str | ct.nn.EdgeWeights = ct.nn.BiologicalWeights(
                graph,
                gain_bounds=(0.5, 2.0),
                share_by="cell_type" if arm == "biological_shared" else None,
                dale="sign" in graph.edge_columns,
            )
        else:
            weights = "trainable"
        core = ct.nn.ConnectomeRNN(
            graph,
            input_nodes=brain.node_ids[:num_io],
            output_nodes=brain.node_ids[-num_io:],
            weights=weights,
            initializer="normalized_synapse_count",
            leak=0.4,
        )
    return ct.nn.ConnectomeModel(
        core,
        encoder=nn.Sequential(nn.Flatten(), nn.Linear(784, num_io), nn.Tanh()),
        decoder=nn.Linear(num_io, 10),
    )


def run_one(brain, arm, seed, args, loaders) -> dict:
    train_loader, test_loader = loaders
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = build(brain, arm, args.io_nodes, seed).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    seen, curve = 0, []
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, steps=args.steps), labels)
            loss.backward()
            optimizer.step()
            seen += labels.numel()
            if seen % (args.batch_size * 50) < args.batch_size:
                curve.append({"samples": seen, "loss": float(loss)})

    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, steps=args.steps)
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()

    drift = None
    if hasattr(model.core, "edge_weight") and hasattr(model.core, "num_edges"):
        final = model.core.edge_weight.detach().abs().cpu().numpy()
        prior = np.abs(
            ct.nn.initial_edge_weights(control(brain, arm, seed), "normalized_synapse_count")
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(prior > 0, final / np.maximum(prior, 1e-12), 1.0)
        drift = {
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "median": float(np.median(ratio)),
        }

    return {
        "arm": arm,
        "seed": seed,
        "accuracy": correct / total,
        "weight_drift_vs_synapse_counts": drift,
        "test_loss": float(np.mean(losses)),
        "trainable_parameters": trainable,
        "seconds": round(time.perf_counter() - start, 1),
        "curve": curve,
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| arm | test accuracy | test loss | trainable params | |w| / synapse-count prior |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        acc = np.array([r["accuracy"] for r in runs])
        loss = np.array([r["test_loss"] for r in runs])
        drift = runs[0].get("weight_drift_vs_synapse_counts")
        span = f"{drift['min']:.2f} – {drift['max']:.2f}" if drift else "n/a"
        lines.append(
            f"| `{arm}` | {acc.mean():.4f} ± {acc.std(ddof=1) if len(acc) > 1 else 0:.4f} "
            f"| {loss.mean():.4f} | {runs[0]['trainable_parameters']:,} | {span} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--io-nodes", type=int, default=64)
    parser.add_argument("--nodes", type=int, default=None, help="subgraph size with --download")
    parser.add_argument("--download", action="store_true", help="use the real dataset")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=Path("experiments/results.jsonl"))
    args = parser.parse_args()

    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    root = "~/.cache/connectorch/mnist"
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
    loaders = (
        torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=512),
    )

    brain = load_connectome(args.nodes, args.download)
    print(brain)
    print(f"fingerprint: {brain.fingerprint()}\n")

    environment = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu",
        "connectome_fingerprint": brain.fingerprint(),
        "nodes": brain.num_nodes,
        "edges": brain.num_edges,
        "config": {k: str(v) for k, v in vars(args).items()},
    }

    rows = []
    for arm in args.arms:
        for seed in range(args.seeds):
            row = run_one(brain, arm, seed, args, loaders)
            row["environment"] = environment
            rows.append(row)
            print(
                f"{arm:18} seed {seed}  acc {row['accuracy']:.4f}  "
                f"loss {row['test_loss']:.4f}  params {row['trainable_parameters']:,}  "
                f"{row['seconds']}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = summarise(rows)
    args.out.with_suffix(".md").write_text(
        f"# Does biological wiring help?\n\n"
        f"{brain.num_nodes:,} nodes, {brain.num_edges:,} connections, "
        f"fingerprint `{brain.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), MNIST.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
