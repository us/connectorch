"""Worm chemotaxis: temporal gradient discrimination on the real sensorimotor circuit.

Probes showed real 0.85-0.90 vs rewired ~0.51 on ramp up/down classification
with frozen biology. This is the full experiment: start-matched stimulus
(no level confound), frozen count prior + Dale-style fixed weights, only the
motor readout trains.

    python experiments/10_chemotaxis.py --seeds 5

Writes one JSON line per run, and a markdown table.
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
from connectorch.nn import FixedWeights
from connectorch.nn.weights import initial_edge_weights
from connectorch.transforms import (
    degree_preserving_rewire,
    random_topology,
    shuffle_edge_weights,
)

ARMS = ["real", "degree_preserving", "random", "shuffled_weights"]

SENSORY = ["ASEL", "ASER", "AWCL", "AWCR", "ASHL", "ASHR", "ADFL", "ADFR"]
INTER = ["AVAL", "AVAR", "AVBL", "AVBR"]
VENTRAL = ("DA", "DB", "VA", "VB")

STEPS = 12


def load_circuit() -> tuple[ct.Connectome, np.ndarray, np.ndarray]:
    """Whole worm plus sensory input ids and motor readout ids."""
    brain = ct.datasets.elegans(download=True)
    names = np.array(brain.nodes.column("cell_name").to_pylist())
    ids = np.array(brain.node_ids)
    by_name = {n: i for i, n in enumerate(names)}
    input_ids = np.array([ids[by_name[n]] for n in SENSORY])
    motor = [n for n in names if n[:2] in VENTRAL]
    readout_ids = np.array([ids[by_name[n]] for n in motor])
    assert len(input_ids) == len(SENSORY) and len(readout_ids) > 0
    return brain, input_ids, readout_ids


def control(brain: ct.Connectome, arm: str, seed: int) -> ct.Connectome:
    if arm == "real":
        return brain
    if arm == "degree_preserving":
        return degree_preserving_rewire(brain, seed=seed)
    if arm == "random":
        return random_topology(brain, seed=seed)
    if arm == "shuffled_weights":
        return shuffle_edge_weights(brain, seed=seed)
    raise ValueError(arm)


def build(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    arm: str,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    graph = control(brain, arm, seed)
    prior = torch.as_tensor(
        initial_edge_weights(graph, "normalized_synapse_count"), dtype=torch.float32
    )
    core = ct.nn.ConnectomeRNN(
        graph,
        input_nodes=input_ids,
        output_nodes=readout_ids,
        weights=FixedWeights(prior),
        initializer="normalized_synapse_count",
        activation="tanh",
        leak=0.5,
        bias=False,
        dtype=torch.float32,
    )
    decoder = nn.Linear(len(readout_ids), 2)
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder)


def make_batch(
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Up vs down ramps with matched start level (no level confound)."""
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    starts = torch.rand(batch_size, generator=generator) * 0.5
    pos = torch.linspace(0, 1, steps)
    ramp = starts[:, None] + (pos[None, :] - 0.5) * 0.5
    up = labels == 0
    ramps = torch.where(up[:, None], ramp, ramp.flip(1))
    x = ramps[:, :, None].expand(-1, -1, len(SENSORY)).clone()
    return x, labels


def run_one(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    arm: str,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    model = build(brain, input_ids, readout_ids, arm, seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1_000_000 + seed)

    start = time.perf_counter()
    steps_per_epoch = max(1, args.train_samples // args.batch_size)
    for _ in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            x, labels = make_batch(args.batch_size, STEPS, gen)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_batch(args.batch_size, STEPS, gen)
            logits = model(x)
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()

    with torch.no_grad():
        sil_correct = sil_total = 0
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_batch(args.batch_size, STEPS, gen)
            traj = model.core(x)
            pooled = traj[:, -1, :].zero_()
            logits = model.decoder(pooled)
            sil_correct += int((logits.argmax(1) == labels).sum())
            sil_total += labels.numel()

    return {
        "arm": arm,
        "seed": seed,
        "accuracy": correct / total,
        "test_loss": float(np.mean(losses)),
        "silenced_accuracy": sil_correct / sil_total,
        "trainable_parameters": trainable,
        "seconds": round(time.perf_counter() - start, 1),
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| arm | test accuracy | test loss | silenced acc | trainable params |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        accs = [r["accuracy"] for r in rows if r["arm"] == arm]
        if not accs:
            continue
        loss = np.mean([r["test_loss"] for r in rows if r["arm"] == arm])
        sil = np.mean([r["silenced_accuracy"] for r in rows if r["arm"] == arm])
        params = rows[[r["arm"] for r in rows].index(arm)]["trainable_parameters"]
        lines.append(
            f"| `{arm}` | {np.mean(accs):.4f} ± {np.std(accs):.4f} "
            f"| {loss:.4f} | {sil:.4f} | {params:,} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out", default="experiments/results_chemotaxis.jsonl")
    args = parser.parse_args()

    brain, input_ids, readout_ids = load_circuit()
    print(
        f"inputs: {len(input_ids)} chemosensory, readout: {len(readout_ids)} motor, "
        f"circuit: {brain.num_nodes} nodes / {brain.num_edges} edges",
        flush=True,
    )
    rows = []
    for arm in args.arms:
        for seed in range(args.seeds):
            row = run_one(brain, input_ids, readout_ids, arm, seed, args)
            rows.append(row)
            print(
                f"{arm:20s} seed {seed}  acc {row['accuracy']:.4f}  "
                f"sil {row['silenced_accuracy']:.4f}  params {row['trainable_parameters']:,}  "
                f"{row['seconds']}s",
                flush=True,
            )
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    table = summarise(rows)
    print(f"\n{table}\n", flush=True)
    md = (
        "# Does the worm circuit compute chemotaxis?\n\n"
        f"{brain.num_nodes} nodes, {brain.num_edges} connections. "
        f"{args.seeds} seeds, {args.epochs} epoch(s), start-matched ramp direction.\n\n"
        f"{table}\n"
    )
    Path(args.out).with_suffix(".md").write_text(md)
    print(f"wrote {len(rows)} runs to {args.out}", flush=True)

    prov = {
        "experiment": "10_chemotaxis",
        "platform": platform.platform(),
        "seeds": args.seeds,
        "epochs": args.epochs,
    }
    Path(args.out).with_suffix(".prov.json").write_text(json.dumps(prov, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
