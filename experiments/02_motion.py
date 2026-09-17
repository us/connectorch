"""Motion direction on the real early-visual circuit.

Exp 01 showed topology does not matter with 100 random neurons and a static
task. This experiment changes the task before changing anything else: a
direction-discrimination sequence on the actual motion pathway
(L1 -> Mi1/Mi4/Mi9 -> T4, as reconstructed), with inputs mapped
retinotopically onto L1 via the published optic-lobe hex coordinates and
readout from the T4 direction-selective cells.

    real                    the circuit as reconstructed, weights free to move
    biological              synapse counts held as a prior, a bounded gain on top
    biological_shared       the same, with one gain per pair of cell types
    degree_preserving       every neuron keeps its in- and out-degree, wiring shuffled
    random                  same node and edge count, nothing else preserved
    shuffled_weights        same wiring, synapse counts permuted across connections
    dense_rnn               an ordinary RNN at a matched parameter budget

Stimulus: a Gaussian bump sweeping across the 36 hex columns over T steps,
left-to-right or right-to-left. Label: sweep direction.

    python experiments/02_motion.py --arms real --seeds 1 --epochs 1 --train-samples 256
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/02_motion.py --seeds 5 --epochs 3

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

#: Columnar early-visual types: lamina + medulla + transmedullary + T4/T5.
COLUMNAR = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "C2",
    "C3",
    "T1",
    "Mi1",
    "Mi4",
    "Mi9",
    "Tm1",
    "Tm2",
    "Tm4",
    "Tm9",
    "Tm20",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
]

ANNOTATION_COLUMNS = {
    "type": "cell_type",
    "superclass": "superclass",
    "somaSide": "side",
    "assignedOlHex1": "hex1",
    "assignedOlHex2": "hex2",
    "status": "status",
}


def load_circuit() -> tuple[ct.Connectome, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Full early-visual circuit plus ordered L1 input ids and T4 readout ids.

    Returns the signed circuit, L1 input node ids ordered by hex column,
    T4 readout node ids, the L1 hex positions (float, one per input), and the
    T4 subtype group index per readout id (for subtype mean-pooling).
    """
    brain = ct.datasets.malecns(
        variant="traced-only",
        min_synapses=5,
        annotations=True,
        annotation_columns=ANNOTATION_COLUMNS,
        neurotransmitters=True,
        download=True,
    )
    brain = infer_signs(brain, DROSOPHILA_POLARITY)
    circuit = brain.subgraph(brain.where(cell_type=COLUMNAR))

    cell_type = np.array(circuit.nodes.column("cell_type").to_pylist())
    hex1 = np.array(circuit.nodes.column("hex1").to_pylist())
    ids = np.array(circuit.node_ids)

    l1 = (cell_type == "L1") & (hex1 != "")
    # One representative L1 per hex column: the stimulus is one-dimensional
    # over the 36 columns, so driving all ~49 cells per column is redundant.
    # This also keeps the input width (36) small enough for a matched dense
    # control. Deterministic: lowest body id per column.
    seen: dict[float, int] = {}
    for i in np.argsort(ids[l1], kind="stable"):
        h = float(hex1[l1][i])
        seen.setdefault(h, int(ids[l1][i]))
    cols = sorted(seen)
    input_ids = np.array([seen[h] for h in cols])
    input_hex = np.array(cols)

    readout_mask = np.isin(cell_type, ["T4a", "T4b", "T4c", "T4d"])
    readout_ids = ids[readout_mask]
    subtypes = ["T4a", "T4b", "T4c", "T4d"]
    readout_groups = [subtypes.index(c) for c in cell_type[readout_mask]]
    assert len(input_ids) > 0 and len(readout_ids) > 0
    assert circuit.num_edges > 0
    return circuit, input_ids, readout_ids, input_hex, readout_groups


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


class SubtypeMean(nn.Module):
    """Frozen mean-pool over readout neurons grouped by cell type.

    T4a/b/c/d are the four biological direction channels; pooling by subtype
    keeps that signal while making the decoder tiny, so the dense control can
    meet the same parameter budget. Carries no trainable parameters.
    """

    def __init__(self, groups: list[int]) -> None:
        super().__init__()
        n_out = len(set(groups))
        weight = torch.zeros(n_out, len(groups))
        for i, g in enumerate(groups):
            weight[g, i] = 1.0
        counts = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.register_buffer("weight", weight / counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class DenseRNN(nn.Module):
    """A plain recurrent core sized to match the connectome's parameter count."""

    def __init__(self, num_edges: int, num_in: int, num_out: int) -> None:
        super().__init__()
        hidden = max(num_in, num_out, int(round(num_edges**0.5)))
        self.hidden = hidden
        self.cell = nn.Linear(hidden, hidden, bias=False)
        self.num_input_nodes = num_in
        self.num_output_nodes = num_out

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, steps or 1, -1)
        steps = x.shape[1]
        h = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(steps):
            drive = torch.zeros_like(h)
            drive[:, : x.shape[2]] = x[:, t, :]
            h = torch.tanh(self.cell(h) + drive) * 0.4 + h * 0.6
            outputs.append(h[:, : self.num_output_nodes])
        return torch.stack(outputs, dim=1)


def build(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
    arm: str,
    seed: int,
    backend: str,
) -> nn.Module:
    torch.manual_seed(seed)
    n_subtypes = len(set(readout_groups))
    if arm == "dense_rnn":
        core = DenseRNN(brain.num_edges, len(input_ids), n_subtypes)
        decoder: nn.Module = nn.Linear(n_subtypes, 2)
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
            input_nodes=input_ids,
            output_nodes=readout_ids,
            weights=weights,
            initializer="normalized_synapse_count",
            leak=0.4,
            backend=backend,
            dtype=torch.float32,
        )
        decoder = nn.Sequential(SubtypeMean(readout_groups), nn.Linear(n_subtypes, 2))
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder)


def make_sweep_batch(
    batch_size: int,
    steps: int,
    input_hex: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of moving Gaussian bumps over the hex columns.

    Direction 0 sweeps low->high hex, direction 1 high->low. Each sample
    starts at a random position and covers only part of the column range, so
    the direction has to come from local motion, not from where the stimulus
    starts or ends. Small observation noise keeps single-step decoding honest.
    """
    lo, hi = float(input_hex.min()), float(input_hex.max())
    full = hi - lo
    speed = full / 36  # one column per step at the reference sweep
    span = speed * (steps - 1)
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    starts = torch.rand(batch_size, generator=generator) * (full - span)
    pos = torch.linspace(0, 1, steps)
    if len(input_hex) == 0:
        raise ValueError("no input neurons")
    hex_t = torch.from_numpy(input_hex.astype(np.float32))
    x = torch.empty(batch_size, steps, len(input_hex))
    for b in range(batch_size):
        s = lo + starts[b].item()
        p0, p1 = (s, s + span) if labels[b] == 0 else (s + span, s)
        center = p0 + (p1 - p0) * pos
        bump = torch.exp(-((hex_t[None, :] - center[:, None]) ** 2) / (2 * sigma**2))
        x[b] = bump + 0.05 * torch.randn(steps, len(input_hex), generator=generator)
    return x, labels


def run_one(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
    input_hex: np.ndarray,
    arm: str,
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict:
    model = build(brain, input_ids, readout_ids, readout_groups, arm, seed, args.backend).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1_000_000 + seed)

    seen, curve = 0, []
    start = time.perf_counter()
    steps_per_epoch = max(1, args.train_samples // args.batch_size)
    for _ in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            x, labels = make_sweep_batch(args.batch_size, args.steps, input_hex, gen, args.sigma)
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
            seen += labels.numel()
            curve.append({"samples": seen, "loss": float(loss)})

    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_sweep_batch(args.batch_size, args.steps, input_hex, gen, args.sigma)
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--backend", type=str, default="metal_csr")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=Path("experiments/results_motion.jsonl"))
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps" if args.backend == "metal_csr" and torch.backends.mps.is_available() else "cpu"
        )
    device = args.device

    circuit, input_ids, readout_ids, input_hex, readout_groups = load_circuit()
    print(
        f"circuit: {circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`"
    )
    print(
        f"inputs: {len(input_ids)} L1, readout: {len(readout_ids)} T4, "
        f"device={device}, backend={args.backend}\n"
    )

    environment = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "connectome_fingerprint": circuit.fingerprint(),
        "nodes": circuit.num_nodes,
        "edges": circuit.num_edges,
        "inputs": len(input_ids),
        "readouts": len(readout_ids),
        "config": {k: str(v) for k, v in vars(args).items()},
    }

    rows = []
    for arm in args.arms:
        for seed in range(args.seeds):
            row = run_one(
                circuit, input_ids, readout_ids, readout_groups, input_hex, arm, seed, args, device
            )
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
        f"# Does the motion circuit help with motion?\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), moving-bar direction.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
