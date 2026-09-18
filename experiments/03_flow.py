"""2-D optic flow on the real early-visual circuit, T4+T5 readout.

Exp 02's 1-D sweep was still a toy: one axis, T4-only readout. This goes one
step closer to what the circuit is for: a Gaussian blob translating in one
of four cardinal directions over the 2-D optic-lobe hex layout, injected
into one L1 cell per (hex1, hex2) position, read out from the T4a-d and
T5a-d direction-selective cells pooled by subtype.

    real                    the circuit as reconstructed, weights free to move
    biological              synapse counts held as a prior, a bounded gain on top
    biological_shared       the same, with one gain per pair of cell types
    degree_preserving       every neuron keeps its in- and out-degree, wiring shuffled
    random                  same node and edge count, nothing else preserved
    shuffled_weights        same wiring, synapse counts permuted across connections
    dense_rnn               an ordinary RNN at a matched parameter budget

(hex1, hex2) is treated as Cartesian stimulus space. The real lattice is
hexagonal, so the four directions are stimulus labels, not visual-angle
claims.

    python experiments/03_flow.py --arms real --seeds 1 --epochs 1 --train-samples 256
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/03_flow.py --seeds 5 --epochs 3

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

#: The eight direction-selective readout subtypes, in pool order.
READOUT_SUBTYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

#: Four cardinal directions in (hex1, hex2) stimulus space.
DIRECTIONS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]

ANNOTATION_COLUMNS = {
    "type": "cell_type",
    "superclass": "superclass",
    "somaSide": "side",
    "assignedOlHex1": "hex1",
    "assignedOlHex2": "hex2",
    "status": "status",
}


def load_circuit() -> tuple[ct.Connectome, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Full early-visual circuit plus 2-D L1 input positions and T4+T5 readout.

    Returns the signed circuit, one L1 input node id per unique (hex1, hex2)
    position (lowest body id wins), T4a-d/T5a-d readout node ids, the input
    positions as a [P, 2] float array, and the subtype group index per
    readout id (for subtype mean-pooling).
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
    hex2 = np.array(circuit.nodes.column("hex2").to_pylist())
    ids = np.array(circuit.node_ids)

    l1 = (cell_type == "L1") & (hex1 != "") & (hex2 != "")
    # One representative L1 per hex position: ~2 cells share each of the 892
    # positions, and driving both is redundant. Deterministic: lowest body id.
    seen: dict[tuple[float, float], int] = {}
    order = np.argsort(ids[l1], kind="stable")
    l1_ids, l1_h1, l1_h2 = ids[l1][order], hex1[l1][order], hex2[l1][order]
    for i in range(len(l1_ids)):
        seen.setdefault((float(l1_h1[i]), float(l1_h2[i])), int(l1_ids[i]))
    keys = sorted(seen)
    input_ids = np.array([seen[k] for k in keys])
    input_pos = np.array(keys, dtype=np.float64)

    readout_mask = np.isin(cell_type, READOUT_SUBTYPES)
    readout_ids = ids[readout_mask]
    readout_groups = [READOUT_SUBTYPES.index(c) for c in cell_type[readout_mask]]
    assert len(input_ids) > 0 and len(readout_ids) > 0
    assert circuit.num_edges > 0
    return circuit, input_ids, readout_ids, input_pos, readout_groups


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

    T4a-d/T5a-d are the eight biological direction channels; pooling by
    subtype keeps that signal while making the decoder tiny, so the dense
    control can meet the same parameter budget. No trainable parameters.
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
    """A plain recurrent core sized to match the connectome's parameter count.

    With ~892 inputs a direct drive no longer fits a budget-matched hidden
    state, so inputs arrive through a projection and the hidden size solves
    ``hidden^2 + num_in * hidden ≈ num_edges``.
    """

    def __init__(self, num_edges: int, num_in: int, num_out: int) -> None:
        super().__init__()
        hidden = max(num_out, int(round((np.sqrt(num_in**2 + 4 * num_edges) - num_in) / 2)))
        self.hidden = hidden
        self.projection = nn.Linear(num_in, hidden, bias=False)
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
            h = torch.tanh(self.cell(h) + self.projection(x[:, t, :])) * 0.4 + h * 0.6
            outputs.append(h[:, : self.num_output_nodes])
        return torch.stack(outputs, dim=1)


def build(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
    n_classes: int,
    arm: str,
    seed: int,
    backend: str,
    leak: float,
    readout: str,
) -> nn.Module:
    torch.manual_seed(seed)
    n_subtypes = len(set(readout_groups))
    if arm == "dense_rnn":
        core = DenseRNN(brain.num_edges, len(input_ids), n_subtypes)
        decoder: nn.Module = nn.Linear(n_subtypes, n_classes)
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
            leak=leak,
            backend=backend,
            dtype=torch.float32,
        )
        decoder = nn.Sequential(SubtypeMean(readout_groups), nn.Linear(n_subtypes, n_classes))
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder, readout=readout)


def make_flow_batch(
    batch_size: int,
    steps: int,
    input_pos: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
    speed: float = 1.0,
    noise: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of Gaussian blobs translating in one of four directions.

    Each sample starts at a random position with the whole path inside the
    hex range, so direction has to come from motion, not from where the
    stimulus starts or ends. Small observation noise keeps single-step
    decoding honest.
    """
    lo = input_pos.min(axis=0)
    hi = input_pos.max(axis=0)
    span = speed * (steps - 1)
    labels = torch.randint(0, len(DIRECTIONS), (batch_size,), generator=generator)
    pos_t = torch.linspace(0, 1, steps)
    grid = torch.from_numpy(input_pos.astype(np.float32))
    x = torch.empty(batch_size, steps, len(input_pos))
    for b in range(batch_size):
        direction = torch.tensor(DIRECTIONS[int(labels[b])])
        # Feasible start interval per axis: the whole path must stay inside
        # [lo, hi], i.e. start in [lo - min(reach,0), hi - max(reach,0)].
        reach = direction * span
        lo_t = torch.from_numpy(lo.astype(np.float32))
        hi_t = torch.from_numpy(hi.astype(np.float32))
        feasible_lo = lo_t - torch.clamp(reach, max=0)
        feasible_hi = hi_t - torch.clamp(reach, min=0)
        start = feasible_lo + torch.rand(2, generator=generator) * (
            feasible_hi - feasible_lo
        ).clamp_min(0)
        center = start[None, :] + pos_t[:, None] * reach[None, :]
        dist2 = ((grid[None, :, :] - center[:, None, :]) ** 2).sum(dim=2)
        x[b] = torch.exp(-dist2 / (2 * sigma**2))
        if noise > 0:
            x[b] += noise * torch.randn(steps, len(input_pos), generator=generator)
    return x, labels


def run_one(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
    input_pos: np.ndarray,
    arm: str,
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict:
    model = build(
        brain,
        input_ids,
        readout_ids,
        readout_groups,
        len(DIRECTIONS),
        arm,
        seed,
        args.backend,
        args.leak,
        args.readout,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1_000_000 + seed)

    seen, curve = 0, []
    start = time.perf_counter()
    steps_per_epoch = max(1, args.train_samples // args.batch_size)
    per_step = args.readout == "all"
    for _ in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            x, labels = make_flow_batch(
                args.batch_size, args.steps, input_pos, gen, args.sigma, noise=args.noise
            )
            x, labels = (x * args.input_gain).to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if per_step:
                # Dense supervision at every step: direction is in principle
                # decodable from step 1 on, and the final-step-only gradient
                # vanishes through 12 sparse recurrent steps. Same for all arms.
                loss = criterion(
                    logits.transpose(1, 2),
                    labels.unsqueeze(1).expand(-1, logits.shape[1]),
                )
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            seen += labels.numel()
            curve.append({"samples": seen, "loss": float(loss)})

    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_flow_batch(
                args.batch_size, args.steps, input_pos, gen, args.sigma, noise=args.noise
            )
            x, labels = (x * args.input_gain).to(device), labels.to(device)
            logits = model(x)
            if per_step:
                logits = logits[:, -1, :]
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
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--leak", type=float, default=0.4)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--backend", type=str, default="metal_csr")
    parser.add_argument("--readout", type=str, default="all", choices=("last", "all"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=Path("experiments/results_flow.jsonl"))
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps" if args.backend == "metal_csr" and torch.backends.mps.is_available() else "cpu"
        )
    device = args.device

    circuit, input_ids, readout_ids, input_pos, readout_groups = load_circuit()
    print(
        f"circuit: {circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`"
    )
    print(
        f"inputs: {len(input_ids)} L1 over {len(input_pos)} hex positions, "
        f"readout: {len(readout_ids)} T4+T5, "
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
                circuit,
                input_ids,
                readout_ids,
                readout_groups,
                input_pos,
                arm,
                seed,
                args,
                device,
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
        f"# Does the motion circuit help with 2-D flow?\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), four-direction flow.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
