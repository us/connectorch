"""Direction selectivity emergence, FlyVis-style.

Exps 01-03 asked whether the wiring helps CLASSIFY and got negatives. The
published fly wins never ask that: FlyVis (Nature 2024) fixes the wiring,
trains only per-type parameters on optic flow, and direction selectivity
EMERGES in T4/T5. This experiment copies that recipe as closely as this
runtime allows:

- fixed wiring, count-proportional prior, Dale signs frozen
  (``BiologicalWeights`` shared by cell-type pair: hundreds of params,
  never 262k — the model starts as the measured animal and can only
  modulate it);
- per-type leak (fast L1, slow T4/T5) and heterogeneous synaptic delays
  (slow Mi4/Mi9 flanks vs fast center), the coincidence mechanism;
- graded ``threshold_linear`` output with learnable per-neuron thresholds;
- full-span coherent sweep (the one stimulus our plumbing provably
  carries: 0.70 2-way on ``real`` in diagnostics);
- metric is SELECTIVITY (DSI per T4 subtype), not accuracy alone;
- null set: degree-preserving, random, shuffled-counts wiring, plus a
  T4-silenced readout control (Dhiman-style causality check).

    python experiments/04_selectivity.py --arms real --seeds 1 --epochs 1 --train-samples 256
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/04_selectivity.py --seeds 5 --epochs 3

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

ARMS = ["real", "degree_preserving", "random", "shuffled_weights"]

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

#: The excitatory lamina outputs that actually drive the motion pathway.
#: L1 is glutamatergic and inhibitory in Drosophila (all 1,818 L1->Mi1 edges
#: are sign -1): driving it positive can only ever silence the ON pathway,
#: which is why exps 02-04 drove L1 in vain. L2/L3/L4/L5 are cholinergic;
#: Mi1 is excited by L5+L3, Tm1/Tm2/Tm4 by L2/L4, Tm9 by L3 (traced-only,
#: min_synapses=5). L4 carries no hex coordinates, so the cartridge drive
#: uses L2/L3/L5.
EXCITATORY_LAMINA = ["L2", "L3", "L5"]

READOUT_SUBTYPES = ["T4a", "T4b", "T4c", "T4d"]

ANNOTATION_COLUMNS = {
    "type": "cell_type",
    "superclass": "superclass",
    "somaSide": "side",
    "assignedOlHex1": "hex1",
    "assignedOlHex2": "hex2",
    "status": "status",
}

#: Per-type leak priors, decided upfront (plans/005): L1 follows the
#: stimulus with no memory, medulla integrates briefly, T4 integrates over
#: the delay-line coincidence window. Shared by every arm, so the comparison
#: stays about wiring, not about these numbers.
LEAK_BY = {"L1": 1.0, "T4a": 0.4, "T4b": 0.4, "T4c": 0.4, "T4d": 0.4}
DEFAULT_LEAK = 0.6

#: Synaptic delay priors, decided upfront: the slow OFF flank (Mi4/Mi9)
#: arrives two steps late, Tm9 one step, everything else immediate. This is
#: the Barlow-Levick-style coincidence mechanism, not a tuned knob: null
#: wirings get the identical delays.
DELAY_BY = {"Mi4": 2, "Mi9": 2, "Tm9": 1}


def load_circuit() -> tuple[ct.Connectome, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Circuit plus cartridge drive (L2/L3/L5 per hex column) and T4 readout.

    Returns the signed circuit, the driven lamina node ids, the hex1 column
    of each driven id (the stimulus is built over columns, then broadcast to
    every driven cell in the column), the sorted unique columns, T4 readout
    ids, and the subtype group per readout id.
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

    lamina = np.isin(cell_type, EXCITATORY_LAMINA) & (hex1 != "")
    input_ids = ids[lamina]
    input_col = hex1[lamina].astype(float)
    input_hex = np.array(sorted(set(input_col.tolist())))

    readout_mask = np.isin(cell_type, READOUT_SUBTYPES)
    readout_ids = ids[readout_mask]
    readout_groups = [READOUT_SUBTYPES.index(c) for c in cell_type[readout_mask]]
    assert len(input_ids) > 0 and len(readout_ids) > 0
    assert circuit.num_edges > 0
    return circuit, input_ids, input_col, input_hex, readout_ids, readout_groups


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


class SubtypeMean(nn.Module):
    """Frozen mean-pool over readout neurons grouped by cell type."""

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
    graph = control(brain, arm, seed)
    leak_by = {**{t: DEFAULT_LEAK for t in COLUMNAR}, **LEAK_BY}
    core = ct.nn.ConnectomeRNN(
        graph,
        input_nodes=input_ids,
        output_nodes=readout_ids,
        weights=ct.nn.BiologicalWeights(
            graph,
            gain_bounds=(0.5, 2.0),
            share_by="cell_type",
            dale="sign" in graph.edge_columns,
        ),
        initializer="normalized_synapse_count",
        activation="threshold_linear",
        leak=DEFAULT_LEAK,
        leak_by=leak_by,
        delay_by=DELAY_BY,
        bias=True,
        backend=backend,
        dtype=torch.float32,
    )
    n_subtypes = len(set(readout_groups))
    decoder = nn.Sequential(SubtypeMean(readout_groups), nn.Linear(n_subtypes, 2))
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder)


def make_fullspan_batch(
    batch_size: int,
    steps: int,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coherent full-span sweep, low->high or high->low. Direction 0/1.

    The sweep is built over hex columns, then broadcast to every driven
    lamina cell in each column (one column drives its cartridge's L2/L3/L5).
    """
    lo, hi = float(input_hex.min()), float(input_hex.max())
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    pos = torch.linspace(0, 1, steps)
    hex_t = torch.from_numpy(input_hex.astype(np.float32))
    col_index = torch.from_numpy(np.searchsorted(input_hex, input_col).astype(np.int64))
    x = torch.empty(batch_size, steps, len(input_col))
    for b in range(batch_size):
        p0, p1 = (lo, hi) if labels[b] == 0 else (hi, lo)
        center = p0 + (p1 - p0) * pos
        columns = torch.exp(-((hex_t[None, :] - center[:, None]) ** 2) / (2 * sigma**2))
        x[b] = columns[:, col_index]
    return x, labels


@torch.no_grad()
def direction_selectivity(
    model: nn.Module,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    readout_groups: list[int],
    device: str,
    n_probes: int = 64,
    steps: int = 36,
) -> dict:
    """Mean DSI over T4 subtypes on held-out full-span sweeps.

    DSI_sub = (R_pref - R_null) / (R_pref + R_null + eps) with pref chosen
    post-hoc per subtype, so DSI >= 0 by construction; the null wirings set
    the chance level, and ranking against them is the result.
    """
    model.eval()
    gen = torch.Generator().manual_seed(77)
    responses: dict[int, list[float]] = {0: [], 1: []}
    pools: dict[int, list[list[float]]] = {}
    n_subtypes = len(set(readout_groups))
    groups_t = torch.tensor(readout_groups)
    for _ in range(n_probes // 8):
        x, labels = make_fullspan_batch(8, steps, input_col, input_hex, gen)
        traj = model.core(x.to(device)).cpu()  # [B, T, n_out], no grad
        pooled = torch.stack(
            [traj[:, :, groups_t == s].mean(dim=2) for s in range(n_subtypes)],
            dim=2,
        )  # [B, T, S]
        mean_resp = pooled.mean(dim=1)  # [B, S]
        for b in range(8):
            responses[int(labels[b])].append(float(mean_resp[b].mean()))
            for s in range(n_subtypes):
                pools.setdefault(s, []).append([int(labels[b]), float(mean_resp[b, s])])
    dsi = []
    for s in range(n_subtypes):
        r0 = np.mean([r for lab, r in pools[s] if lab == 0])
        r1 = np.mean([r for lab, r in pools[s] if lab == 1])
        pref, null = (r0, r1) if r0 >= r1 else (r1, r0)
        dsi.append(float((pref - null) / (pref + null + 1e-6)))
    return {
        "mean_dsi": float(np.mean(dsi)),
        "per_subtype_dsi": [round(v, 4) for v in dsi],
        "fraction_selective": float(np.mean([v > 0.2 for v in dsi])),
    }


def run_one(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    input_col: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
    input_hex: np.ndarray,
    arm: str,
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict:
    model = build(brain, input_ids, readout_ids, readout_groups, arm, seed, args.backend)
    model.to(device)

    n_gains = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1_000_000 + seed)

    start = time.perf_counter()
    steps_per_epoch = max(1, args.train_samples // args.batch_size)
    for _ in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            x, labels = make_fullspan_batch(
                args.batch_size, args.steps, input_col, input_hex, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_fullspan_batch(
                args.batch_size, args.steps, input_col, input_hex, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()

    dsi = direction_selectivity(
        model, input_col, input_hex, readout_groups, device, steps=args.steps
    )

    # Dhiman-style causality: silence T4 at readout; the decision must collapse.
    with torch.no_grad():
        silenced_correct = total_sil = 0
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_fullspan_batch(
                args.batch_size, args.steps, input_col, input_hex, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            traj = model.core(x)
            pooled = model.decoder[0](traj[:, -1, :]).zero_()
            logits = model.decoder[1](pooled)
            silenced_correct += int((logits.argmax(1) == labels).sum())
            total_sil += labels.numel()

    return {
        "arm": arm,
        "seed": seed,
        "accuracy": correct / total,
        "test_loss": float(np.mean(losses)),
        "mean_dsi": dsi["mean_dsi"],
        "per_subtype_dsi": dsi["per_subtype_dsi"],
        "fraction_selective": dsi["fraction_selective"],
        "silenced_accuracy": silenced_correct / total_sil,
        "trainable_parameters": n_gains,
        "seconds": round(time.perf_counter() - start, 1),
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| arm | test accuracy | mean T4 DSI | selective fraction | silenced acc | trainable params |",
        "|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        acc = np.array([r["accuracy"] for r in runs])
        dsi = np.array([r["mean_dsi"] for r in runs])
        frac = np.array([r["fraction_selective"] for r in runs])
        sil = np.array([r["silenced_accuracy"] for r in runs])
        lines.append(
            f"| `{arm}` | {acc.mean():.4f} ± {acc.std(ddof=1) if len(acc) > 1 else 0:.4f} "
            f"| {dsi.mean():.4f} ± {dsi.std(ddof=1) if len(dsi) > 1 else 0:.4f} "
            f"| {frac.mean():.2f} | {sil.mean():.4f} | {runs[0]['trainable_parameters']:,} |"
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
    parser.add_argument("--out", type=Path, default=Path("experiments/results_selectivity.jsonl"))
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps" if args.backend == "metal_csr" and torch.backends.mps.is_available() else "cpu"
        )
    device = args.device

    circuit, input_ids, input_col, input_hex, readout_ids, readout_groups = load_circuit()
    print(
        f"circuit: {circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`"
    )
    print(
        f"inputs: {len(input_ids)} L2/L3/L5, readout: {len(readout_ids)} T4, "
        f"device={device}, backend={args.backend}\n"
    )

    environment = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "connectome_fingerprint": circuit.fingerprint(),
        "nodes": circuit.num_nodes,
        "edges": circuit.num_edges,
        "config": {k: str(v) for k, v in vars(args).items()},
    }

    rows = []
    for arm in args.arms:
        for seed in range(args.seeds):
            row = run_one(
                circuit,
                input_ids,
                input_col,
                readout_ids,
                readout_groups,
                input_hex,
                arm,
                seed,
                args,
                device,
            )
            row["environment"] = environment
            rows.append(row)
            print(
                f"{arm:18} seed {seed}  acc {row['accuracy']:.4f}  "
                f"dsi {row['mean_dsi']:.4f}  sil {row['silenced_accuracy']:.4f}  "
                f"params {row['trainable_parameters']:,}  {row['seconds']}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = summarise(rows)
    args.out.with_suffix(".md").write_text(
        f"# Does direction selectivity emerge from the motion wiring?\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), full-span coherent sweep.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
