"""2-D endpoint-free flow with the correct drive, T4+T5 readout.

Exp03 aimed here but drove L1 (inhibitory: the network was exactly silent
past the lamina) and floored. Exp06 proved the mechanism in 1-D
endpoint-free with the cartridge drive (L2/L3/L5, all cholinergic). This
is exp03's ambition with exp06's correct stimulus: a translating blob in
(hex1, hex2) space, random start, whole path inside the range, injected
into all 4,432 cartridge cells over 892 positions, read out from T4a-d +
T5a-d pooled by subtype (8 channels → 4 directions).

    python experiments/07_flow_v2.py --arms real random --seeds 1 --epochs 1 --train-samples 256
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/07_flow_v2.py --seeds 5 --epochs 3

Writes one JSON line per run, and a markdown table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import connectorch as ct

_spec = importlib.util.spec_from_file_location("exp04", Path(__file__).parent / "04_selectivity.py")
assert _spec is not None and _spec.loader is not None
e04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e04)

_spec6 = importlib.util.spec_from_file_location("exp06", Path(__file__).parent / "06_mechanism.py")
assert _spec6 is not None and _spec6.loader is not None
e06 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(e06)

ARMS = ["real", "no_delay", "sign_shuffled", "ei_collapsed", "random"]

READOUT_SUBTYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]

DIRECTIONS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]

CARTRIDGE = ["L2", "L3", "L5"]


def load_circuit() -> tuple[
    ct.Connectome, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]
]:
    """Circuit + cartridge drive over (hex1, hex2) + T4/T5 readout.

    Returns the signed circuit, driven cartridge node ids, their (hex1,
    hex2) positions [P, 2], the sorted unique positions, T4+T5 readout ids,
    and the subtype group per readout id.
    """
    brain = ct.datasets.malecns(
        variant="traced-only",
        min_synapses=5,
        annotations=True,
        annotation_columns=e04.ANNOTATION_COLUMNS,
        neurotransmitters=True,
        download=True,
    )
    from connectorch.transforms import DROSOPHILA_POLARITY, infer_signs

    brain = infer_signs(brain, DROSOPHILA_POLARITY)
    circuit = brain.subgraph(brain.where(cell_type=e04.COLUMNAR))

    cell_type = np.array(circuit.nodes.column("cell_type").to_pylist())
    hex1 = np.array(circuit.nodes.column("hex1").to_pylist())
    hex2 = np.array(circuit.nodes.column("hex2").to_pylist())
    ids = np.array(circuit.node_ids)

    cart = np.isin(cell_type, CARTRIDGE) & (hex1 != "") & (hex2 != "")
    input_ids = ids[cart]
    input_pos = np.stack([hex1[cart].astype(float), hex2[cart].astype(float)], axis=1)
    uniq = np.unique(input_pos, axis=0)
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    positions = uniq[order]

    readout_mask = np.isin(cell_type, READOUT_SUBTYPES)
    readout_ids = ids[readout_mask]
    readout_groups = [READOUT_SUBTYPES.index(c) for c in cell_type[readout_mask]]
    assert len(input_ids) > 0 and len(readout_ids) > 0
    assert circuit.num_edges > 0
    return circuit, input_ids, input_pos, positions, readout_ids, readout_groups


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
    graph = e06.control(brain, arm, seed)
    leak_by = {**{t: e04.DEFAULT_LEAK for t in e04.COLUMNAR}, **e04.LEAK_BY}
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
        leak=e04.DEFAULT_LEAK,
        leak_by=leak_by,
        delay_by=None if arm == "no_delay" else e04.DELAY_BY,
        bias=True,
        backend=backend,
        dtype=torch.float32,
    )
    n_subtypes = len(set(readout_groups))
    decoder = nn.Sequential(e04.SubtypeMean(readout_groups), nn.Linear(n_subtypes, len(DIRECTIONS)))
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder)


def make_cartridge_flow_batch(
    batch_size: int,
    steps: int,
    input_pos: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
    speed: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translating blob in hex space, sampled at each cartridge cell.

    Each sample moves one of four cardinal directions with a random start
    whose whole path fits inside the hex range. Direction must come from
    motion: start/end positions are uniform over directions by construction.
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
    return x, labels


def run_one(
    brain: ct.Connectome,
    input_ids: np.ndarray,
    input_pos: np.ndarray,
    readout_ids: np.ndarray,
    readout_groups: list[int],
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
            x, labels = make_cartridge_flow_batch(
                args.batch_size, args.steps, input_pos, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    losses = []
    conf = np.zeros((len(DIRECTIONS), len(DIRECTIONS)), dtype=np.int64)
    with torch.no_grad():
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_cartridge_flow_batch(
                args.batch_size, args.steps, input_pos, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, labels)))
            pred = logits.argmax(1)
            correct += int((pred == labels).sum())
            total += labels.numel()
            for t, p in zip(labels.cpu().tolist(), pred.cpu().tolist(), strict=True):
                conf[t, p] += 1

    with torch.no_grad():
        silenced_correct = total_sil = 0
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_cartridge_flow_batch(
                args.batch_size, args.steps, input_pos, gen, args.sigma
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
        "confusion": conf.tolist(),
        "silenced_accuracy": silenced_correct / total_sil,
        "trainable_parameters": n_gains,
        "seconds": round(time.perf_counter() - start, 1),
    }


def summarise(rows: list[dict]) -> str:
    lines = [
        "| arm | test accuracy | test loss | silenced acc | trainable params |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        acc = np.array([r["accuracy"] for r in runs])
        loss = np.array([r["test_loss"] for r in runs])
        sil = np.array([r["silenced_accuracy"] for r in runs])
        lines.append(
            f"| `{arm}` | {acc.mean():.4f} ± {acc.std(ddof=1) if len(acc) > 1 else 0:.4f} "
            f"| {loss.mean():.4f} | {sil.mean():.4f} | {runs[0]['trainable_parameters']:,} |"
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
    parser.add_argument("--backend", type=str, default="metal_csr")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=Path("experiments/results_flow_v2.jsonl"))
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps" if args.backend == "metal_csr" and torch.backends.mps.is_available() else "cpu"
        )
    device = args.device

    circuit, input_ids, input_pos, positions, readout_ids, readout_groups = load_circuit()
    print(
        f"circuit: {circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`"
    )
    print(
        f"inputs: {len(input_ids)} cartridge cells over {len(positions)} positions, "
        f"readout: {len(readout_ids)} T4+T5, device={device}, backend={args.backend}\n"
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
                input_pos,
                readout_ids,
                readout_groups,
                arm,
                seed,
                args,
                device,
            )
            row["environment"] = environment
            rows.append(row)
            print(
                f"{arm:18} seed {seed}  acc {row['accuracy']:.4f}  "
                f"loss {row['test_loss']:.4f}  sil {row['silenced_accuracy']:.4f}  "
                f"params {row['trainable_parameters']:,}  {row['seconds']}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = summarise(rows)
    args.out.with_suffix(".md").write_text(
        f"# 2-D endpoint-free flow with the correct drive (T4+T5)\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), four-direction cartridge flow.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
