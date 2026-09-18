"""Mechanism dissection on the working motion circuit (endpoint-free).

Exp 04's positive used full-span sweeps, which carry an endpoint confound:
start/end position alone predicts direction. All arms shared it, so the
ranking stands — but a publishable claim needs the confound gone AND the
mechanism nailed. This experiment does both at once:

- stimulus: partial-span sweeps broadcast to the cartridge (L2/L3/L5 per
  hex column), random start, path strictly inside the range. Direction must
  come from local motion. Same cartridge drive that unlocked exp 04.
- arms: real wiring + delays (replication), real wiring WITHOUT delays
  (delay ablation: coincidence mechanism causal?), sign-shuffled E/I
  (polarity causal?), E/I-collapsed to all +1 (inhibition causal?),
  random wiring (floor reference).
- FlyVis recipe otherwise identical: fixed wiring, count prior, Dale
  frozen, per-type gains only, per-type leak, threshold-linear + bias.

    python experiments/06_mechanism.py --arms real no_delay --seeds 1 --epochs 1 --train-samples 256
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/06_mechanism.py --seeds 5 --epochs 3

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

from connectorch.transforms import collapse_ei, shuffle_signs  # noqa: E402

ARMS = ["real", "no_delay", "sign_shuffled", "ei_collapsed", "random"]


def control(brain: ct.Connectome, arm: str, seed: int) -> ct.Connectome:
    if arm in ("real", "no_delay"):
        return brain
    if arm == "sign_shuffled":
        return shuffle_signs(brain, seed=seed)
    if arm == "ei_collapsed":
        return collapse_ei(brain)
    if arm == "random":
        from connectorch.transforms import random_topology

        return random_topology(brain, seed=seed)
    raise ValueError(arm)


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
    decoder = nn.Sequential(e04.SubtypeMean(readout_groups), nn.Linear(n_subtypes, 2))
    return ct.nn.ConnectomeModel(core, encoder=None, decoder=decoder)


def make_partial_cartridge_batch(
    batch_size: int,
    steps: int,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
    speed: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial-span sweep over columns, broadcast to cartridge cells.

    Each sample starts at a random position with the whole path inside the
    hex range (speed = columns per step at the reference sweep). Start and
    end positions carry no direction information by construction.
    """
    lo, hi = float(input_hex.min()), float(input_hex.max())
    full = hi - lo
    span = speed * (steps - 1)
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    pos = torch.linspace(0, 1, steps)
    col_index = torch.from_numpy(np.searchsorted(input_hex, input_col).astype(np.int64))
    hex_t = torch.from_numpy(input_hex.astype(np.float32))
    x = torch.empty(batch_size, steps, len(input_col))
    for b in range(batch_size):
        s = lo + torch.rand((), generator=generator).item() * (full - span)
        p0, p1 = (s, s + span) if labels[b] == 0 else (s + span, s)
        center = p0 + (p1 - p0) * pos
        columns = torch.exp(-((hex_t[None, :] - center[:, None]) ** 2) / (2 * sigma**2))
        x[b] = columns[:, col_index]
    return x, labels


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
            x, labels = make_partial_cartridge_batch(
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
            x, labels = make_partial_cartridge_batch(
                args.batch_size, args.steps, input_col, input_hex, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, labels)))
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()

    dsi = e04.direction_selectivity(
        model, input_col, input_hex, readout_groups, device, steps=args.steps
    )

    with torch.no_grad():
        silenced_correct = total_sil = 0
        for _ in range(max(1, args.test_samples // args.batch_size)):
            x, labels = make_partial_cartridge_batch(
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
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--backend", type=str, default="metal_csr")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--out", type=Path, default=Path("experiments/results_mechanism.jsonl"))
    args = parser.parse_args()

    if args.device is None:
        args.device = (
            "mps" if args.backend == "metal_csr" and torch.backends.mps.is_available() else "cpu"
        )
    device = args.device

    circuit, input_ids, input_col, input_hex, readout_ids, readout_groups = e04.load_circuit()
    print(
        f"circuit: {circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`"
    )
    print(
        f"inputs: {len(input_ids)} L2/L3/L5 (partial-span, endpoint-free), "
        f"readout: {len(readout_ids)} T4, device={device}, backend={args.backend}\n"
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
        f"# What in the wiring computes direction? (endpoint-free)\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), partial-span cartridge sweeps.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
