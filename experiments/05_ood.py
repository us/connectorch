"""OOD robustness of the motion wiring: train clean, test shifted.

The strongest cluster in the literature review (plans/004) is robustness:
NCPs lose in-distribution and win under noise; VOneNet gains +18% on
corruption aggregates. Exp 02's in-distribution negative (real 0.7422 vs
random 0.7555) is exactly the NCP pattern, so the honest follow-up is to
test under shift rather than declare defeat.

Protocol: train the FlyVis-style exp-04 model (fixed wiring, per-type
gains, per-type leak, delays, threshold-linear) on clean full-span sweeps,
then evaluate accuracy AND mean T4 DSI under five shifts, each arm at
matched seeds. Metric: drop = clean minus shifted, per seed. A wiring win
is a SMALLER drop, not a higher absolute number.

    python experiments/05_ood.py --arms real random --seeds 2
    PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/05_ood.py --seeds 5 --epochs 3

Writes one JSON line per run (with the full shift table), and a markdown table.
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

import connectorch as ct  # noqa: F401  (backend selection via model code)

_spec = importlib.util.spec_from_file_location("exp04", Path(__file__).parent / "04_selectivity.py")
assert _spec is not None and _spec.loader is not None
e04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e04)

ARMS = ["real", "degree_preserving", "random", "shuffled_weights"]

SHIFTS = ["clean", "contrast_low", "contrast_high", "slow", "fast", "noise", "occlusion", "partial"]


def make_shifted_batch(
    shift: str,
    batch_size: int,
    steps: int,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-span sweep transformed by the named shift (see module docstring)."""
    if shift == "partial":
        return make_partial_batch(batch_size, steps, input_col, input_hex, generator, sigma)
    eff_steps = steps
    if shift == "slow":
        eff_steps = steps * 2
    elif shift == "fast":
        eff_steps = max(4, steps // 2)
    x, labels = e04.make_fullspan_batch(
        batch_size, eff_steps, input_col, input_hex, generator, sigma
    )
    if shift == "contrast_low":
        x = x * 0.5
    elif shift == "contrast_high":
        x = (x * 2.0).clamp_max(2.0)
    elif shift == "noise":
        x = x + 0.15 * torch.randn(x.shape, generator=generator)
    elif shift == "occlusion":
        # Middle third of columns masked: the bump passes behind an occluder.
        col_of_id = torch.from_numpy(np.searchsorted(input_hex, input_col).astype(np.int64))
        ncol = len(input_hex)
        keep = (col_of_id < ncol // 3) | (col_of_id >= 2 * ncol // 3)
        x = x * keep.to(x.dtype)[None, None, :]
    return x, labels


def make_partial_batch(
    batch_size: int,
    steps: int,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    generator: torch.Generator,
    sigma: float = 1.5,
    speed: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exp-02-style partial sweep broadcast to cartridge cells (unseen in training)."""
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


@torch.no_grad()
def eval_shift(
    model: nn.Module,
    shift: str,
    input_col: np.ndarray,
    input_hex: np.ndarray,
    readout_groups: list[int],
    device: str,
    n_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> dict:
    """Accuracy and mean T4 DSI under one shift (fresh generator per shift)."""
    gen = torch.Generator().manual_seed(2_000_000 + seed * 131 + hash(shift) % 10_000)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    losses = []
    # DSI pools
    n_subtypes = len(set(readout_groups))
    groups_t = torch.tensor(readout_groups)
    pools: dict[int, list[list[float]]] = {}
    n_batches = max(1, n_samples // batch_size)
    for _ in range(n_batches):
        x, labels = make_shifted_batch(shift, batch_size, steps, input_col, input_hex, gen)
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        losses.append(float(criterion(logits, labels)))
        correct += int((logits.argmax(1) == labels).sum())
        total += labels.numel()
        traj = model.core(x).cpu()
        pooled = torch.stack(
            [traj[:, :, groups_t == s].mean(dim=2) for s in range(n_subtypes)],
            dim=2,
        ).mean(dim=1)
        for b in range(x.shape[0]):
            for s in range(n_subtypes):
                pools.setdefault(s, []).append([int(labels[b]), float(pooled[b, s])])
    dsi = []
    for s in range(n_subtypes):
        r0 = np.mean([r for lab, r in pools[s] if lab == 0])
        r1 = np.mean([r for lab, r in pools[s] if lab == 1])
        pref, null = (r0, r1) if r0 >= r1 else (r1, r0)
        dsi.append(float((pref - null) / (pref + null + 1e-6)))
    return {
        "accuracy": correct / total,
        "loss": float(np.mean(losses)),
        "mean_dsi": float(np.mean(dsi)),
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
    model = e04.build(brain, input_ids, readout_ids, readout_groups, arm, seed, args.backend)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1_000_000 + seed)

    start = time.perf_counter()
    steps_per_epoch = max(1, args.train_samples // args.batch_size)
    for _ in range(args.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            x, labels = e04.make_fullspan_batch(
                args.batch_size, args.steps, input_col, input_hex, gen, args.sigma
            )
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
    train_seconds = round(time.perf_counter() - start, 1)

    shifts = {}
    for shift in SHIFTS:
        shifts[shift] = eval_shift(
            model,
            shift,
            input_col,
            input_hex,
            readout_groups,
            device,
            args.test_samples,
            args.batch_size,
            args.steps,
            seed,
        )
    clean_acc = shifts["clean"]["accuracy"]
    drops = {shift: round(clean_acc - shifts[shift]["accuracy"], 4) for shift in SHIFTS}
    return {
        "arm": arm,
        "seed": seed,
        "shifts": shifts,
        "accuracy_drops_vs_clean": drops,
        "trainable_parameters": n_params,
        "seconds": train_seconds,
    }


def summarise(rows: list[dict]) -> str:
    header = "| arm | " + " | ".join(SHIFTS) + " |"
    lines = [header, "|" + "|".join(["---"] * (len(SHIFTS) + 1)) + "|"]
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        cells = []
        for shift in SHIFTS:
            acc = np.array([r["shifts"][shift]["accuracy"] for r in runs])
            cells.append(f"{acc.mean():.3f} ± {acc.std(ddof=1) if len(acc) > 1 else 0:.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Accuracy drops vs clean (mean over seeds):")
    lines.append("")
    lines.append("| arm | " + " | ".join(s for s in SHIFTS if s != "clean") + " |")
    lines.append("|" + "|".join(["---"] * len(SHIFTS)) + "|")
    for arm in ARMS:
        runs = [r for r in rows if r["arm"] == arm]
        if not runs:
            continue
        cells = []
        for shift in SHIFTS:
            if shift == "clean":
                continue
            drop = np.array([r["accuracy_drops_vs_clean"][shift] for r in runs])
            cells.append(f"{drop.mean():+.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
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
    parser.add_argument("--out", type=Path, default=Path("experiments/results_ood.jsonl"))
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
    print(f"device={device}, backend={args.backend}\n")

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
            drops = " ".join(
                f"{s}={row['accuracy_drops_vs_clean'][s]:+.3f}" for s in SHIFTS if s != "clean"
            )
            print(
                f"{arm:18} seed {seed}  clean {row['shifts']['clean']['accuracy']:.3f}  "
                f"{drops}  {row['seconds']}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = summarise(rows)
    args.out.with_suffix(".md").write_text(
        f"# Does the motion wiring survive distribution shift?\n\n"
        f"{circuit.num_nodes:,} nodes, {circuit.num_edges:,} connections, "
        f"fingerprint `{circuit.fingerprint()}`. {args.seeds} seeds, "
        f"{args.epochs} epoch(s), train clean full-span, test shifted.\n\n{table}\n"
    )
    print(f"\n{table}\n\nwrote {len(rows)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
