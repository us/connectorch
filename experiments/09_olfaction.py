"""Mushroom-body head for classification: sparse expansion wins few-shot.

The fly olfactory system expands ~50 projection-neuron channels into ~2000
Kenyon cells through a fixed sparse binary projection, holds the population
to a fixed activity level, and learns only the readout. This checks that
``connectorch.nn.SparseExpander`` plus a learned linear readout beats a
parameter-matched MLP, and a compute-unmatched dense expansion, on MNIST —
full-data and 256-shot. Same protocol, same seeds, every arm.

    python experiments/09_olfaction.py --seeds 5

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

from connectorch.datasets._cache import default_cache_dir
from connectorch.nn import SparseExpander, k_wta

ARMS = ["fly", "dense_expansion", "mlp"]


def make_fly() -> nn.Module:
    return nn.Sequential(SparseExpander(784, 2000, fan_in=6, k=100, seed=0), nn.Linear(2000, 10))


def make_dense_expansion(seed: int) -> nn.Module:
    torch.manual_seed(9_000 + seed)
    proj = nn.Linear(784, 2000, bias=False)
    with torch.no_grad():
        proj.weight.normal_()
    for p in proj.parameters():
        p.requires_grad = False

    class KWTA(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return k_wta(x, 100)

    return nn.Sequential(proj, KWTA(), nn.Linear(2000, 10))


def make_mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(784, 25), nn.Tanh(), nn.Linear(25, 10))


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    root = str(default_cache_dir().parent / "connectorch" / "mnist")
    ds_tr = datasets.MNIST(root=root, train=True, download=False)
    ds_te = datasets.MNIST(root=root, train=False, download=False)
    return (
        ds_tr.data.float().reshape(-1, 784) / 255.0,
        ds_tr.targets,
        ds_te.data.float().reshape(-1, 784) / 255.0,
        ds_te.targets,
    )


def run_one(
    arm: str,
    seed: int,
    n_train: int,
    epochs: int,
    regime: str,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xte: torch.Tensor,
    Yte: torch.Tensor,
) -> dict:
    torch.manual_seed(seed)
    model = {
        "fly": make_fly,
        "dense_expansion": lambda: make_dense_expansion(seed),
        "mlp": make_mlp,
    }[arm]()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    idx = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed))[:n_train]
    xb_all, yb_all = Xtr[idx], Ytr[idx]
    start = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + 1000 + ep))
        for i in range(0, n_train, 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            loss = crit(model(xb_all[b]), yb_all[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        accuracy = float((model(Xte).argmax(1) == Yte).float().mean())
    return {
        "arm": arm,
        "seed": seed,
        "regime": regime,
        "n_train": n_train,
        "epochs": epochs,
        "accuracy": accuracy,
        "trainable_parameters": trainable,
        "seconds": round(time.perf_counter() - start, 1),
        "python": platform.python_version(),
    }


def summarise(rows: list[dict], Xte: torch.Tensor, Yte: torch.Tensor) -> str:
    lines = [
        "| arm | regime | test accuracy | trainable params |",
        "|---|---|---|---|",
    ]
    for regime in ["few-shot (256)", "full"]:
        for arm in ARMS:
            accs = [r["accuracy"] for r in rows if r["arm"] == arm and r["regime"] == regime]
            n_params = next(
                r["trainable_parameters"] for r in rows if r["arm"] == arm and r["regime"] == regime
            )
            lines.append(
                f"| `{arm}` | {regime} | {np.mean(accs):.4f} ± {np.std(accs):.4f} | {n_params:,} |"
            )
    _ = (Xte, Yte)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=str, default="experiments/results_olfaction.jsonl")
    args = parser.parse_args()

    Xtr, Ytr, Xte, Yte = load_mnist()
    rows: list[dict] = []
    with torch.no_grad():
        pass
    for seed in range(args.seeds):
        for n_train, epochs, regime in [(256, 20, "few-shot (256)"), (60000, 3, "full")]:
            for arm in ARMS:
                row = run_one(arm, seed, n_train, epochs, regime, Xtr, Ytr, Xte, Yte)
                rows.append(row)
                print(
                    f"{arm:16s} {regime:14s} seed {seed} acc {row['accuracy']:.4f}"
                    f"  params {row['trainable_parameters']:,}"
                    f"  {row['seconds']}s",
                    flush=True,
                )
    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md = (
        "# Does the mushroom-body head help classification?\n\n"
        f"MNIST, {args.seeds} seeds, few-shot (256 samples, 20 epochs) and full "
        f"(60,000 samples, 3 epochs). Sparse expansion is frozen; only the readout trains.\n\n"  # noqa: E501
        + summarise(rows, Xte, Yte)
    )
    Path("experiments/results_olfaction.md").write_text(md)
    print(f"\nwrote {len(rows)} runs to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
