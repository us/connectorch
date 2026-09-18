"""Sparse expansion plus winner-take-all: the mushroom-body motif as a module.

The fly olfactory system expands ~50 projection-neuron channels into ~2000
Kenyon cells through a fixed sparse binary projection, then an anterior
paired lateral neuron holds the population to a fixed sparsity (k winners).
Dasgupta, Stevens and Navlakha (Science 2017) showed this motif matches
dense random projections at ~20x fewer operations and beats classic LSH
~2.8x at matched cost; BioHash (ICML 2020) showed motif plus learned
plasticity beats motif alone.

This module is the motif: a frozen sparse binary projection followed by
top-k binarisation. No learning happens inside it; what wraps it learns.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SparseExpander", "k_wta"]


def k_wta(x: Tensor, k: int) -> Tensor:
    """Keep the top-k values of each row, zero the rest (ties broken arbitrarily)."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if k > x.shape[-1]:
        raise ValueError(f"k={k} exceeds the last dimension {x.shape[-1]}.")
    top, _ = torch.topk(x, k, dim=-1)
    threshold = top[..., -1].unsqueeze(-1)
    return torch.where(x >= threshold, x, torch.zeros((), dtype=x.dtype, device=x.device))


class SparseExpander(nn.Module):
    """Fixed sparse binary fan-in plus k-winner-take-all.

    Parameters
    ----------
    in_features:
        Width of the incoming vector (projection-neuron channels).
    out_features:
        Width of the expanded code (Kenyon cells).
    fan_in:
        How many inputs each output cell samples, without replacement.
    k:
        How many outputs stay active per row.
    seed:
        Seed for the fixed sampling. The projection is frozen: the same seed
        always builds the same wiring, and there is nothing to train inside.

    The projection sums ``fan_in`` binary-weighted inputs per output cell, so
    with graded inputs the scale grows with ``fan_in``; the k-WTA that follows
    only cares about the ranking, and a downstream linear layer absorbs scale.
    """

    projection: Tensor

    def __init__(
        self, in_features: int, out_features: int, fan_in: int = 6, k: int = 100, seed: int = 0
    ) -> None:
        super().__init__()
        if not 1 <= fan_in <= in_features:
            raise ValueError(f"fan_in must be in [1, {in_features}], got {fan_in}.")
        if not 1 <= k <= out_features:
            raise ValueError(f"k must be in [1, {out_features}], got {k}.")
        generator = torch.Generator().manual_seed(seed)
        projection = torch.zeros(out_features, in_features)
        for row in range(out_features):
            choice = torch.randperm(in_features, generator=generator)[:fan_in]
            projection[row, choice] = 1.0
        self.register_buffer("projection", projection, persistent=False)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fan_in = int(fan_in)
        self.k = int(k)

    def forward(self, x: Tensor) -> Tensor:
        """Expand ``[..., in_features]`` to a k-sparse ``[..., out_features]`` code."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {x.shape[-1]}.")
        return k_wta(x @ self.projection.t(), self.k)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, fan_in={self.fan_in}, k={self.k}"
