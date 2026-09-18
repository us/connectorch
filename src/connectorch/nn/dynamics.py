"""Activation functions available to the recurrent runtime."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["ACTIVATIONS", "resolve_activation"]

#: Activation name -> function. ``identity`` exists so the analytic recurrence
#: tests can check the linear algebra without a nonlinearity in the way.
#: ``threshold_linear`` is the graded FlyVis-style output nonlinearity:
#: silent below threshold, linear above. Combined with a per-neuron bias the
#: threshold becomes learnable per neuron; combined with per-type leak it is
#: the closest this runtime gets to the published fly motion dynamics
#: without spike machinery.
ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "softplus": torch.nn.functional.softplus,
    "threshold_linear": lambda x: torch.clamp(x - 1.0, min=0.0),
    "identity": lambda x: x,
}


def resolve_activation(activation: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Return the activation callable for a name, or pass a callable straight through."""
    if callable(activation):
        return activation
    try:
        return ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(
            f"unknown activation {activation!r}; available: {sorted(ACTIVATIONS)}. "
            "A callable Tensor -> Tensor is also accepted."
        ) from None
