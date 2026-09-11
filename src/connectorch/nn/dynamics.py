"""Activation functions available to the recurrent runtime."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["ACTIVATIONS", "resolve_activation"]

#: Activation name -> function. ``identity`` exists so the analytic recurrence
#: tests can check the linear algebra without a nonlinearity in the way.
ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "softplus": torch.nn.functional.softplus,
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
