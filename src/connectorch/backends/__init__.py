"""Propagation backends: how one recurrent step moves messages along the edges."""

from .base import BACKENDS, Propagator, build_propagator
from .dense import DensePropagator
from .metal_csr import MetalCSRPropagator
from .scatter import ScatterPropagator
from .sparse_mm import SparseMMPropagator

__all__ = [
    "Propagator",
    "build_propagator",
    "BACKENDS",
    "DensePropagator",
    "MetalCSRPropagator",
    "ScatterPropagator",
    "SparseMMPropagator",
]
