"""Adapters that turn other people's file formats into a Connectome."""

from .networkx_io import from_networkx, to_networkx
from .neuprint import from_neuprint
from .tabular import connectome_from_table, read_edge_table

__all__ = [
    "read_edge_table",
    "connectome_from_table",
    "from_networkx",
    "to_networkx",
    "from_neuprint",
]
