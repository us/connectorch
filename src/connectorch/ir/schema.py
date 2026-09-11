"""Canonical column names, dtypes and version constants for the Connectome IR."""

from __future__ import annotations

import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "NODE_ID",
    "SOURCE_INDEX",
    "TARGET_INDEX",
    "SYNAPSE_COUNT",
    "WEIGHT",
    "SIGN",
    "RESERVED_EDGE_COLUMNS",
    "SUMMED_EDGE_COLUMNS",
    "VALID_SIGNS",
]

#: Bumped whenever the on-disk layout changes in a way old readers cannot handle.
SCHEMA_VERSION = 1

NODE_ID = "node_id"

SOURCE_INDEX = "source_index"
TARGET_INDEX = "target_index"

SYNAPSE_COUNT = "synapse_count"
WEIGHT = "weight"
SIGN = "sign"

#: Edge columns the IR owns and will not treat as free-form metadata.
RESERVED_EDGE_COLUMNS = (SOURCE_INDEX, TARGET_INDEX)

#: Edge columns that are summed when parallel edges are aggregated. Every other
#: column keeps the value of the first edge in canonical order.
SUMMED_EDGE_COLUMNS = (SYNAPSE_COUNT, WEIGHT)

VALID_SIGNS = (-1, 0, 1)

#: Integer kinds acceptable for biological node ids. Floats are rejected because
#: ids above 2**53 lose precision silently, and connectome body ids are large.
_INTEGER_KINDS = ("i", "u")
_STRING_KINDS = ("U", "S", "O")


def is_valid_node_id_dtype(dtype: np.dtype) -> bool:
    """Return whether ``dtype`` may be used for biological node ids."""
    return dtype.kind in _INTEGER_KINDS or dtype.kind in _STRING_KINDS
