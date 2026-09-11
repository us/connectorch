"""Validation of node and edge tables before they become a :class:`Connectome`.

Every check produces a message that names the offending row and says what to do
about it. An ``IndexError`` from deep inside numpy is never an acceptable way for
a user to learn their edge table is malformed.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import ConnectomeValidationError
from .schema import VALID_SIGNS, is_valid_node_id_dtype

__all__ = [
    "validate_node_ids",
    "validate_edge_endpoints",
    "validate_edge_column",
    "validate_signs",
]

_MAX_LISTED = 5


def _scalar(value: object) -> object:
    """Unwrap a numpy scalar so error messages read ``99`` and not ``np.int64(99)``."""
    return value.item() if isinstance(value, np.generic) else value


def _sample(values: np.ndarray) -> str:
    """Render at most ``_MAX_LISTED`` values for an error message."""
    shown = ", ".join(repr(_scalar(v)) for v in values[:_MAX_LISTED])
    if len(values) > _MAX_LISTED:
        shown += f", ... ({len(values)} total)"
    return shown


def validate_node_ids(node_ids: np.ndarray) -> None:
    """Check that ``node_ids`` can serve as biological identifiers.

    Parameters
    ----------
    node_ids:
        One-dimensional array of candidate identifiers.

    Raises
    ------
    ConnectomeValidationError
        If the ids are the wrong dtype, contain nulls, or contain duplicates.
    """
    if node_ids.ndim != 1:
        raise ConnectomeValidationError(
            f"node ids must be a 1-D array, got shape {node_ids.shape}."
        )
    if node_ids.size == 0:
        raise ConnectomeValidationError(
            "the connectome has no nodes. Pass at least one node, or pass edges "
            "and let the node set be inferred from them."
        )
    if not is_valid_node_id_dtype(node_ids.dtype):
        raise ConnectomeValidationError(
            f"node ids have dtype {node_ids.dtype}, which is not usable as an "
            "identifier. Use an integer or string dtype. Floating-point ids are "
            "rejected because connectome body ids exceed 2**53 and would lose "
            "precision silently; cast with .astype('int64') if the values are "
            "whole numbers."
        )

    if node_ids.dtype.kind in ("U", "S", "O"):
        null_mask = np.array([v is None or v != v for v in node_ids], dtype=bool)
    else:
        null_mask = np.zeros(node_ids.size, dtype=bool)
    if null_mask.any():
        rows = np.flatnonzero(null_mask)
        raise ConnectomeValidationError(
            f"node id rows {_sample(rows)} are null. Every node needs an identifier."
        )

    unique, counts = np.unique(node_ids, return_counts=True)
    duplicated = unique[counts > 1]
    if duplicated.size:
        raise ConnectomeValidationError(
            f"duplicate node ids: {_sample(duplicated)}. Node ids must be unique; "
            "deduplicate the node table before constructing the connectome."
        )


def validate_edge_endpoints(
    source: np.ndarray,
    target: np.ndarray,
    node_ids: np.ndarray,
) -> None:
    """Check that every edge endpoint exists in the node table.

    Parameters
    ----------
    source, target:
        Biological identifiers of the edge endpoints, same length.
    node_ids:
        Sorted array of known node identifiers.

    Raises
    ------
    ConnectomeValidationError
        If lengths disagree, an endpoint is null, or an endpoint is unknown.
    """
    if source.shape != target.shape:
        raise ConnectomeValidationError(
            f"source and target must have the same length, got {source.size} and {target.size}."
        )
    if source.ndim != 1:
        raise ConnectomeValidationError(
            f"edge endpoints must be 1-D arrays, got shape {source.shape}."
        )

    for name, column in (("source", source), ("target", target)):
        if column.dtype.kind == "f":
            nan_rows = np.flatnonzero(np.isnan(column))
            if nan_rows.size:
                raise ConnectomeValidationError(
                    f"edge {name} is NaN on rows {_sample(nan_rows)}. Drop or fix "
                    "those rows before constructing the connectome."
                )
        known = np.isin(column, node_ids)
        if not known.all():
            bad_rows = np.flatnonzero(~known)
            first = bad_rows[0]
            raise ConnectomeValidationError(
                f"edge {int(first)} references {name} node "
                f"{_scalar(column[first])!r}, but that node is not present in the node "
                f"table ({node_ids.size} nodes). "
                f"{bad_rows.size} edge(s) have unknown endpoints. "
                "Hint: pass nodes=None to infer the node set from the edges, or "
                "filter the edge table first."
            )


def validate_edge_column(name: str, values: np.ndarray, num_edges: int) -> None:
    """Check a single edge attribute column.

    Raises
    ------
    ConnectomeValidationError
        If the length is wrong, or a synapse count is negative or non-integral.
    """
    if values.shape[0] != num_edges:
        raise ConnectomeValidationError(
            f"edge column {name!r} has {values.shape[0]} values but there are {num_edges} edges."
        )
    if name in ("weight", "sign") and values.dtype.kind not in "iufb":
        raise ConnectomeValidationError(
            f"edge column {name!r} has dtype {values.dtype}, which is not a number. "
            "The runtime multiplies with this column; it has to be numeric."
        )
    if name == "synapse_count":
        if values.dtype.kind not in "iuf":
            raise ConnectomeValidationError(
                f"synapse_count has dtype {values.dtype}, which is not a number. "
                "Synapse counts are counts; check the column you selected, and "
                "that the file parsed as you expected."
            )
        if values.dtype.kind == "f":
            non_integral = np.flatnonzero(values != np.floor(values))
            if non_integral.size:
                raise ConnectomeValidationError(
                    f"synapse_count is not a whole number on edges "
                    f"{_sample(non_integral)}. Synapse counts are counts."
                )
        negative = np.flatnonzero(values < 0)
        if negative.size:
            raise ConnectomeValidationError(
                f"synapse_count is negative on edges {_sample(negative)}. "
                "Use the 'sign' column to express inhibition, not a negative count."
            )


def validate_signs(signs: np.ndarray) -> None:
    """Check that an edge ``sign`` column holds only -1, 0 or +1."""
    invalid = np.flatnonzero(~np.isin(signs, VALID_SIGNS))
    if invalid.size:
        raise ConnectomeValidationError(
            f"edge sign must be -1, 0 (unknown) or +1; edges {_sample(invalid)} "
            f"have other values (e.g. {_scalar(signs[invalid[0]])!r})."
        )
