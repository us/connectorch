"""Reading connectomes out of the file formats people actually have.

CSV, Parquet and Arrow/Feather all reduce to the same job: pick the two endpoint
columns out of a table, keep the attribute columns that matter, and hand them to
the IR. The only interesting part is not reading a gigabyte of columns nobody asked
for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.feather as feather
import pyarrow.parquet as pq

from ..exceptions import ConnectorchError
from ..ir import Connectome

__all__ = ["read_edge_table", "connectome_from_table"]


def read_edge_table(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    format: str | None = None,
) -> pa.Table:
    """Read a table from CSV, Parquet or Arrow/Feather, projecting only ``columns``.

    Parameters
    ----------
    path:
        File to read.
    columns:
        Column names to keep. ``None`` reads everything, which for a
        gigabyte-scale connectome is usually the wrong choice.
    format:
        ``"csv"``, ``"tsv"``, ``"parquet"`` or ``"feather"``. Inferred from the
        suffix when omitted.
    """
    path = Path(path)
    format = format or _infer_format(path)

    if format == "parquet":
        return pq.read_table(path, columns=columns)
    if format == "feather":
        return feather.read_table(path, columns=columns)
    if format in ("csv", "tsv"):
        options = pa_csv.ParseOptions(delimiter="\t") if format == "tsv" else None
        table = pa_csv.read_csv(path, parse_options=options)
        return table.select(columns) if columns else table
    raise ConnectorchError(
        f"unknown format {format!r} for {path}; use 'csv', 'tsv', 'parquet' or 'feather'."
    )


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix in (".feather", ".arrow", ".ipc"):
        return "feather"
    if suffix == ".tsv":
        return "tsv"
    if suffix in (".csv", ".txt"):
        return "csv"
    raise ConnectorchError(f"cannot tell the format of {path.name} from its suffix; pass format=.")


def connectome_from_table(
    edges: pa.Table,
    *,
    source_column: str,
    target_column: str,
    attributes: dict[str, str] | None = None,
    nodes: pa.Table | None = None,
    node_id_column: str = "node_id",
    node_attributes: dict[str, str] | None = None,
    provenance: dict[str, Any] | None = None,
    aggregate_parallel_edges: bool = True,
) -> Connectome:
    """Build a :class:`~connectorch.ir.Connectome` from Arrow tables.

    Parameters
    ----------
    edges:
        Edge table.
    source_column, target_column:
        Names of the endpoint columns in ``edges``.
    attributes:
        ``{source column name: IR column name}`` for edge attributes to keep, e.g.
        ``{"weight": "synapse_count"}``. Nothing else is carried over.
    nodes:
        Optional node table.
    node_id_column:
        Name of the identifier column in ``nodes``.
    node_attributes:
        ``{source column name: IR column name}`` for node attributes to keep.
        Renaming matters: a connectome's ``class`` and ``type`` columns cannot be
        passed through ``where(class=...)`` in Python, so they are renamed to
        ``cell_class`` and ``cell_type``.
    """
    edge_columns: dict[str, Any] = {
        "source": _column(edges, source_column, "edge"),
        "target": _column(edges, target_column, "edge"),
    }
    for original, renamed in (attributes or {}).items():
        if renamed in edge_columns:
            raise ConnectorchError(
                f"attribute {original!r} would be renamed to {renamed!r}, which is "
                "already the source or target column. Renaming onto an endpoint "
                "would change the graph. Pick another name."
            )
        edge_columns[renamed] = _column(edges, original, "edge")

    node_columns: dict[str, Any] | None = None
    if nodes is not None:
        node_columns = {"node_id": _column(nodes, node_id_column, "node")}
        for original, renamed in (node_attributes or {}).items():
            if renamed in node_columns:
                raise ConnectorchError(
                    f"attribute {original!r} would be renamed to {renamed!r}, which "
                    "is already taken. Pick another name."
                )
            node_columns[renamed] = _column(nodes, original, "node")

    return Connectome(
        nodes=node_columns,
        edges=edge_columns,
        aggregate_parallel_edges=aggregate_parallel_edges,
        provenance=provenance,
    )


def _column(table: pa.Table, name: str, what: str) -> Any:
    if name not in table.column_names:
        raise ConnectorchError(
            f"no {what} column {name!r} in the table; available: {table.column_names}"
        )
    return table.column(name).to_numpy(zero_copy_only=False)
