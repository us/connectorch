"""The Connectome intermediate representation.

A :class:`Connectome` is a validated, deterministically indexed wiring diagram. It
knows nothing about PyTorch: it is the dataset-agnostic middle of the pipeline

``arbitrary connectome -> Connectome IR -> compiler/runtime -> torch.nn.Module``

Biological identifiers are mapped to contiguous integer indices ``0..N-1`` by
sorting, so the same source data always yields the same indices regardless of the
row order it happened to arrive in.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..exceptions import ConnectomeValidationError, ConnectorchError
from .schema import (
    NODE_ID,
    RESERVED_EDGE_COLUMNS,
    SCHEMA_VERSION,
    SIGN,
    SOURCE_INDEX,
    SUMMED_EDGE_COLUMNS,
    SYNAPSE_COUNT,
    TARGET_INDEX,
    WEIGHT,
)
from .validation import (
    validate_edge_column,
    validate_edge_endpoints,
    validate_node_ids,
    validate_signs,
)

__all__ = ["Connectome"]

ArrayLike = Any


def _to_numpy(values: ArrayLike, *, name: str) -> np.ndarray:
    """Coerce a column-ish object to a 1-D numpy array without copying when possible."""
    if isinstance(values, np.ndarray):
        array = values
    elif isinstance(values, pa.Array | pa.ChunkedArray):
        array = values.to_numpy(zero_copy_only=False)
    elif hasattr(values, "to_numpy"):  # pandas Series
        array = values.to_numpy()
    elif isinstance(values, Sequence | Iterable):
        items = list(values)
        array = np.asarray(items)
        # numpy turns [1, "1"] into ["1", "1"], which would merge two different
        # neurons into one. Catch the coercion rather than inherit its result.
        if array.dtype.kind in "US" and any(not isinstance(v, str) for v in items):
            offenders = {type(v).__name__ for v in items}
            raise ConnectomeValidationError(
                f"{name!r} mixes types {sorted(offenders)}. Node identifiers must "
                "be all integers or all strings; numpy would otherwise silently "
                "convert 1 and '1' into the same node."
            )
    else:
        raise ConnectomeValidationError(f"cannot interpret {name!r} of type {type(values)!r}.")
    array = np.asarray(array)
    if array.ndim != 1:
        raise ConnectomeValidationError(f"{name!r} must be 1-D, got shape {array.shape}.")
    return array


def _normalize_ids(values: np.ndarray, *, name: str) -> np.ndarray:
    """Put identifiers into one of exactly two canonical dtypes: int64 or unicode.

    Everything downstream, from ``searchsorted`` to the fingerprint, assumes node
    ids and edge endpoints share a dtype. They do not by default: a ``uint64``
    endpoint compared against ``int64`` node ids makes numpy promote both to
    float64, which silently collapses ids above 2**53 onto each other. Object
    arrays are worse: their raw bytes are pointers, so hashing them gives a
    different answer every process.
    """
    if values.dtype.kind == "u":
        if values.size and int(values.max()) > np.iinfo(np.int64).max:
            raise ConnectomeValidationError(
                f"{name!r} contains an unsigned id above {np.iinfo(np.int64).max}, "
                "which ConnecTorch cannot represent. Biological body ids are far "
                "below this; check the column is really an identifier."
            )
        return values.astype(np.int64)
    if values.dtype.kind == "i":
        return values.astype(np.int64, copy=False)
    if values.dtype.kind == "O":
        # str_ has a stable byte representation; object does not. Check for nulls
        # first: astype would turn None into the perfectly valid id "None".
        if any(v is None or v != v for v in values):
            raise ConnectomeValidationError(
                f"{name!r} contains null identifiers. Every node needs an id; "
                "drop or fill those rows first."
            )
        return values.astype(np.str_)
    if values.dtype.kind in "SU":
        return values.astype(np.str_)
    return values


def _as_columns(table: Any, *, what: str) -> dict[str, np.ndarray]:
    """Normalise a table-ish object into ``{column_name: numpy array}``."""
    if table is None:
        return {}
    if isinstance(table, pa.Table):
        return {name: _to_numpy(table.column(name), name=name) for name in table.column_names}
    if isinstance(table, Mapping):
        return {str(k): _to_numpy(v, name=str(k)) for k, v in table.items()}
    if hasattr(table, "columns") and hasattr(table, "__getitem__"):  # pandas DataFrame
        return {str(c): _to_numpy(table[c], name=str(c)) for c in table.columns}
    raise ConnectomeValidationError(
        f"{what} must be a pyarrow Table, a mapping of columns, or a pandas "
        f"DataFrame; got {type(table)!r}."
    )


def _canonical_column(values: np.ndarray) -> np.ndarray:
    """Give a numeric edge attribute a definite dtype.

    An object array of Python ints round-trips through parquet as int64, so a
    connectome would fail its own fingerprint check after being saved and
    reloaded. Deciding the dtype on the way in makes the in-memory graph and the
    on-disk graph the same graph.
    """
    if values.dtype.kind != "O":
        return values
    if any(v is None or (isinstance(v, float) and v != v) for v in values):
        return values
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return values.astype(bool)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        converted = values.astype(object)
        if all(-(2**63) <= int(v) < 2**64 for v in converted):
            unsigned = any(int(v) > np.iinfo(np.int64).max for v in converted)
            return converted.astype(np.uint64 if unsigned else np.int64)
        raise ConnectomeValidationError(
            "an edge column holds integers outside the 64-bit range and cannot be "
            "stored without losing digits. Convert it to a string column if the "
            "values are identifiers rather than counts."
        )
    if any(isinstance(v, complex) for v in values):
        raise ConnectomeValidationError(
            "an edge column holds complex numbers. Dropping the imaginary part "
            "would change every weight silently; convert the column yourself if "
            "the real part is what you meant."
        )
    if all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values):
        # Mixing ints and floats means the column becomes float64, which silently
        # rounds an integer above 2**53. Refuse rather than round.
        lossy = [v for v in values if isinstance(v, (int, np.integer)) and abs(int(v)) > 2**53]
        if lossy:
            raise ConnectomeValidationError(
                f"an edge column mixes integers and floats, and {len(lossy)} of the "
                "integers are too large to survive the conversion to float "
                "(e.g. {0}). Give the column a single numeric dtype.".format(int(lossy[0]))
            )
        return values.astype(np.float64)
    return values


def _canonical_node_ids(values: np.ndarray) -> np.ndarray:
    """Return the sorted unique node ids that define the index order."""
    return np.unique(values)


class Connectome:
    """A validated wiring diagram with deterministic integer indexing.

    Parameters
    ----------
    nodes:
        Node table containing at least a ``node_id`` column, or ``None`` to infer
        the node set from the edges. Accepts a pyarrow Table, a mapping of column
        name to array, or a pandas DataFrame.
    edges:
        Edge table containing ``source`` and ``target`` columns of biological node
        ids, plus any attribute columns such as ``synapse_count``.
    aggregate_parallel_edges:
        Sum parallel edges (same source and target) into one edge. ``synapse_count``
        and ``weight`` are summed; other attributes keep the first value in
        canonical order. Recorded in :attr:`provenance`.
    provenance:
        Free-form JSON-serialisable record of where this connectome came from.

    Notes
    -----
    Node index order is the ascending sort of the node ids, not the order they
    appear in the input. This makes the compiled graph independent of input row
    order, so two users reading the same file with different filters still agree
    on what index 4211 means.
    """

    def __init__(
        self,
        nodes: Any = None,
        edges: Any = None,
        *,
        aggregate_parallel_edges: bool = True,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        node_columns = _as_columns(nodes, what="nodes")
        edge_columns = _as_columns(edges, what="edges")

        source = edge_columns.pop("source", None)
        target = edge_columns.pop("target", None)
        if source is None or target is None:
            raise ConnectomeValidationError(
                "the edge table needs 'source' and 'target' columns holding "
                "biological node ids. Use Connectome.from_edges(source=..., "
                "target=...) if you have plain arrays."
            )
        clashing = sorted(set(edge_columns) & set(RESERVED_EDGE_COLUMNS))
        if clashing:
            raise ConnectomeValidationError(
                f"edge column(s) {clashing} are reserved: ConnecTorch uses them for "
                "the contiguous endpoint indices it computes from 'source' and "
                "'target'. Rename your column."
            )
        source = _normalize_ids(source, name="source")
        target = _normalize_ids(target, name="target")

        if NODE_ID in node_columns:
            node_ids = _normalize_ids(node_columns.pop(NODE_ID), name=NODE_ID)
            validate_node_ids(node_ids)
            _validate_node_column_lengths(node_columns, node_ids.size)
            order = np.argsort(node_ids, kind="stable")
            node_ids = node_ids[order]
            node_columns = {name: col[order] for name, col in node_columns.items()}
        elif node_columns:
            raise ConnectomeValidationError(
                "the node table has columns "
                f"{sorted(node_columns)} but no 'node_id' column to key them by."
            )
        else:
            node_ids = _canonical_node_ids(np.concatenate([source, target]))
            validate_node_ids(node_ids)

        if node_ids.dtype.kind != source.dtype.kind:
            raise ConnectomeValidationError(
                f"node ids have dtype {node_ids.dtype} but the edge endpoints have "
                f"dtype {source.dtype}. Identifiers must be all integers or all "
                "strings on both tables."
            )
        validate_edge_endpoints(source, target, node_ids)

        source_index = np.searchsorted(node_ids, source).astype(np.int64, copy=False)
        target_index = np.searchsorted(node_ids, target).astype(np.int64, copy=False)

        num_edges = source_index.size
        edge_columns = {name: _canonical_column(values) for name, values in edge_columns.items()}
        for name, values in edge_columns.items():
            validate_edge_column(name, values, num_edges)
        if SIGN in edge_columns:
            validate_signs(edge_columns[SIGN])

        source_index, target_index, edge_columns, n_merged = _canonicalise_edges(
            source_index, target_index, edge_columns, aggregate=aggregate_parallel_edges
        )

        self._node_ids = node_ids
        self._node_columns = {name: np.asarray(col) for name, col in node_columns.items()}
        self._source_index = source_index
        self._target_index = target_index
        self._edge_columns = edge_columns

        record: dict[str, Any] = dict(provenance or {})
        record.setdefault("filters", {})
        record["node_order"] = "sorted"
        record["aggregate_parallel_edges"] = bool(aggregate_parallel_edges)
        if aggregate_parallel_edges and n_merged:
            record["parallel_edges_merged"] = int(n_merged)
        self._provenance = record

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        source: ArrayLike,
        target: ArrayLike,
        *,
        nodes: Any = None,
        aggregate_parallel_edges: bool = True,
        provenance: Mapping[str, Any] | None = None,
        **edge_columns: ArrayLike,
    ) -> Connectome:
        """Build a connectome from parallel source/target arrays.

        Examples
        --------
        >>> brain = Connectome.from_edges(
        ...     source=["A", "B", "C"],
        ...     target=["B", "C", "A"],
        ...     weight=[1.0, 0.5, 0.2],
        ... )
        >>> brain.num_nodes, brain.num_edges
        (3, 3)
        """
        edges: dict[str, Any] = {
            "source": _to_numpy(source, name="source"),
            "target": _to_numpy(target, name="target"),
        }
        for name, values in edge_columns.items():
            if values is not None:
                edges[name] = _to_numpy(values, name=name)
        return cls(
            nodes=nodes,
            edges=edges,
            aggregate_parallel_edges=aggregate_parallel_edges,
            provenance=provenance,
        )

    @classmethod
    def _from_parts(
        cls,
        node_ids: np.ndarray,
        node_columns: dict[str, np.ndarray],
        source_index: np.ndarray,
        target_index: np.ndarray,
        edge_columns: dict[str, np.ndarray],
        provenance: dict[str, Any],
    ) -> Connectome:
        """Construct without re-validating, for internal transforms that preserve invariants."""
        obj = cls.__new__(cls)
        obj._node_ids = node_ids
        obj._node_columns = node_columns
        obj._source_index = source_index
        obj._target_index = target_index
        obj._edge_columns = edge_columns
        obj._provenance = provenance
        return obj

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return int(self._node_ids.size)

    @property
    def num_edges(self) -> int:
        """Number of edges, after parallel-edge aggregation."""
        return int(self._source_index.size)

    @property
    def node_ids(self) -> np.ndarray:
        """Biological node ids, ordered by index. ``node_ids[i]`` is the id of node ``i``."""
        return self._node_ids

    @property
    def nodes(self) -> pa.Table:
        """Node table in index order, with ``node_id`` as the first column."""
        columns = {NODE_ID: self._node_ids, **self._node_columns}
        return pa.table(columns)

    @property
    def edges(self) -> pa.Table:
        """Edge table in canonical order.

        Endpoints are given as contiguous integer indices, not biological ids;
        map them back with :attr:`node_ids` or :meth:`id_of`.
        """
        columns = {
            SOURCE_INDEX: self._source_index,
            TARGET_INDEX: self._target_index,
            **self._edge_columns,
        }
        return pa.table(columns)

    @property
    def edge_index(self) -> np.ndarray:
        """``[2, E]`` array of ``(source_index, target_index)`` in canonical order."""
        return np.stack([self._source_index, self._target_index])

    @property
    def provenance(self) -> dict[str, Any]:
        """Where this connectome came from and what was done to it."""
        return self._provenance

    @property
    def edge_columns(self) -> tuple[str, ...]:
        """Names of the edge attribute columns present."""
        return tuple(self._edge_columns)

    @property
    def node_columns(self) -> tuple[str, ...]:
        """Names of the node attribute columns present, excluding ``node_id``."""
        return tuple(self._node_columns)

    def edge_attribute(self, name: str) -> np.ndarray:
        """Return one edge attribute column in canonical order."""
        try:
            return self._edge_columns[name]
        except KeyError:
            raise KeyError(
                f"edge column {name!r} not present; available: {sorted(self._edge_columns)}"
            ) from None

    # ------------------------------------------------------------------
    # id <-> index
    # ------------------------------------------------------------------

    def index_of(self, node_id: ArrayLike) -> np.ndarray | int:
        """Map biological node id(s) to contiguous index/indices.

        Raises
        ------
        KeyError
            If any id is not in the connectome.
        """
        scalar = np.isscalar(node_id) or (isinstance(node_id, np.generic) and np.ndim(node_id) == 0)
        query = np.atleast_1d(np.asarray(node_id))
        # Compare like with like. A float or uint64 query against int64 ids makes
        # numpy promote both to float64, where 2**53 and 2**53+1 are the same
        # number and a missing id can match its neighbour.
        query, representable = _cast_query_to_id_dtype(query, self._node_ids.dtype)
        position = np.searchsorted(self._node_ids, query)
        position = np.clip(position, 0, self._node_ids.size - 1)
        missing = (self._node_ids[position] != query) | ~representable
        if missing.any():
            raise KeyError(
                f"node id(s) {np.atleast_1d(np.asarray(node_id))[missing][:5].tolist()} "
                f"are not in this "
                f"connectome ({self.num_nodes} nodes)."
            )
        return int(position[0]) if scalar else position.astype(np.int64)

    def id_of(self, index: ArrayLike) -> np.ndarray | Any:
        """Map contiguous index/indices back to biological node id(s)."""
        return self._node_ids[index]

    # ------------------------------------------------------------------
    # queries and transforms
    # ------------------------------------------------------------------

    def where(self, **conditions: Any) -> np.ndarray:
        """Return the node ids whose attributes match every condition.

        A scalar value tests equality; a list, tuple, set or array tests membership.

        Examples
        --------
        >>> brain.where(cell_class="descending")            # doctest: +SKIP
        >>> brain.where(region="ME_R", side="R")            # doctest: +SKIP
        >>> brain.where(type=["DNge104", "DNp01"])          # doctest: +SKIP
        """
        mask = np.ones(self.num_nodes, dtype=bool)
        for name, wanted in conditions.items():
            if name == NODE_ID:
                column = self._node_ids
            elif name in self._node_columns:
                column = self._node_columns[name]
            else:
                raise KeyError(
                    f"no node column {name!r}; available: {sorted((NODE_ID, *self._node_columns))}"
                )
            if isinstance(wanted, list | tuple | set | frozenset | np.ndarray):
                mask &= np.isin(column, np.asarray(list(wanted)))
            else:
                mask &= column == wanted
        return self._node_ids[mask]

    def subgraph(self, node_ids: ArrayLike) -> Connectome:
        """Return the subgraph induced on ``node_ids``, reindexed from zero.

        Edges with an endpoint outside the selection are dropped, and the count of
        dropped edges is recorded in provenance.
        """
        keep_ids = np.unique(np.asarray(node_ids))
        keep_index = self.index_of(keep_ids)
        keep_index = np.atleast_1d(np.asarray(keep_index, dtype=np.int64))

        remap = np.full(self.num_nodes, -1, dtype=np.int64)
        remap[keep_index] = np.arange(keep_index.size, dtype=np.int64)

        edge_mask = (remap[self._source_index] >= 0) & (remap[self._target_index] >= 0)
        new_source = remap[self._source_index[edge_mask]]
        new_target = remap[self._target_index[edge_mask]]

        provenance = _extend_provenance(
            self._provenance,
            {
                "op": "subgraph",
                "nodes_kept": int(keep_index.size),
                "nodes_dropped": int(self.num_nodes - keep_index.size),
                "edges_dropped": int(self.num_edges - int(edge_mask.sum())),
            },
        )
        return Connectome._from_parts(
            node_ids=self._node_ids[keep_index],
            node_columns={name: col[keep_index] for name, col in self._node_columns.items()},
            source_index=new_source,
            target_index=new_target,
            edge_columns={name: col[edge_mask] for name, col in self._edge_columns.items()},
            provenance=provenance,
        )

    def filter_nodes(self, **conditions: Any) -> Connectome:
        """Keep only nodes matching the conditions, as :meth:`where` understands them."""
        return self.subgraph(self.where(**conditions))

    def filter_edges(
        self,
        *,
        min_synapses: int | None = None,
        min_weight: float | None = None,
        drop_self_loops: bool = False,
    ) -> Connectome:
        """Keep only edges passing the thresholds. Nodes are never removed.

        Self-loops survive unless ``drop_self_loops=True`` is passed explicitly;
        they are biologically real and dropping them silently would be a lie.
        """
        mask = np.ones(self.num_edges, dtype=bool)
        applied: dict[str, Any] = {"op": "filter_edges"}
        if min_synapses is not None:
            mask &= self.edge_attribute(SYNAPSE_COUNT) >= min_synapses
            applied["min_synapses"] = int(min_synapses)
        if min_weight is not None:
            mask &= self.edge_attribute(WEIGHT) >= min_weight
            applied["min_weight"] = float(min_weight)
        if drop_self_loops:
            mask &= self._source_index != self._target_index
            applied["drop_self_loops"] = True
        applied["edges_dropped"] = int(self.num_edges - int(mask.sum()))

        return Connectome._from_parts(
            node_ids=self._node_ids,
            node_columns=dict(self._node_columns),
            source_index=self._source_index[mask],
            target_index=self._target_index[mask],
            edge_columns={name: col[mask] for name, col in self._edge_columns.items()},
            provenance=_extend_provenance(self._provenance, applied),
        )

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    def content_hash(self) -> str:
        """Hash every byte of the tables, not just the structural columns.

        :meth:`fingerprint` deliberately covers only what the runtime reads, so
        two graphs that compute identically share one. That makes it the wrong
        tool for detecting a corrupted or edited file, where a changed
        ``cell_type`` matters as much as a changed weight. This covers the whole
        of both tables.
        """
        digest = hashlib.sha256()
        digest.update(f"connectorch-content-v{SCHEMA_VERSION}".encode())
        for table in (self.nodes, self.edges):
            for name in table.column_names:
                digest.update(name.encode())
                column = table.column(name)
                digest.update(str(column.type).encode())
                for buffer in column.combine_chunks().buffers():
                    digest.update(b"" if buffer is None else buffer.to_pybytes())
        return digest.hexdigest()[:16]

    def fingerprint(self) -> str:
        """Return a stable 16-character hash of the graph's *structure*.

        Covers the schema version, node ids, edge endpoints, and the ``weight``,
        ``synapse_count`` and ``sign`` columns: everything the runtime reads. A
        connectome whose signs were flipped from excitatory to inhibitory is a
        different graph and fingerprints as one. Free-text provenance (local paths, timestamps)
        is deliberately excluded, so re-downloading a dataset to a different
        directory fingerprints identically.

        String ids are hashed as UTF-8 text rather than as raw array bytes.
        Hashing the bytes of a fixed-width or object array would fold in padding
        and, for object arrays, machine pointers, so the same graph would
        fingerprint differently after a save and reload.
        """
        digest = hashlib.sha256()
        digest.update(f"connectorch-v{SCHEMA_VERSION}".encode())
        digest.update(_identifier_bytes(self._node_ids))
        digest.update(b"|edges|")
        digest.update(np.ascontiguousarray(self._source_index).tobytes())
        digest.update(np.ascontiguousarray(self._target_index).tobytes())
        for name in (*SUMMED_EDGE_COLUMNS, SIGN):
            if name in self._edge_columns:
                column = self._edge_columns[name]
                digest.update(name.encode())
                # Hash integers as integers. Casting a synapse count to float64
                # first makes 2**53 and 2**53+1 hash identically, so the integrity
                # check would not notice one turning into the other.
                digest.update(column.dtype.kind.encode())
                if column.dtype.kind in "iu":
                    widened = column.astype(np.uint64 if column.dtype.kind == "u" else np.int64)
                else:
                    widened = column.astype(np.float64)
                digest.update(np.ascontiguousarray(widened).tobytes())
        digest.update(str(self._provenance.get("node_order", "sorted")).encode())
        return digest.hexdigest()[:16]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the connectome as a ``.ct`` directory of parquet files plus metadata.

        The layout is deliberately inspectable without ConnecTorch::

            brain.ct/
                metadata.json
                nodes.parquet
                edges.parquet

        No pickle is used anywhere.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.nodes, directory / "nodes.parquet", compression="zstd")
        pq.write_table(self.edges, directory / "edges.parquet", compression="zstd")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "fingerprint": self.fingerprint(),
            "content_hash": self.content_hash(),
            "provenance": self._provenance,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        return directory

    @classmethod
    def load(cls, path: str | Path) -> Connectome:
        """Read a connectome previously written by :meth:`save`."""
        directory = Path(path)
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ConnectorchError(
                f"{directory} is not a ConnecTorch dataset directory (no metadata.json)."
            )
        metadata = json.loads(metadata_path.read_text())
        version = metadata.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ConnectorchError(
                f"{directory} was written with schema version {version}, but this "
                f"ConnecTorch reads version {SCHEMA_VERSION}."
            )

        node_table = pq.read_table(directory / "nodes.parquet")
        edge_table = pq.read_table(directory / "edges.parquet")
        node_columns = {
            name: _to_numpy(node_table.column(name), name=name) for name in node_table.column_names
        }
        node_ids = node_columns.pop(NODE_ID)
        edge_columns = {
            name: _to_numpy(edge_table.column(name), name=name) for name in edge_table.column_names
        }
        source_index = edge_columns.pop(SOURCE_INDEX).astype(np.int64, copy=False)
        target_index = edge_columns.pop(TARGET_INDEX).astype(np.int64, copy=False)

        obj = cls._from_parts(
            node_ids=node_ids,
            node_columns=node_columns,
            source_index=source_index,
            target_index=target_index,
            edge_columns=edge_columns,
            provenance=metadata.get("provenance", {}),
        )
        expected_content = metadata.get("content_hash")
        if expected_content and obj.content_hash() != expected_content:
            raise ConnectorchError(
                f"{directory} failed its integrity check: metadata records content "
                f"hash {expected_content}, the files on disk hash to "
                f"{obj.content_hash()}. They have been modified or truncated."
            )
        expected = metadata.get("fingerprint")
        if expected and obj.fingerprint() != expected:
            raise ConnectorchError(
                f"{directory} failed its structural check: metadata records "
                f"fingerprint {expected}, loaded data hashes to "
                f"{obj.fingerprint()}. The graph's nodes, edges or weights differ "
                "from what was written."
            )
        return obj

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        source = self._provenance.get("dataset") or self._provenance.get("source")
        parts = [
            f"nodes={self.num_nodes:,}",
            f"edges={self.num_edges:,}",
            "directed=True",
        ]
        if source:
            parts.append(f'source="{source}"')
        joined = ",\n    ".join(parts)
        return f"Connectome(\n    {joined}\n)"


def _canonicalise_edges(
    source_index: np.ndarray,
    target_index: np.ndarray,
    edge_columns: dict[str, np.ndarray],
    *,
    aggregate: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int]:
    """Sort edges into COO-coalesced order and optionally merge parallel edges.

    Canonical order is ascending ``(target_index, source_index)``, which is exactly
    the order ``torch.sparse_coo_tensor(...).coalesce()`` produces for an adjacency
    indexed as ``A[target, source]``. Sorting here, once, means the runtime never
    has to call ``coalesce()`` on a tensor backed by a trainable Parameter, where
    the value permutation would silently misalign weights with edges.

    Returns
    -------
    source_index, target_index, edge_columns, num_merged
    """
    num_edges = source_index.size
    if num_edges == 0:
        return source_index, target_index, edge_columns, 0

    order = np.lexsort((*_tiebreakers(edge_columns), source_index, target_index))
    source_index = source_index[order]
    target_index = target_index[order]
    edge_columns = {name: col[order] for name, col in edge_columns.items()}

    if not aggregate:
        return source_index, target_index, edge_columns, 0

    is_new = np.empty(num_edges, dtype=bool)
    is_new[0] = True
    is_new[1:] = (source_index[1:] != source_index[:-1]) | (target_index[1:] != target_index[:-1])
    group = np.cumsum(is_new) - 1
    num_groups = int(group[-1]) + 1
    if num_groups == num_edges:
        return source_index, target_index, edge_columns, 0

    first = np.flatnonzero(is_new)
    merged_columns: dict[str, np.ndarray] = {}
    for name, column in edge_columns.items():
        if name in SUMMED_EDGE_COLUMNS and column.dtype.kind in "biuf":
            merged_columns[name] = _group_sums(column, first)
        else:
            merged_columns[name] = column[first]

    return (
        source_index[first],
        target_index[first],
        merged_columns,
        num_edges - num_groups,
    )


def _identifier_bytes(ids: np.ndarray) -> bytes:
    """Hashable bytes for an identifier array, stable across processes and reloads."""
    if ids.dtype.kind in "OUS":
        # Length-prefix each id rather than joining on a separator: any separator
        # can itself appear inside an id, and "a" + sep + "b\0c" would then hash
        # the same as "a\0b" + sep + "c". The leading tag keeps a graph of empty
        # strings from hashing like a graph of zeros.
        parts = bytearray(b"str:")
        for value in ids:
            encoded = str(value).encode("utf-8")
            parts += len(encoded).to_bytes(8, "little") + encoded
        return bytes(parts)
    return b"int:" + np.ascontiguousarray(ids.astype(np.int64)).tobytes()


def _cast_query_to_id_dtype(query: np.ndarray, id_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Bring a lookup key into the connectome's identifier dtype, losslessly.

    Returns the cast query and a mask of which entries an identifier could
    represent at all. A float that is not a whole number, a NaN, an infinity, or
    an unsigned value past the int64 range names no node; each is reported
    missing rather than rounded onto a neighbour. The mask is returned separately
    because any in-band sentinel could itself be somebody's node id.
    """
    if id_dtype.kind in "US":
        return query.astype(np.str_), np.ones(query.shape, dtype=bool)
    if query.dtype.kind == "f":
        representable = np.isfinite(query) & (query == np.floor(query))
        representable &= np.abs(query) <= float(2**53)
        return np.where(representable, query, 0.0).astype(np.int64), representable
    if query.dtype.kind == "u":
        representable = query <= np.uint64(np.iinfo(np.int64).max)
        return np.where(representable, query, 0).astype(np.int64), representable
    if query.dtype.kind == "i":
        return query.astype(np.int64), np.ones(query.shape, dtype=bool)
    return query, np.ones(query.shape, dtype=bool)


def _group_sums(column: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Sum each contiguous group of a sorted column without losing anything.

    Two obvious implementations are both wrong. ``np.bincount(group, weights=...)``
    forces everything through float64, so an int64 synapse count above 2**53 loses
    its last digits, and casting back to the input dtype wraps around, turning two
    int8 counts of 100 into -56. Differencing a global prefix sum is exact for
    integers but catastrophic for floats: in ``[1e16, 1e16, 1.0]`` the running
    total stops changing, and the last edge's weight subtracts to zero.

    ``np.add.reduceat`` sums each group on its own, which is exact for both. The
    input is widened first, because the sum of two counts may not fit where one
    count did, and the widened dtype is kept rather than narrowed back.
    """
    kind = column.dtype.kind
    if kind == "b":
        # Two parallel edges both marked present are two connections, not one.
        widened = column.astype(np.int64)
    elif kind == "u":
        widened = column.astype(np.uint64)
    elif kind == "i":
        widened = column.astype(np.int64)
    else:
        widened = column.astype(np.float64)
    return np.add.reduceat(widened, first)


def _tiebreakers(edge_columns: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Sort keys that make the order *within* a group of parallel edges deterministic.

    Without these, two readings of the same edge set in different row orders can
    disagree: floating-point sums are not associative, so aggregating in a
    different order changes the result in the last bits, and a non-summed column
    would keep a different "first" value. Sorting on the attribute values
    themselves removes the dependence on input row order entirely.

    Object-dtype columns are skipped because ``np.lexsort`` cannot order them.
    """
    keys = []
    for name in sorted(edge_columns):
        column = edge_columns[name]
        if column.dtype.kind in "biufUS":
            keys.append(column)
        elif column.dtype.kind == "O":
            # np.lexsort cannot order object arrays, but leaving them out puts the
            # order back at the mercy of the input rows. Sorting on their string
            # form costs a pass and restores determinism.
            keys.append(column.astype(np.str_))
    return keys


def _validate_node_column_lengths(columns: Mapping[str, np.ndarray], num_nodes: int) -> None:
    """Reject a node table whose attribute columns disagree with its id column.

    Zipping mismatched columns would quietly drop the overhang, which means losing
    real annotations without a word about it.
    """
    for name, column in columns.items():
        if column.shape[0] != num_nodes:
            raise ConnectomeValidationError(
                f"node column {name!r} has {column.shape[0]} values but there are "
                f"{num_nodes} node ids."
            )


def _extend_provenance(base: Mapping[str, Any], applied: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``applied`` appended to its filter history."""
    record = dict(base)
    history = list(record.get("history", []))
    history.append(dict(applied))
    record["history"] = history
    return record
