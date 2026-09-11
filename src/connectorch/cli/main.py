"""The ``connectorch`` command line.

Deliberately small. The Python API is the product; this exists so you can look at a
dataset without opening an interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import __version__
from ..exceptions import ConnectorchError
from ..ir import Connectome

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="connectorch",
        description="Compile biological connectomes into trainable PyTorch networks.",
    )
    parser.add_argument("--version", action="version", version=f"connectorch {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("info", help="describe a saved .ct connectome")
    show.add_argument("path", type=Path)

    check = commands.add_parser(
        "validate", help="load a table and report whether it is a valid connectome"
    )
    check.add_argument("path", type=Path)
    check.add_argument("--source-column", default="source")
    check.add_argument("--target-column", default="target")

    datasets = commands.add_parser("datasets", help="list and fetch built-in datasets")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    dataset_commands.add_parser("list", help="names of the known datasets")
    dataset_info = dataset_commands.add_parser("info", help="metadata for one dataset")
    dataset_info.add_argument("name")
    dataset_download = dataset_commands.add_parser("download", help="fetch a dataset's files")
    dataset_download.add_argument("name")
    dataset_download.add_argument("--variant", default="traced-only")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except ConnectorchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "info":
        return _info(args.path)
    if args.command == "validate":
        return _validate(args.path, args.source_column, args.target_column)
    if args.command == "datasets":
        return _datasets(args)
    raise ConnectorchError(f"unhandled command {args.command!r}")


def _info(path: Path) -> int:
    brain = Connectome.load(path)
    provenance = brain.provenance
    print(f"Dataset:     {provenance.get('dataset', provenance.get('source', 'unknown'))}")
    print(f"Nodes:       {brain.num_nodes:,}")
    print(f"Edges:       {brain.num_edges:,}")
    print("Directed:    yes")
    print(f"Edge columns: {', '.join(brain.edge_columns) or 'none'}")
    print(f"Node columns: {', '.join(brain.node_columns) or 'none'}")
    print(f"License:     {provenance.get('license', 'unknown')}")
    print(f"Fingerprint: {brain.fingerprint()}")
    return 0


def _validate(path: Path, source_column: str, target_column: str) -> int:
    from ..io.tabular import connectome_from_table, read_edge_table

    table = read_edge_table(path)
    # Carry every other column through as an edge attribute. Validating only the
    # endpoints would call a table with negative synapse counts valid.
    attributes = {
        name: name for name in table.column_names if name not in (source_column, target_column)
    }
    brain = connectome_from_table(
        table,
        source_column=source_column,
        target_column=target_column,
        attributes=attributes,
    )
    print(f"ok: {brain.num_nodes:,} nodes, {brain.num_edges:,} edges")
    if brain.provenance.get("parallel_edges_merged"):
        print(f"note: merged {brain.provenance['parallel_edges_merged']:,} parallel edges")
    return 0


def _datasets(args: argparse.Namespace) -> int:
    from ..datasets import registry

    if args.dataset_command == "list":
        for name in registry.list_datasets():
            record = registry.info(name)
            print(f"{name:18} {record['version']:6} {record['description']}")
        return 0

    if args.dataset_command == "info":
        print(json.dumps(registry.info(args.name), indent=2, default=str))
        return 0

    if args.dataset_command == "download":
        if args.name != "male-cns":
            raise ConnectorchError(
                f"{args.name} has nothing to download; it is bundled or synthetic."
            )
        from ..datasets.malecns import malecns

        brain = malecns(variant=args.variant, download=True)
        print(f"{brain.num_nodes:,} nodes, {brain.num_edges:,} edges")
        return 0

    raise ConnectorchError(f"unhandled datasets subcommand {args.dataset_command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
