"""The dataset registry behind ``connectorch datasets list|info``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..exceptions import ConnectorchError
from ..ir import Connectome
from .malecns import DATASET_INFO as _MALECNS_INFO
from .malecns import malecns as _load_malecns
from .malecns import malecns_sample as _load_malecns_sample

__all__ = ["list_datasets", "info", "get"]

_REGISTRY: dict[str, dict[str, Any]] = {
    "male-cns": {**_MALECNS_INFO, "loader": _load_malecns},
    "male-cns-sample": {
        "name": "male-cns-sample",
        "version": "v1.0",
        "description": "Small offline subset of MaleCNS v1.0, bundled with the wheel",
        "license": "CC-BY-4.0",
        "citation": _MALECNS_INFO["citation"],
        "homepage": _MALECNS_INFO["homepage"],
        "loader": _load_malecns_sample,
    },
}


def list_datasets() -> list[str]:
    """Names of the datasets ConnecTorch knows how to load."""
    return sorted(_REGISTRY)


def info(name: str) -> dict[str, Any]:
    """Metadata for one dataset: version, description, license, citation, sizes."""
    try:
        record = _REGISTRY[name]
    except KeyError:
        raise ConnectorchError(f"unknown dataset {name!r}; available: {list_datasets()}") from None
    return {k: v for k, v in record.items() if k != "loader"}


def get(name: str, **kwargs: Any) -> Connectome:
    """Load a dataset by name, forwarding keyword arguments to its loader."""
    try:
        loader: Callable[..., Connectome] = _REGISTRY[name]["loader"]
    except KeyError:
        raise ConnectorchError(f"unknown dataset {name!r}; available: {list_datasets()}") from None
    return loader(**kwargs)
