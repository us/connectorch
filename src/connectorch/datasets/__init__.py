"""Ready-made connectomes."""

from .elegans import elegans
from .malecns import DATASET_INFO, VARIANTS, malecns, malecns_sample
from .registry import get, info, list_datasets
from .toy import chain, random_sparse, ring

__all__ = [
    "elegans",
    "malecns",
    "malecns_sample",
    "VARIANTS",
    "DATASET_INFO",
    "list_datasets",
    "info",
    "get",
    "chain",
    "ring",
    "random_sparse",
]
