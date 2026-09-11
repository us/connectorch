"""The dataset-agnostic Connectome intermediate representation."""

from .connectome import Connectome
from .schema import SCHEMA_VERSION

__all__ = ["Connectome", "SCHEMA_VERSION"]
