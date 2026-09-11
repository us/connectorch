"""Run the Python blocks printed in the README and docs.

A quickstart that does not run is worse than no quickstart: it is the first thing
a new user tries, and it is the first impression the project makes. One review
round found exactly that failure, so the documented code is now executed here
rather than trusted.

Blocks that need the network, a GPU or a large download are skipped by marker: a
line containing ``# doctest: +SKIP``, or any reference to ``download=True``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]

BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Substrings that mark a block as illustrative rather than runnable here.
SKIP_MARKERS = (
    "doctest: +SKIP",
    "download=True",
    "malecns(",
    "torchvision",
    "images",
    "NeuronCriteria",
    "my_connectome",
    "edges.parquet",
    "...",
)


def runnable_blocks(path: Path) -> list[tuple[int, str]]:
    """Blocks in one document, in the order a reader would run them."""
    text = path.read_text()
    blocks = []
    for match in BLOCK.finditer(text):
        code = match.group(1)
        if any(marker in code for marker in SKIP_MARKERS):
            continue
        blocks.append((text[: match.start()].count("\n") + 1, code))
    return blocks


DOCUMENTS = [path for path in SOURCES if path.exists() and runnable_blocks(path)]


def test_there_are_documents_to_check() -> None:
    """Guard against the regex silently matching nothing and the suite passing empty."""
    total = sum(len(runnable_blocks(path)) for path in DOCUMENTS)
    assert total >= 8, f"only found {total} runnable documented blocks"


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
def test_documented_code_runs(path: Path) -> None:
    """Run a document's blocks in order, sharing one namespace.

    Later blocks lean on names the earlier ones defined, exactly as they do for
    someone reading the page top to bottom.
    """
    namespace: dict[str, object] = {"__name__": "__doc_example__"}
    for line, code in runnable_blocks(path):
        try:
            exec(compile(code, f"{path.name}:{line}", "exec"), namespace)  # noqa: S102
        except Exception as error:  # pragma: no cover - the message is the point
            pytest.fail(f"{path.name} line {line} does not run: {type(error).__name__}: {error}")
