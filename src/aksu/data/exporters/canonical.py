"""Export MorphEntry records to canonical TSV format."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path

from aksu.resource.schema import MorphEntry


def to_canonical_tsv(entries: Iterable[MorphEntry], output: Path | str) -> int:
    """Write entries to a tab-separated file: surface\\tcanonical_tags\\tpos\\ttier.

    Returns the number of entries written.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write("surface\tcanonical_tags\tpos\ttier\n")
        for entry in entries:
            fh.write(f"{entry.surface}\t{entry.canonical_tags}\t{entry.pos}\t{entry.tier}\n")
            count += 1
    return count
