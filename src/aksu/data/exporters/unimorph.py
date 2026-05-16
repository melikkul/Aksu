"""Export MorphEntry records to UniMorph 3-column TSV format."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from aksu.resource.schema import MorphEntry


def to_unimorph_tsv(entries: Iterable[MorphEntry], output: Path | str) -> int:
    """Write entries to UniMorph format: lemma\\tsurface\\ttags.

    Returns the number of entries written.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            # Strip the lemma prefix from canonical_tags to get the tag sequence only
            # e.g. "ev +PLU +POSS.3SG +ABL" → "+PLU+POSS.3SG+ABL"
            tags = entry.canonical_tags
            if tags.startswith(entry.lemma):
                tags = tags[len(entry.lemma):].strip()
            # UniMorph uses semicolons to join tags, remove our + prefix
            unimorph_tags = ";".join(t.lstrip("+") for t in tags.split() if t.startswith("+"))
            fh.write(f"{entry.lemma}\t{entry.surface}\t{unimorph_tags}\n")
            count += 1
    return count
