"""Export MorphEntry records to the tr_gold_morph CoNLL-U schema."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path

from aksu.resource.schema import MorphEntry


def to_conllu(entries: Iterable[MorphEntry], output: Path | str) -> int:
    """Write entries in the tr_gold_morph CoNLL-U schema.

    Each entry produces a single-token sentence with the canonical tags in MISC.

    Returns the number of entries written.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for i, entry in enumerate(entries, 1):
            fh.write(f"# sent_id = morph_{i}\n")
            fh.write(f"# text = {entry.surface}\n")
            fh.write(
                f"1\t{entry.surface}\t{entry.lemma}\t{entry.pos}\t_\t_\t0\troot\t_\t"
                f"Canonical={entry.canonical_tags};Source={entry.source};"
                f"Tier={entry.tier};Conf={entry.confidence:.4f}\n\n"
            )
            count += 1
    return count
