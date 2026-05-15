"""Export MorphEntry records to surface-level UD CoNLL-U format."""
from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path

from aksu.resource.schema import MorphEntry

# Minimal UD POS tag mapping from our canonical POS names
_POS_TO_UPOS = {
    "NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJ", "ADV": "ADV",
    "PRON": "PRON", "DET": "DET", "NUM": "NUM", "ADP": "ADP",
    "CONJ": "CCONJ", "PUNCT": "PUNCT", "PROPN": "PROPN", "INTJ": "INTJ",
}


def to_ud_conllu(entries: Iterable[MorphEntry], output: Path | str) -> int:
    """Write entries to a surface-level UD CoNLL-U file (one token per sentence).

    Returns the number of entries written.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for i, entry in enumerate(entries, 1):
            upos = _POS_TO_UPOS.get(entry.pos.upper(), "X")
            fh.write(f"# sent_id = entry_{i}\n")
            fh.write(f"# text = {entry.surface}\n")
            # ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            fh.write(
                f"1\t{entry.surface}\t{entry.lemma}\t{upos}\t_\t_\t0\troot\t_\t"
                f"Canonical={entry.canonical_tags.replace(' ', '|')};Tier={entry.tier}\n\n"
            )
            count += 1
    return count
