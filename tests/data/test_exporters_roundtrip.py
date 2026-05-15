"""Round-trip tests for the four exporter formats (C-Step 6)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from aksu.resource.schema import MorphEntry
from aksu.data.exporters import (
    to_canonical_tsv,
    to_conllu,
    to_ud_conllu,
    to_unimorph_tsv,
)

# Minimal in-memory fixture — 5 representative entries
_FIXTURE: list[MorphEntry] = [
    MorphEntry("evlerinden", "ev", "ev +Noun +PLU +POSS.3SG +ABL", "NOUN", "boun", 0.98, 12, "gold"),
    MorphEntry("gidiyordum",  "gitmek", "gitmek +Verb +PROG +PAST +1SG", "VERB", "zeyrek", 0.87, 5, "silver"),
    MorphEntry("güzel",       "güzel",  "güzel +Adj", "ADJ", "boun", 0.95, 88, "gold"),
    MorphEntry("çocukların",  "çocuk",  "çocuk +Noun +PLU +GEN", "NOUN", "imst", 0.75, 20, "bronze"),
    MorphEntry("yavaşça",     "yavaşça", "yavaşça +Adv", "ADV", "unimorph", 0.60, 3, "bronze"),
]


def test_canonical_tsv_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "canonical.tsv"
    n = to_canonical_tsv(_FIXTURE, out)
    assert n == len(_FIXTURE)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "surface\tcanonical_tags\tpos\ttier"
    assert len(lines) == len(_FIXTURE) + 1  # header + rows
    for i, entry in enumerate(_FIXTURE, 1):
        parts = lines[i].split("\t")
        assert parts[0] == entry.surface
        assert parts[1] == entry.canonical_tags
        assert parts[2] == entry.pos
        assert parts[3] == entry.tier


def test_conllu_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "out.conllu"
    n = to_conllu(_FIXTURE, out)
    assert n == len(_FIXTURE)
    text = out.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    assert len(blocks) == len(_FIXTURE)
    for block, entry in zip(blocks, _FIXTURE):
        lines = block.splitlines()
        token_line = next(l for l in lines if l.startswith("1\t"))
        parts = token_line.split("\t")
        assert parts[1] == entry.surface
        assert parts[2] == entry.lemma
        assert f"Tier={entry.tier}" in parts[9]


def test_ud_conllu_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "ud.conllu"
    n = to_ud_conllu(_FIXTURE, out)
    assert n == len(_FIXTURE)
    text = out.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    assert len(blocks) == len(_FIXTURE)
    for block, entry in zip(blocks, _FIXTURE):
        lines = block.splitlines()
        token_line = next(l for l in lines if l.startswith("1\t"))
        parts = token_line.split("\t")
        assert parts[1] == entry.surface
        assert parts[2] == entry.lemma


def test_unimorph_tsv_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "unimorph.tsv"
    n = to_unimorph_tsv(_FIXTURE, out)
    assert n == len(_FIXTURE)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_FIXTURE)
    for line, entry in zip(lines, _FIXTURE):
        parts = line.split("\t")
        assert len(parts) == 3
        assert parts[0] == entry.lemma
        assert parts[1] == entry.surface
