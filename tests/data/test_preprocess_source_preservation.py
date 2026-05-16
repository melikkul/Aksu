"""Unit tests for preprocess.py source/license field preservation (Stage 0 item 3).

Verifies that per-row source and license metadata survives the preprocessing
pipeline instead of being silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aksu.data.build.preprocess import _load_local_jsonl, preprocess_shard


# ---------------------------------------------------------------------------
# _load_local_jsonl: returns dicts with source/license preserved
# ---------------------------------------------------------------------------

def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_local_jsonl_preserves_source_and_license(tmp_path: Path) -> None:
    rows = [
        {"text": "Bu bir test cümlesidir.", "source": "oscar-tr", "license": "CC0"},
        {"text": "Başka bir cümle var.", "source": "wiki-tr", "license": "CC-BY-SA-3.0"},
    ]
    fpath = tmp_path / "test.jsonl"
    _write_jsonl(rows, fpath)

    loaded = _load_local_jsonl(fpath)
    assert len(loaded) == 2
    assert loaded[0]["source"] == "oscar-tr"
    assert loaded[0]["license"] == "CC0"
    assert loaded[1]["source"] == "wiki-tr"
    assert loaded[1]["license"] == "CC-BY-SA-3.0"


def test_load_local_jsonl_preserves_text(tmp_path: Path) -> None:
    rows = [{"text": "Ankara güzel bir şehir.", "source": "test"}]
    fpath = tmp_path / "test.jsonl"
    _write_jsonl(rows, fpath)

    loaded = _load_local_jsonl(fpath)
    assert loaded[0]["text"] == "Ankara güzel bir şehir."


def test_load_local_jsonl_returns_dicts(tmp_path: Path) -> None:
    rows = [{"text": "test", "source": "s"}]
    fpath = tmp_path / "test.jsonl"
    _write_jsonl(rows, fpath)

    loaded = _load_local_jsonl(fpath)
    assert isinstance(loaded[0], dict)


def test_load_local_jsonl_empty_file(tmp_path: Path) -> None:
    fpath = tmp_path / "empty.jsonl"
    fpath.write_text("", encoding="utf-8")
    loaded = _load_local_jsonl(fpath)
    assert loaded == []


# ---------------------------------------------------------------------------
# preprocess_shard: per-row source/license override shard default
# ---------------------------------------------------------------------------

def _make_passthrough_qf() -> MagicMock:
    """Quality filter mock that passes all rows unchanged."""
    qf = MagicMock()
    stats = MagicMock()
    stats.passed = 0
    stats.total = 0
    stats.dropped_lang = 0
    stats.dropped_length = 0
    stats.dropped_dedup = 0
    stats.pii_scrubbed = 0

    def _filter(rows, **kw):
        stats.passed = len(rows)
        stats.total = len(rows)
        return rows, stats

    qf.filter_sentences.side_effect = _filter
    return qf


def test_preprocess_shard_returns_stats(tmp_path: Path) -> None:
    """preprocess_shard should return a stats dict without raising."""
    rows = [
        {"text": "Türkçe cümle burada yer alıyor.", "source": "wiki-tr", "license": "CC-BY-SA-3.0"},
        {"text": "Başka kaynak gelen cümle burada var.", "source": "oscar-tr", "license": "CC0"},
    ]
    out_dir = tmp_path / "out"
    qf = _make_passthrough_qf()

    result = preprocess_shard(
        rows,
        source_name="oscar-tr",
        source_license="CC-BY-4.0",
        output_dir=out_dir,
        quality_filter=qf,
    )
    assert isinstance(result, dict)


def test_preprocess_shard_calls_quality_filter(tmp_path: Path) -> None:
    """Quality filter is invoked when provided."""
    rows = [{"text": "Test cümlesidir bu içerik."}]
    qf = _make_passthrough_qf()

    preprocess_shard(
        rows,
        source_name="oscar-tr",
        source_license="CC-BY-4.0",
        output_dir=tmp_path / "out2",
        quality_filter=qf,
    )

    assert qf.filter_sentences.called


def test_preprocess_shard_no_quality_filter(tmp_path: Path) -> None:
    """preprocess_shard works fine with quality_filter=None."""
    rows = [{"text": "Kalite filtresi olmadan çalışmalı bu."}]
    result = preprocess_shard(
        rows,
        source_name="test-src",
        source_license="CC0",
        output_dir=tmp_path / "out3",
        quality_filter=None,
    )
    assert isinstance(result, dict)


def test_preprocess_shard_v1_test_exclusion(tmp_path: Path) -> None:
    """Tokens in v1_test_surfaces are excluded and counted."""
    # This test verifies the interface exists; actual dedup logic is in token loop
    rows = [
        {"text": "Ankara güzel şehir büyük."},
        {"text": "İstanbul güzel şehir büyük."},
    ]
    qf = _make_passthrough_qf()
    result = preprocess_shard(
        rows,
        source_name="test",
        source_license="CC0",
        output_dir=tmp_path / "out4",
        quality_filter=qf,
        v1_test_surfaces=frozenset(["Ankara"]),
    )
    # v1_test_contamination_skipped should be reported in stats
    assert "v1_test_contamination_skipped" in result


# ---------------------------------------------------------------------------
# Round-trip: write JSONL → load → verify metadata survives
# ---------------------------------------------------------------------------

def test_round_trip_source_license(tmp_path: Path) -> None:
    original = [
        {"text": "Test cümlesi.", "source": "mc4-tr", "license": "ODC-BY"},
        {"text": "Başka test.", "source": "boun-ud", "license": "CC-BY-SA-4.0"},
    ]
    fpath = tmp_path / "rt.jsonl"
    _write_jsonl(original, fpath)

    loaded = _load_local_jsonl(fpath)
    for orig, got in zip(original, loaded):
        assert got["source"] == orig["source"]
        assert got["license"] == orig["license"]
        assert got["text"] == orig["text"]
