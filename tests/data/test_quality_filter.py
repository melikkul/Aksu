"""Unit tests for quality_filter.py (§H+ A / Stage 0 item 4).

Covers: PII scrubbing, language filter, length filter, and sentence-level dedup.
Language detection is mocked via patch.object(QualityFilter, '_is_turkish').
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from aksu.data.build.quality_filter import QualityFilter, _scrub_pii


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_tag", [
    ("Bana mail at: ahmet@example.com", "<EMAIL>"),
    ("IP: 192.168.1.1 ve URL: https://example.com/path", "<URL>"),
    ("IBAN TR330006100519786457841326 ile ödeme", "<IBAN>"),
    ("TC kimlik: 12345678901 gerekli", "<TCNO>"),
    ("Telefon: +905321234567 ara", "<PHONE>"),
    ("Telefon: 05321234567 ara", "<PHONE>"),
])
def test_pii_scrub_replaces_tag(text: str, expected_tag: str) -> None:
    scrubbed = _scrub_pii(text)
    assert expected_tag in scrubbed


def test_pii_scrub_no_false_positive_on_clean_text() -> None:
    clean = "Türkiye güzel bir ülkedir ve tarihi zengindir."
    scrubbed = _scrub_pii(clean)
    assert scrubbed == clean


def test_pii_scrub_multiple_in_one_sentence() -> None:
    text = "ahmet@test.com ile https://site.com adresine bak"
    scrubbed = _scrub_pii(text)
    assert "<EMAIL>" in scrubbed
    assert "<URL>" in scrubbed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(text: str, source: str = "oscar-tr") -> dict:
    return {"text": text, "source": source}


# ---------------------------------------------------------------------------
# Length filter
# ---------------------------------------------------------------------------

def test_length_filter_min() -> None:
    qf = QualityFilter(lang_threshold=0.85, min_tokens=3, max_tokens=50)
    short = _make_row("Evet hayır")   # 2 tokens → drop
    ok = _make_row("Bu bir test cümlesidir içerik var")  # 6 tokens → keep
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences([short, ok])
    assert len(kept) == 1
    assert stats.dropped_length == 1


def test_length_filter_max() -> None:
    qf = QualityFilter(lang_threshold=0.85, min_tokens=3, max_tokens=5)
    too_long = _make_row("bir iki üç dört beş altı yedi")   # 7 tokens → drop
    ok = _make_row("bir iki üç dört")                        # 4 tokens → keep
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences([too_long, ok])
    assert len(kept) == 1
    assert stats.dropped_length == 1


# ---------------------------------------------------------------------------
# Language filter
# ---------------------------------------------------------------------------

def test_language_filter_drops_low_score() -> None:
    qf = QualityFilter(lang_threshold=0.85)
    rows = [_make_row("This is English text here please")]
    with patch.object(QualityFilter, "_is_turkish", return_value=False):
        kept, stats = qf.filter_sentences(rows)
    assert len(kept) == 0
    assert stats.dropped_lang == 1


def test_language_filter_passes_turkish() -> None:
    qf = QualityFilter(lang_threshold=0.85)
    rows = [_make_row("Ankara Türkiye'nin başkentidir.")]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    assert len(kept) == 1
    assert stats.dropped_lang == 0


def test_mc4_higher_threshold_config() -> None:
    """A QualityFilter with higher lang_threshold=0.95 drops score-0.90 rows."""
    qf_strict = QualityFilter(lang_threshold=0.95)
    qf_lenient = QualityFilter(lang_threshold=0.85)
    row = _make_row("Türkçe metin burada gerçekten güzel")

    # strict filter uses 0.95 threshold: score 0.90 → fail (return False)
    with patch.object(QualityFilter, "_is_turkish", return_value=False):
        kept_strict, _ = qf_strict.filter_sentences([row])
    assert len(kept_strict) == 0

    # lenient filter: score 0.90 → pass (return True)
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept_lenient, _ = qf_lenient.filter_sentences([dict(row)])
    assert len(kept_lenient) == 1


# ---------------------------------------------------------------------------
# Deduplication (sentence-level md5, §H+ C)
# ---------------------------------------------------------------------------

def test_dedup_removes_exact_duplicates() -> None:
    qf = QualityFilter()
    text = "Ankara Türkiye'nin başkentidir ve önemli bir şehirdir."
    rows = [_make_row(text), _make_row(text), _make_row(text)]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    assert len(kept) == 1
    assert stats.dropped_dedup == 2


def test_dedup_keeps_different_sentences() -> None:
    qf = QualityFilter()
    rows = [
        _make_row("Ankara Türkiye'nin başkentidir."),
        _make_row("İstanbul kültür merkezi ve ticaret şehridir."),
    ]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    assert len(kept) == 2
    assert stats.dropped_dedup == 0


def test_dedup_same_word_different_context_both_kept() -> None:
    """Same surface word in different sentences = two entries (§H+ C)."""
    qf = QualityFilter()
    rows = [
        _make_row("Kedi evin içinde uyuyor."),      # ≥3 tokens, unique sentence
        _make_row("Kedi bahçede oyun oynuyor."),     # ≥3 tokens, unique sentence
    ]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# PII scrubbing flag
# ---------------------------------------------------------------------------

def test_pii_scrubbed_count_tracked() -> None:
    qf = QualityFilter()
    rows = [_make_row("Bana ahmet@example.com yaz lütfen hemen")]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    # PII scrubbed but row is kept (not dropped)
    assert len(kept) == 1
    assert stats.pii_scrubbed >= 1


# ---------------------------------------------------------------------------
# FilterStats totals
# ---------------------------------------------------------------------------

def test_filter_stats_total_consistent() -> None:
    qf = QualityFilter(lang_threshold=0.85, min_tokens=3, max_tokens=50)
    rows = [
        _make_row("ok"),           # too short → dropped_length
        _make_row("Türkçe metin kelimeler bulunmaktadır gerçekten"),  # ok
        _make_row("Türkçe metin kelimeler bulunmaktadır gerçekten"),  # dedup
    ]
    with patch.object(QualityFilter, "_is_turkish", return_value=True):
        kept, stats = qf.filter_sentences(rows)
    assert stats.total == 3
    assert stats.passed == len(kept)
    assert stats.dropped_length + stats.dropped_lang + stats.dropped_dedup + stats.passed == stats.total
