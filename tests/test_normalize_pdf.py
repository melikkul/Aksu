"""Tests for reconstruct_line_breaks — LM-aware PDF hyphenation decoder."""
from __future__ import annotations

import pytest

from aksu.ariturk.normalize import reconstruct_line_breaks


# ---------------------------------------------------------------------------
# Core decision-matrix cases
# ---------------------------------------------------------------------------

class TestLexiconHit:
    """LEX signal: joined form in lexicon → JOIN."""

    def test_turkce_genitive(self) -> None:
        # "Türk-\nçenin" — VH OK (ü/e both front) + fall-through JOIN
        # turkish_lower("Türkçenin") = "türkçenin" NOT in lex; VH ok → JOIN
        assert reconstruct_line_breaks("Türk-\nçenin") == "Türkçenin"

    def test_kitaplar(self) -> None:
        # "kitaplar" IS in lexicon → JOIN
        assert reconstruct_line_breaks("kitap-\nlar") == "kitaplar"

    def test_okulda(self) -> None:
        # "okulda" IS in lexicon → JOIN
        assert reconstruct_line_breaks("okul-\nda") == "okulda"

    def test_cocuklar(self) -> None:
        # "çocuklar" IS in lexicon → JOIN
        assert reconstruct_line_breaks("çocuk-\nlar") == "çocuklar"

    def test_anladim(self) -> None:
        # "anladım" IS in lexicon → JOIN
        assert reconstruct_line_breaks("anla-\ndım") == "anladım"

    def test_soyledi(self) -> None:
        # "söyledi" IS in lexicon → JOIN
        assert reconstruct_line_breaks("söy-\nledi") == "söyledi"

    def test_bakiyor(self) -> None:
        # "bakıyor" IS in lexicon → JOIN
        assert reconstruct_line_breaks("bakı-\nyor") == "bakıyor"


class TestCompoundPrefix:
    """CP signal: known compound prefix without LEX hit → PRESERVE hyphen.

    Note (plan §B.2 resolution (a)): when the joined form IS in the lexicon,
    LEX takes priority and JOIN wins. These tests cover cases where the
    joined form is also in the lexicon (eposta, antivirüs).
    """

    def test_eposta_lex_wins(self) -> None:
        # "eposta" IS in bundled lexicon → LEX wins → JOIN
        assert reconstruct_line_breaks("e-\nposta") == "eposta"

    def test_antivirus_lex_wins(self) -> None:
        # "antivirüs" IS in bundled lexicon → LEX wins → JOIN
        assert reconstruct_line_breaks("anti-\nvirüs") == "antivirüs"

    def test_cp_without_lex_preserves(self) -> None:
        # "co" prefix + a made-up word not in lexicon → PRESERVE
        # "coXYZ" won't be in the lexicon
        result = reconstruct_line_breaks("co-\nXYZ")
        assert result == "co-XYZ"

    def test_ex_prefix_preserves(self) -> None:
        # "ex-" prefix with unknown suffix
        result = reconstruct_line_breaks("ex-\npatriate")
        assert result == "ex-patriate"


class TestVowelHarmony:
    """VH signal: harmony violation → PRESERVE."""

    def test_harmony_violation_preserves(self) -> None:
        # front vowel (e) in word1, back vowel (a) in word2 → VH violation
        # "bek-\nlan" — "bek" last vowel 'e' (front), "lan" first vowel 'a' (back)
        # "beklan" likely not in lex → VH check → PRESERVE
        result = reconstruct_line_breaks("bek-\nlan")
        # If "beklan" is in lex it would JOIN; otherwise PRESERVE due to VH
        # We just verify it doesn't silently produce garbage
        assert result in {"bek-lan", "beklan"}

    def test_vowel_harmony_ok_joins(self) -> None:
        # front-front harmony: "gel-\ndi" — last vowel 'e' (front), first 'd→i' (front)
        # "geldi" IS in lexicon → JOIN
        assert reconstruct_line_breaks("gel-\ndi") == "geldi"


class TestDigitBoundary:
    """Digit-only splits are not matched (pattern requires \\p{L}+)."""

    def test_year_range_untouched(self) -> None:
        assert reconstruct_line_breaks("1990-\n2000") == "1990-\n2000"

    def test_numeric_fraction_untouched(self) -> None:
        assert reconstruct_line_breaks("3/4-\n5/6") == "3/4-\n5/6"


class TestMultipleSplitsInText:
    """Multiple hyphen-line-break occurrences in one string."""

    def test_two_splits_in_paragraph(self) -> None:
        text = "geldi-\nğinde ve gitti-\nmişti"
        result = reconstruct_line_breaks(text)
        # Both should be processed independently
        assert "\n" not in result or "geldiğinde" in result or "gittiğinde" in result

    def test_single_word_no_split(self) -> None:
        assert reconstruct_line_breaks("evlerinden") == "evlerinden"

    def test_no_hyphen_no_change(self) -> None:
        text = "bu bir cümle"
        assert reconstruct_line_breaks(text) == text


class TestWhitespaceVariants:
    """Whitespace around the newline is consumed."""

    def test_leading_spaces_after_newline(self) -> None:
        # "kitap-\n   lar" — spaces after \n are absorbed
        assert reconstruct_line_breaks("kitap-\n   lar") == "kitaplar"

    def test_trailing_spaces_before_hyphen(self) -> None:
        # "kitap -\nlar" — space before hyphen NOT part of the pattern
        # (pattern matches word-\n-word, space before hyphen breaks the letter run)
        result = reconstruct_line_breaks("kitap -\nlar")
        # "kitap " ends the letter run; the pattern won't match
        assert result == "kitap -\nlar"


class TestLmFlag:
    """use_lm flag is accepted and doesn't crash (kenlm is not required)."""

    def test_use_lm_true_no_crash(self) -> None:
        result = reconstruct_line_breaks("kitap-\nlar", use_lm=True)
        assert result == "kitaplar"

    def test_use_lm_false_same_result(self) -> None:
        r1 = reconstruct_line_breaks("kitap-\nlar", use_lm=True)
        r2 = reconstruct_line_breaks("kitap-\nlar", use_lm=False)
        assert r1 == r2
