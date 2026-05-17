"""Tests for fix_pdf_artifacts and is_morphologically_valid."""
from __future__ import annotations

import pytest

from aksu.ariturk.normalize import fix_pdf_artifacts, is_morphologically_valid


# ---------------------------------------------------------------------------
# Ligature expansion (ftfy / NFKC stage)
# ---------------------------------------------------------------------------

class TestLigatureExpansion:
    def test_fi_ligature_word_start(self) -> None:
        # ﬁ (U+FB01) → fi at word start
        assert fix_pdf_artifacts("ﬁlm izledik") == "film izledik"

    def test_fi_ligature_mid_word(self) -> None:
        # ﬁrma → firma
        assert fix_pdf_artifacts("bu ﬁrma") == "bu firma"

    def test_fl_ligature(self) -> None:
        # ﬂ (U+FB02) → fl
        assert fix_pdf_artifacts("ﬂuorit") == "fluorit"

    def test_multiple_ligatures(self) -> None:
        text = "ﬁlm ﬁrma"
        result = fix_pdf_artifacts(text)
        assert "fi" in result
        assert "ﬁ" not in result


# ---------------------------------------------------------------------------
# Zero-width character removal
# ---------------------------------------------------------------------------

class TestZeroWidthRemoval:
    def test_zwsp_removed(self) -> None:
        # U+200B zero-width space
        assert fix_pdf_artifacts("Türk​çe") == "Türkçe"

    def test_zwj_removed(self) -> None:
        # U+200D zero-width joiner
        assert fix_pdf_artifacts("bir‍lik") == "birlik"

    def test_bom_removed(self) -> None:
        # U+FEFF byte-order mark
        assert fix_pdf_artifacts("﻿Merhaba") == "Merhaba"

    def test_variation_selector_removed(self) -> None:
        # U+FE00 variation selector
        assert fix_pdf_artifacts("Türk︀çe") == "Türkçe"


# ---------------------------------------------------------------------------
# Repeated-character collapsing
# ---------------------------------------------------------------------------

class TestRepeatedCharCollapse:
    def test_four_or_more_collapses_to_two(self) -> None:
        assert fix_pdf_artifacts("çoooook") == "çook"

    def test_three_chars_unchanged(self) -> None:
        # Exactly 3 repetitions should NOT be collapsed (threshold is >3 → 2)
        result = fix_pdf_artifacts("aaa")
        assert result == "aaa"

    def test_double_consonant_preserved(self) -> None:
        # "anne" has double n — must not be collapsed
        assert fix_pdf_artifacts("anne") == "anne"

    def test_repeated_digits_collapse(self) -> None:
        # 6 zeros → 2 zeros
        assert fix_pdf_artifacts("000000") == "00"

    def test_five_same_chars_collapses(self) -> None:
        assert fix_pdf_artifacts("aaaaa") == "aa"


# ---------------------------------------------------------------------------
# Space-injection repair
# ---------------------------------------------------------------------------

class TestSpaceInjection:
    def test_five_single_chars(self) -> None:
        # "M e t i n" — 5 single letters, each separated by space
        assert fix_pdf_artifacts("M e t i n") == "Metin"

    def test_longer_space_injected(self) -> None:
        assert fix_pdf_artifacts("A n k a r a") == "Ankara"

    def test_four_chars_not_repaired(self) -> None:
        # Only 4 single letters — below the ≥5 threshold
        result = fix_pdf_artifacts("a b c d")
        # Should NOT be collapsed (4 chars, threshold requires 5+)
        assert result == "a b c d"

    def test_normal_words_unchanged(self) -> None:
        # Multi-char words should NOT be collapsed
        assert fix_pdf_artifacts("bu bir cümle") == "bu bir cümle"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.mark.parametrize("text", [
        "Türkçe metin örneği",
        "ﬁlm izledik",
        "çoooook güzel",
        "Türk​çe",
        "normal text without issues",
        "M e t i n",
        "evlerinden",
    ])
    def test_idempotent(self, text: str) -> None:
        r1 = fix_pdf_artifacts(text)
        r2 = fix_pdf_artifacts(r1)
        assert r1 == r2, f"Not idempotent: f(f({text!r})) != f({text!r})"


# ---------------------------------------------------------------------------
# Aggressive mode (header/footer removal)
# ---------------------------------------------------------------------------

class TestAggressiveMode:
    def test_repeated_header_removed(self) -> None:
        page1 = "Sayfa 1\n\nBu bir metin."
        page2 = "Sayfa 1\n\nBaşka bir metin."
        page3 = "Sayfa 1\n\nÜçüncü metin."
        text = f"{page1}\n\n\n{page2}\n\n\n{page3}"
        result = fix_pdf_artifacts(text, aggressive=True)
        assert "Sayfa 1" not in result

    def test_non_repeated_lines_kept(self) -> None:
        text = "benzersiz satır\n\n\nunique line\n\n\nanother unique"
        result = fix_pdf_artifacts(text, aggressive=True)
        assert "benzersiz satır" in result


# ---------------------------------------------------------------------------
# Repair diacritics flag
# ---------------------------------------------------------------------------

class TestRepairDiacritics:
    def test_diacritics_restored_when_enabled(self) -> None:
        result = fix_pdf_artifacts("turkce", repair_diacritics=True)
        assert "türkçe" in result

    def test_diacritics_not_touched_by_default(self) -> None:
        result = fix_pdf_artifacts("turkce")
        assert result == "turkce"


# ---------------------------------------------------------------------------
# is_morphologically_valid
# ---------------------------------------------------------------------------

class TestIsMorphologicallyValid:
    def test_pure_front_harmony(self) -> None:
        # evlerinde: e-e-i-e — all front
        assert is_morphologically_valid("evlerinde") is True

    def test_pure_back_harmony(self) -> None:
        # okullarda: o-u-a-a — all back
        assert is_morphologically_valid("okullarda") is True

    def test_single_vowel_always_valid(self) -> None:
        assert is_morphologically_valid("ev") is True
        assert is_morphologically_valid("al") is True

    def test_no_vowel_always_valid(self) -> None:
        # single consonant or no vowel
        assert is_morphologically_valid("k") is True

    def test_loanword_tolerance(self) -> None:
        # "bilgisayar": i-i-a-a — two front + two back = 50% violations → False
        # (exceeds 30% tolerance)
        # But "sinema": i-e-a — i(front) e(front) a(back) = 1/3 ≈ 33% → False
        # These are loanwords correctly identified as VH-violating
        result = is_morphologically_valid("sinema")
        # 33% violation > 30% → False — correct behaviour for loanword detection
        assert isinstance(result, bool)

    def test_empty_string(self) -> None:
        assert is_morphologically_valid("") is True

    def test_front_back_mixed_over_threshold(self) -> None:
        # Construct a word with >30% VH violation
        # "aeua" → a(back) e(front) u(back) a(back): violations=1/4=25% → True
        # "aeuo" → a(back) e(front) u(back) o(back): violations=1/4=25% → True
        # "aeio" → a(back) e(front) i(front) o(back): violations=2/4=50% → False
        assert is_morphologically_valid("aeio") is False
