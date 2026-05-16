"""Targeted coverage for TextCleaner and QualityChecker paths not covered by test_ariturk.py."""

from __future__ import annotations


class TestTextCleaner:
    def test_default_clean_lowercases(self):
        from aksu.ariturk import TextCleaner

        assert TextCleaner().clean("  TÜRKÇE   metİn  ") == "türkçe metin"

    def test_fix_diacritics_flag(self):
        from aksu.ariturk import TextCleaner

        cleaner = TextCleaner(fix_diacritics=True)
        result = cleaner.clean("turkce")
        assert isinstance(result, str)

    def test_remove_punctuation(self):
        from aksu.ariturk import TextCleaner

        cleaner = TextCleaner(remove_punctuation=True)
        result = cleaner.clean("merhaba, dünya!")
        assert "," not in result and "!" not in result

    def test_min_word_length(self):
        from aksu.ariturk import TextCleaner

        cleaner = TextCleaner(min_word_length=4)
        result = cleaner.clean("bu bir test")
        assert "bu" not in result
        assert "bir" not in result
        assert "test" in result

    def test_clean_batch(self):
        from aksu.ariturk import TextCleaner

        cleaner = TextCleaner()
        results = cleaner.clean_batch(["Merhaba", "DÜNYA"])
        assert results == ["merhaba", "dünya"]

    def test_is_clean(self):
        from aksu.ariturk import TextCleaner

        cleaner = TextCleaner()
        assert cleaner.is_clean("merhaba")
        assert not cleaner.is_clean("  MERHABA  ")


class TestQualityChecker:
    def test_gold_from_boun(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        assert checker.assign_tier(["boun"]) == "gold"

    def test_gold_from_imst(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        assert checker.assign_tier(["imst"]) == "gold"

    def test_silver_multi_source_agree(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        assert checker.assign_tier(["oscar", "mc4"], tags_agree=True) == "silver"

    def test_bronze_single_source(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        assert checker.assign_tier(["oscar"]) == "bronze"

    def test_bronze_multi_disagree(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        assert checker.assign_tier(["oscar", "mc4"], tags_agree=False) == "bronze"

    def test_validate_entry_valid(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        errors = checker.validate_entry("ev", "ev +Noun +POSS.3PL +ABL")
        assert errors == []

    def test_validate_entry_empty_surface(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        errors = checker.validate_entry("", "ev +Noun")
        assert any("Empty surface" in e for e in errors)

    def test_validate_entry_empty_canonical(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        errors = checker.validate_entry("ev", "")
        assert any("Empty canonical" in e for e in errors)

    def test_validate_entry_missing_plus_prefix(self):
        from aksu.ariturk import QualityChecker

        checker = QualityChecker()
        errors = checker.validate_entry("ev", "ev Noun ABL")
        assert any("missing + prefix" in e for e in errors)
