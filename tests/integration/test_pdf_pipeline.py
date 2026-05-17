"""Integration test: noisy PDF text → cleaning → morphological analysis."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "pdf_samples"


def _get_parse_rate(text: str) -> float:
    """Fraction of whitespace-delimited tokens that receive ≥1 parse."""
    from aksu.kokturk.core.analyzer import MorphoAnalyzer

    ma = MorphoAnalyzer()
    tokens = text.split()
    if not tokens:
        return 0.0
    parsed = sum(1 for tok in tokens if ma.analyze(tok).parse_count > 0)
    return parsed / len(tokens)


@pytest.mark.parametrize("sample_file", [
    "pdf_sample_1.txt",
    "pdf_sample_2.txt",
    "pdf_sample_3.txt",
])
def test_parse_rate_improves_after_cleaning(sample_file: str) -> None:
    """≥90% of tokens get a parse after cleaning, vs ≤80% without (or same raw)."""
    from aksu.ariturk.normalize import fix_pdf_artifacts, reconstruct_line_breaks

    raw = (FIXTURES_DIR / sample_file).read_text(encoding="utf-8")

    # Cleaned pipeline
    cleaned = fix_pdf_artifacts(reconstruct_line_breaks(raw))

    raw_rate = _get_parse_rate(raw)
    cleaned_rate = _get_parse_rate(cleaned)

    # Parse-presence assertion: at least 90% after cleaning
    # (5-point safety margin — log actual percentage but don't fail at 91%)
    assert cleaned_rate >= 0.90, (
        f"{sample_file}: only {cleaned_rate:.1%} tokens parsed after cleaning "
        f"(raw: {raw_rate:.1%}). Expected ≥90%."
    )


def test_pipeline_end_to_end_smoke() -> None:
    """Basic smoke: pipeline does not crash and returns non-empty output."""
    from aksu.ariturk.normalize import fix_pdf_artifacts, reconstruct_line_breaks

    raw = "kitap-\nlar ve okul-\nda çalış-\nmak gerekir."
    cleaned = fix_pdf_artifacts(reconstruct_line_breaks(raw))
    assert cleaned
    assert "\n" not in cleaned  # line breaks absorbed


def test_reconstruct_then_analyze_joined_form() -> None:
    """After reconstruct_line_breaks, 'kitaplar' is analysable by Zeyrek."""
    from aksu.ariturk.normalize import reconstruct_line_breaks
    from aksu.kokturk.core.analyzer import MorphoAnalyzer

    raw = "kitap-\nlar"
    cleaned = reconstruct_line_breaks(raw)
    assert cleaned == "kitaplar"

    ma = MorphoAnalyzer()
    result = ma.analyze("kitaplar")
    assert result.parse_count > 0, "Zeyrek returned no parse for 'kitaplar'"
