"""High-level Turkish text cleaning interface."""
from __future__ import annotations

from aksu.ariturk.normalize import (
    fix_pdf_artifacts,
    is_valid_turkish,
    normalize_surface,
    reconstruct_line_breaks,
    restore_diacritics,
    turkish_lower,
)


class TextCleaner:
    """Turkish text cleaning pipeline.

    Args:
        lowercase: Apply Turkish-correct lowercasing.
        fix_diacritics: Attempt to restore missing ç,ğ,ı,ö,ş,ü.
        remove_punctuation: Strip all non-alphanumeric characters.
        min_word_length: Drop words shorter than this.
    """

    def __init__(
        self,
        lowercase: bool = True,
        fix_diacritics: bool = False,
        remove_punctuation: bool = False,
        min_word_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self._fix_diacritics = fix_diacritics
        self.remove_punctuation = remove_punctuation
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """Clean a single text string."""
        text = normalize_surface(text)
        if self.lowercase:
            text = turkish_lower(text)
        if self._fix_diacritics:
            text = restore_diacritics(text)
        if self.remove_punctuation:
            text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        if self.min_word_length > 1:
            words = text.split()
            text = " ".join(w for w in words if len(w) >= self.min_word_length)
        return text.strip()

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a list of texts."""
        return [self.clean(t) for t in texts]

    def is_clean(self, text: str) -> bool:
        """Check if text is already clean Turkish."""
        return is_valid_turkish(text) and text == normalize_surface(text)

    def fix_line_breaks(self, text: str, *, use_lm: bool = True) -> str:
        """Re-join words split across PDF line breaks.

        Delegates to :func:`aksu.ariturk.normalize.reconstruct_line_breaks`.

        Args:
            text: Raw PDF-extracted text with soft hyphens.
            use_lm: Passed through to the underlying function (reserved for
                kenlm integration in v1.1; currently a no-op without aksu[full]).

        Returns:
            Text with PDF line-break hyphens reconstructed.
        """
        return reconstruct_line_breaks(text, use_lm=use_lm)

    def fix_artifacts(
        self,
        text: str,
        *,
        aggressive: bool = False,
        repair_diacritics: bool = False,
    ) -> str:
        """Remove PDF extraction artifacts (mojibake, zero-width chars, etc.).

        Delegates to :func:`aksu.ariturk.normalize.fix_pdf_artifacts`.

        Args:
            text: Raw PDF-extracted text.
            aggressive: When True, also strip repeated header/footer lines.
            repair_diacritics: When True, apply the 11-entry diacritic stub.

        Returns:
            Cleaned text. Idempotent for typical inputs.
        """
        return fix_pdf_artifacts(
            text, aggressive=aggressive, repair_diacritics=repair_diacritics
        )
