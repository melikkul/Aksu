"""arı-türk: Turkish Text Cleaning & Normalization Library.

Features:
    - Turkish-correct case handling (I→ı, İ→i)
    - Unicode NFC normalization
    - Diacritics restoration
    - Quality tier assignment (gold/silver/bronze)
    - Morpheme boundary extraction

Example::

    >>> from ariturk import TextCleaner
    >>> cleaner = TextCleaner()
    >>> cleaner.clean("  TÜRKÇE   metİn  ")
    'türkçe metin'
"""
from __future__ import annotations

from aksu._version import __version__

__all__ = [
    "TextCleaner",
    "QualityChecker",
    "BoundaryExtractor",
    "normalize_surface",
    "turkish_lower",
    "turkish_upper",
    "is_valid_turkish",
    # v1.1 additions
    "reconstruct_line_breaks",
    "fix_pdf_artifacts",
    "is_morphologically_valid",
]

from aksu.ariturk.boundaries import BoundaryExtractor
from aksu.ariturk.cleaner import TextCleaner
from aksu.ariturk.normalize import (
    fix_pdf_artifacts,
    is_morphologically_valid,
    is_valid_turkish,
    normalize_surface,
    reconstruct_line_breaks,
    turkish_lower,
    turkish_upper,
)
from aksu.ariturk.quality import QualityChecker
