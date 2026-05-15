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
]

from aksu.ariturk.normalize import (
    normalize_surface,
    turkish_lower,
    turkish_upper,
    is_valid_turkish,
)
from aksu.ariturk.cleaner import TextCleaner
from aksu.ariturk.quality import QualityChecker
from aksu.ariturk.boundaries import BoundaryExtractor
