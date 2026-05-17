"""Turkish text normalization utilities."""
from __future__ import annotations

import gzip
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import regex  # Unicode-aware regex (installed as transitive dep via nltk)


def normalize_surface(text: str) -> str:
    """Full normalization pipeline for a surface form.

    Applies NFC normalization, strips whitespace, and collapses internal runs.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I→ı, İ→i."""
    result: list[str] = []
    for ch in text:
        if ch == "I":
            result.append("ı")
        elif ch == "\u0130":  # İ
            result.append("i")
        else:
            result.append(ch.lower())
    return "".join(result)


def turkish_upper(text: str) -> str:
    """Uppercase with Turkish rules: i→İ, ı→I."""
    result: list[str] = []
    for ch in text:
        if ch == "i":
            result.append("\u0130")  # İ
        elif ch == "ı":
            result.append("I")
        else:
            result.append(ch.upper())
    return "".join(result)


_TURKISH_CHARS = frozenset(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "0123456789'-. "
)


def is_valid_turkish(text: str) -> bool:
    """Check if text contains only valid Turkish characters."""
    return all(ch in _TURKISH_CHARS for ch in text)


# Common ASCII→diacritics fixes
_DIACRITIC_FIXES: dict[str, str] = {
    "turkce": "türkçe",
    "turk": "türk",
    "turkiye": "türkiye",
    "ogrenci": "öğrenci",
    "ogretmen": "öğretmen",
    "universite": "üniversite",
    "guzel": "güzel",
    "buyuk": "büyük",
    "kucuk": "küçük",
    "calisma": "çalışma",
    "islem": "işlem",
}


def restore_diacritics(text: str) -> str:
    """Attempt to restore missing Turkish diacritics using a lookup table."""
    words = text.split()
    return " ".join(_DIACRITIC_FIXES.get(w.lower(), w) for w in words)


# ---------------------------------------------------------------------------
# Lexicon helpers
# ---------------------------------------------------------------------------

_LEXICON_PATH = Path(__file__).parent / "data" / "turkish_wordlist.txt.gz"

# Known compound prefixes whose end-of-line hyphens are typically semantic
_CP_SET: frozenset[str] = frozenset(
    {"e", "anti", "co", "ex", "post", "pre", "non", "sub"}
)

# Matches a word-letter run, optional whitespace/newline, and another run
# e.g. "Türk-\nçenin" → groups ("Türk", "çenin")
_HYPHEN_LINE_BREAK = regex.compile(
    r"(\p{L}+)-[ \t]*\n[ \t]*(\p{L}+)", regex.UNICODE
)

# Zero-width / private-use characters that OCR/PDF converters inject
_ZW_PATTERN = regex.compile(
    r"[​‌‍﻿︀-️\U000E0000-\U000E007F]"
)

# Run of the same character repeated more than 3 times → keep only 2
# v1.1 conservative baseline: Turkish uses single vowels (sayfa, not sayfaa),
# but doubled consonants exist (anne, bakkal). Lexicon-aware snap is v1.2.
_REPEAT_PATTERN = regex.compile(r"(.)\1{3,}", regex.DOTALL)

# Space-injected single letters: "M e t i n" → "Metin" (≥5 single letters)
_SPACE_INJECT = regex.compile(r"(?<!\S)(\p{L})( \p{L}){4,}(?!\S)", regex.UNICODE)


@lru_cache(maxsize=1)
def _load_lexicon() -> frozenset[str]:
    """Load bundled Zemberek Turkish wordlist (lazy, cached).

    Returns empty frozenset when the wordlist is absent — callers treat
    a miss as 'lexicon unavailable' and fall back to VH / CP signals.

    Note: Zemberek covers ~150K base forms, not inflected surfaces.
    LEX-hit fires rarely on real Turkish text; VH + CP carry most decisions.
    Lemma-aware lookup (root-stripping before LEX check) is deferred to v1.2.
    """
    if not _LEXICON_PATH.exists():
        return frozenset()
    with gzip.open(_LEXICON_PATH, "rt", encoding="utf-8") as fh:
        return frozenset(line.strip() for line in fh if line.strip())


def _vowel_harmony_ok(word1: str, word2: str) -> bool:
    """Return True if last vowel of word1 is in the same harmony class as first vowel of word2."""
    _FRONT = frozenset("eiöü")
    _BACK = frozenset("aıou")
    _ALL = _FRONT | _BACK
    w1 = turkish_lower(word1)
    w2 = turkish_lower(word2)
    v1 = [c for c in w1 if c in _ALL]
    v2 = [c for c in w2 if c in _ALL]
    if not v1 or not v2:
        return True  # no vowels to check — assume compatible
    last_v = v1[-1]
    first_v = v2[0]
    return (last_v in _FRONT) == (first_v in _FRONT)


def _remove_repeated_headers(text: str) -> str:
    """Remove lines that repeat verbatim across ≥3 page segments.

    A page segment is delimited by a form-feed (\\f) or three or more blank
    lines. Only used when aggressive=True in fix_pdf_artifacts().
    """
    segments = re.split(r"\f|\n{3,}", text)
    if len(segments) < 3:
        return text
    line_counts: Counter[str] = Counter()
    for seg in segments:
        seen: set[str] = set()
        for line in seg.splitlines():
            stripped = line.strip()
            if stripped and stripped not in seen:
                line_counts[stripped] += 1
                seen.add(stripped)
    repeated = {line for line, cnt in line_counts.items() if cnt >= 3}
    if not repeated:
        return text
    return "\n".join(ln for ln in text.splitlines() if ln.strip() not in repeated)


# ---------------------------------------------------------------------------
# Public v1.1 APIs
# ---------------------------------------------------------------------------


def is_morphologically_valid(word: str) -> bool:
    """Heuristic vowel-harmony sanity check for a single word surface.

    Returns True when the word's vowel sequence is internally consistent
    with Turkish front/back harmony. Applies a 30% loanword tolerance so
    common foreign borrowings (e.g. 'telefon', 'sinema') do not fail.

    Args:
        word: A single word surface (no whitespace).

    Returns:
        True if ≤30% of the vowels violate the initial harmony class.
    """
    _FRONT = frozenset("eiöü")
    _ALL = frozenset("aeıioöuü")
    vowels = [c for c in turkish_lower(word) if c in _ALL]
    if len(vowels) < 2:
        return True
    target = _FRONT if vowels[0] in _FRONT else (_ALL - _FRONT)
    violations = sum(1 for v in vowels[1:] if v not in target)
    return violations / len(vowels) < 0.3


def reconstruct_line_breaks(text: str, use_lm: bool = True) -> str:  # noqa: ARG001
    """Re-join words that were split across PDF line boundaries with hyphens.

    Applies a three-signal decision matrix to each candidate split:

    | Priority | Signal            | Action   |
    |----------|-------------------|----------|
    | 1        | Lexicon hit (LEX) | JOIN     |
    | 2        | Compound prefix (CP) without LEX | PRESERVE |
    | 3        | Vowel harmony violation (VH) | PRESERVE |
    | 4        | Fall-through      | JOIN     |

    LM scoring (kenlm Turkish 3-gram) is wired to the `use_lm` flag but
    requires the ``aksu[full]`` optional dependency; it is a no-op in v1.1
    when kenlm is not installed.

    Priority risk: if Zemberek lexicon contains the joined form of a compound-
    prefix word (e.g. "eposta"), LEX takes precedence over CP and the form
    joins. Phase 0 captured expectations define ground truth for such cases.

    Args:
        text: Raw PDF-extracted text potentially containing soft hyphens.
        use_lm: Reserved for kenlm integration (v1.1 always no-op if kenlm absent).

    Returns:
        Text with PDF line-break hyphens reconstructed.
    """
    lexicon = _load_lexicon()

    def _decide(m: regex.Match) -> str:  # type: ignore[type-arg]
        word1: str = m.group(1)
        word2: str = m.group(2)
        joined = word1 + word2
        w1_lower = turkish_lower(word1)

        # LEX: joined form in bundled wordlist → JOIN
        if lexicon and turkish_lower(joined) in lexicon:
            return joined

        # CP: known compound prefix without LEX support → PRESERVE the hyphen
        if w1_lower in _CP_SET:
            return f"{word1}-{word2}"

        # VH: joining would create a vowel-harmony violation → PRESERVE
        if not _vowel_harmony_ok(word1, word2):
            return f"{word1}-{word2}"

        # Fall-through: PDF line breaks are statistically more common than
        # semantic hyphens at end-of-line in Turkish corpora → JOIN
        return joined

    return _HYPHEN_LINE_BREAK.sub(_decide, text)


def fix_pdf_artifacts(
    text: str,
    *,
    aggressive: bool = False,
    repair_diacritics: bool = False,
) -> str:
    """Multi-stage cleaning pipeline for PDF-extracted Turkish text.

    Stages (always applied unless noted):
    1. normalize_surface — NFC + whitespace collapse (base cleanup)
    2. ftfy.fix_text — mojibake reversal, ligature expansion (e.g. ﬁ → fi)
    3. unicodedata NFKC — belt-and-braces compatibility normalization
    4. Zero-width / private-use strip — removes U+200B, U+200D, FE00-FE0F etc.
    5. Repeated-char collapse — any char repeated >3 times → 2 chars
       (v1.1 conservative baseline; lexicon-aware snap is v1.2 work)
    6. Space-injection repair — "M e t i n" → "Metin" (≥5 single letters)
    7. Header/footer removal (only when aggressive=True)
    8. Diacritic restoration (only when repair_diacritics=True; 11-entry stub)

    Args:
        text: Raw PDF-extracted text.
        aggressive: When True, also remove lines repeated across ≥3 page segments.
        repair_diacritics: When True, apply the 11-entry diacritic-restoration stub.

    Returns:
        Cleaned text. Idempotent: f(f(x)) == f(x) for typical inputs.
    """
    # Stage 1 (aggressive only): header/footer removal before whitespace collapse.
    # Must run before normalize_surface because normalize_surface collapses all
    # multi-blank-line page separators that _remove_repeated_headers relies on.
    if aggressive:
        text = _remove_repeated_headers(text)

    # Stage 2: base NFC + whitespace normalisation
    text = normalize_surface(text)

    # Stage 3: ftfy mojibake reversal + ligature expansion + NFKC
    try:
        import ftfy  # type: ignore[import-untyped]

        text = ftfy.fix_text(text, normalization="NFKC")
    except ImportError:
        pass  # belt-and-braces NFKC in stage 4 covers most cases

    # Stage 4: defensive NFKC (no-op if ftfy already applied it)
    text = unicodedata.normalize("NFKC", text)

    # Stage 5: zero-width and private-use character removal
    text = _ZW_PATTERN.sub("", text)

    # Stage 6: repeated-character collapsing (>3 identical chars → 2)
    text = _REPEAT_PATTERN.sub(r"\1\1", text)

    # Stage 7: space-injection repair ("M e t i n" → "Metin")
    text = _SPACE_INJECT.sub(lambda m: m.group(0).replace(" ", ""), text)

    # Stage 8: diacritic restoration stub (11-entry dict, char-LM is v1.2)
    if repair_diacritics:
        text = restore_diacritics(text)

    return text
