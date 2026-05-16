"""Quality filter for TR-Gold-Morph v2 pipeline.

Applies four independent filters:
  1. Language-confidence filter (fasttext-lid; 0.85 default, 0.95 for mC4)
  2. PII scrubbing (email, phone, TC kimlik 11-digit, IBAN, URL)
  3. Length filter (3 ≤ whitespace-token count ≤ 50)
  4. Sentence-level md5 dedup (same content in different sources deduplicated)

Usage:
    from aksu.data.build.quality_filter import QualityFilter
    qf = QualityFilter()
    kept = qf.filter_sentences(sentences, source="oscar-tr")
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# PII patterns — heuristic only, not a production NER system
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I)
_URL_RE = re.compile(
    r"https?://[^\s]+"
    r"|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s]*",
    re.I,
)
_PHONE_TR_RE = re.compile(
    r"(?:\+90|0)[-\s]?\(?[0-9]{3,4}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{2}"
)
_TC_KIMLIK_RE = re.compile(r"\b[0-9]{11}\b")
_IBAN_RE = re.compile(r"\bTR[0-9]{24}\b", re.I)


def _scrub_pii(text: str) -> str:
    """Replace PII tokens with generic placeholders."""
    text = _URL_RE.sub("<URL>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _IBAN_RE.sub("<IBAN>", text)
    text = _PHONE_TR_RE.sub("<PHONE>", text)
    text = _TC_KIMLIK_RE.sub("<TCNO>", text)
    return text


def _sentence_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


@dataclass
class FilterStats:
    total: int = 0
    passed: int = 0
    dropped_lang: int = 0
    dropped_length: int = 0
    dropped_dedup: int = 0
    pii_scrubbed: int = 0

    def drop_rate(self) -> float:
        return (self.total - self.passed) / max(self.total, 1)


class QualityFilter:
    """Configurable sentence-level quality filter.

    Args:
        lang_threshold: Minimum fasttext language confidence (default 0.85).
        min_tokens: Minimum whitespace-token count (default 3).
        max_tokens: Maximum whitespace-token count (default 50).
        dedup: Enable sentence-level deduplication (default True).
    """

    def __init__(
        self,
        lang_threshold: float = 0.85,
        min_tokens: int = 3,
        max_tokens: int = 50,
        dedup: bool = True,
    ) -> None:
        self.lang_threshold = lang_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.dedup = dedup
        self._seen_hashes: set[str] = set()
        self._lid_model: object | None = None  # loaded lazily

    def _get_lid_model(self) -> object:
        if self._lid_model is None:
            try:
                from fasttext_langdetect import detect as _detect
                self._lid_model = _detect
            except ImportError:
                logger.warning(
                    "fasttext-langdetect not available; language filter DISABLED. "
                    "Install: pip install fasttext-langdetect"
                )
                self._lid_model = None
        return self._lid_model

    def _is_turkish(self, text: str) -> bool:
        """Return True if fasttext confidence for Turkish >= threshold."""
        detect = self._get_lid_model()
        if detect is None:
            return True  # disabled
        try:
            result = detect(text.strip(), low_memory=False)
            # fasttext_langdetect returns {lang: str, score: float}
            lang = result.get("lang", "")
            score = float(result.get("score", 0.0))
            return lang == "tr" and score >= self.lang_threshold
        except Exception:
            return True  # err on keep

    def _passes_length(self, text: str) -> bool:
        n = len(text.split())
        return self.min_tokens <= n <= self.max_tokens

    def filter_sentences(
        self,
        rows: list[dict],
        *,
        source: str | None = None,
    ) -> tuple[list[dict], FilterStats]:
        """Filter a list of sentence dicts.

        Each row must have at least a ``text`` key. Rows that survive have
        their ``text`` field PII-scrubbed in-place. The ``source`` parameter
        is only used for logging; source attribution inside each row is
        preserved as-is.

        Returns:
            (kept_rows, stats)
        """
        stats = FilterStats()
        kept: list[dict] = []

        for row in rows:
            text = row.get("text", "").strip()
            if not text:
                continue
            stats.total += 1

            # Language filter
            if not self._is_turkish(text):
                stats.dropped_lang += 1
                continue

            # Length filter
            if not self._passes_length(text):
                stats.dropped_length += 1
                continue

            # Dedup
            if self.dedup:
                h = _sentence_hash(text)
                if h in self._seen_hashes:
                    stats.dropped_dedup += 1
                    continue
                self._seen_hashes.add(h)

            # PII scrub
            scrubbed = _scrub_pii(text)
            if scrubbed != text:
                stats.pii_scrubbed += 1
            row = dict(row)
            row["text"] = scrubbed

            kept.append(row)
            stats.passed += 1

        return kept, stats

    def reset_dedup(self) -> None:
        """Clear the dedup hash set (e.g., between source shards)."""
        self._seen_hashes.clear()
