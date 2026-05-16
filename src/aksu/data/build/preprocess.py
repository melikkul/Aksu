"""Preprocessing pipeline for TR-Gold-Morph corpus harvest.

Produces three artifacts in data/intermediate/:
  tokens.jsonl       — one row per token occurrence
  sentences.jsonl    — one row per unique sentence (de-duplicated)
  token_sentences.jsonl — one row per unique token with up to 32 sentence refs

Usage:
    python -m aksu.data.build.preprocess --shard oscar-tr --max-tokens 3000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_SENTENCE_REFS = 32  # cap per token to avoid huge lists for high-freq types


def _sentence_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tokenize_simple(text: str) -> list[str]:
    """Whitespace tokenizer; replaced by Stanza-tr when available."""
    return re.findall(r"\S+", text)


def _load_v1_test_surfaces(path: Path) -> frozenset[str]:
    """Load v1 test-split surface forms for contamination exclusion."""
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip())


def preprocess_shard(
    rows: list[dict],
    source_name: str,
    source_license: str,
    *,
    output_dir: Path,
    max_tokens: int | None = None,
    quality_filter: object | None = None,
    v1_test_surfaces: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """Process a list of sentence dicts into the three intermediate files.

    Each row should have at least a ``text`` field; ``source`` and ``license``
    fields are preserved if present, otherwise ``source_name``/``source_license``
    are used as fallbacks.

    Returns stats dict with token_count, unique_sentence_count, unique_token_count,
    v1_test_contamination_skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / "tokens.jsonl"
    sentences_path = output_dir / "sentences.jsonl"
    token_sents_path = output_dir / "token_sentences.jsonl"

    # Apply quality filter if provided
    if quality_filter is not None:
        rows, qstats = quality_filter.filter_sentences(rows, source=source_name)
        logger.info(
            "Quality filter: %d kept / %d total (lang_drop=%d len_drop=%d dedup_drop=%d pii_scrub=%d)",
            qstats.passed, qstats.total,
            qstats.dropped_lang, qstats.dropped_length,
            qstats.dropped_dedup, qstats.pii_scrubbed,
        )

    seen_sentences: set[str] = set()
    token_to_sids: dict[str, list[str]] = defaultdict(list)

    token_count = 0
    unique_sentence_count = 0
    v1_test_skipped = 0

    try:
        import stanza
        pipeline = stanza.Pipeline("tr", processors="tokenize", verbose=False)
        use_stanza = True
    except (ImportError, Exception):
        use_stanza = False
        logger.warning("Stanza unavailable; using whitespace tokenizer")

    with tokens_path.open("a", encoding="utf-8") as tf, \
         sentences_path.open("a", encoding="utf-8") as sf:

        for row in rows:
            text = row.get("text", "") if isinstance(row, dict) else str(row)
            if not text or not text.strip():
                continue

            # Per-row source/license override shard defaults
            row_source = (row.get("source") if isinstance(row, dict) else None) or source_name
            row_license = (row.get("license") if isinstance(row, dict) else None) or source_license

            # Sentence segmentation
            if use_stanza:
                doc = pipeline(text)
                sents = [s.text for s in doc.sentences]
            else:
                sents = [text.strip()]

            for sent_text in sents:
                sid = _sentence_id(sent_text)

                if sid not in seen_sentences:
                    seen_sentences.add(sid)
                    sf.write(json.dumps({
                        "sentence_id": sid,
                        "text": sent_text,
                        "source": row_source,
                        "source_lic": row_license,
                    }, ensure_ascii=False) + "\n")
                    unique_sentence_count += 1

                tokens = _tokenize_simple(sent_text)
                for pos, token in enumerate(tokens):
                    if not token:
                        continue

                    # v1 test contamination exclusion
                    if token in v1_test_surfaces:
                        v1_test_skipped += 1
                        continue

                    tf.write(json.dumps({
                        "token": token,
                        "sentence_id": sid,
                        "source": row_source,
                        "source_lic": row_license,
                        "position": pos,
                    }, ensure_ascii=False) + "\n")
                    token_count += 1

                    refs = token_to_sids[token]
                    if len(refs) < _MAX_SENTENCE_REFS and sid not in refs:
                        refs.append(sid)

                if max_tokens and token_count >= max_tokens:
                    logger.info("max_tokens=%d reached", max_tokens)
                    break
            if max_tokens and token_count >= max_tokens:
                break

    # Write token→sentence index
    unique_token_count = len(token_to_sids)
    with token_sents_path.open("a", encoding="utf-8") as tsf:
        for token, sids in token_to_sids.items():
            tsf.write(json.dumps({
                "token": token,
                "sentence_ids": sids,
            }, ensure_ascii=False) + "\n")

    return {
        "token_count": token_count,
        "unique_sentence_count": unique_sentence_count,
        "unique_token_count": unique_token_count,
        "v1_test_contamination_skipped": v1_test_skipped,
    }


def _load_local_jsonl(path: Path, text_field: str = "text") -> list[dict]:
    """Read sentence rows from a pre-downloaded JSONL file.

    Returns a list of dicts, each with at least a ``text`` key.
    Preserves ``source`` and ``license`` fields if present in the row.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"text": line})
                continue
            if isinstance(obj, str):
                rows.append({"text": obj})
            elif isinstance(obj, dict):
                text = obj.get(text_field) or obj.get("sentence") or obj.get("text") or ""
                if text:
                    row: dict = {"text": text}
                    if "source" in obj:
                        row["source"] = obj["source"]
                    if "license" in obj:
                        row["license"] = obj["license"]
                    rows.append(row)
    return [r for r in rows if r.get("text", "").strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="Source name (e.g. oscar-tr)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--output-dir", default="data/intermediate")
    ap.add_argument("--dry-run", action="store_true", help="Print stats only")
    ap.add_argument(
        "--local-jsonl",
        help="Path to a pre-downloaded JSONL file (one sentence/text per line). "
             "Use this on HPC nodes without internet access. "
             "If not given, streams from HuggingFace (requires internet).",
    )
    ap.add_argument("--lang-threshold", type=float, default=0.85)
    ap.add_argument("--no-quality-filter", action="store_true")
    ap.add_argument(
        "--v1-test-surfaces",
        default="data/intermediate/v1_test_surface_forms.txt",
        help="Path to v1 test-split surface forms for contamination exclusion.",
    )
    args = ap.parse_args()

    from aksu.data.build.sources import SOURCES
    source = next((s for s in SOURCES if s.name == args.shard), None)
    if source is None:
        ap.error(f"Unknown shard {args.shard!r}. Available: {[s.name for s in SOURCES]}")

    if args.local_jsonl:
        local_path = Path(args.local_jsonl)
        if not local_path.exists():
            ap.error(f"--local-jsonl path does not exist: {local_path}")
        logger.info("Loading shard %s from local file %s ...", source.name, local_path)
        rows = _load_local_jsonl(local_path)
    else:
        logger.info("Loading shard %s from HuggingFace %s ...", source.name, source.url)
        try:
            from datasets import load_dataset
            ds = load_dataset(source.url, split="train", streaming=True)
            rows = [
                {"text": row.get("text") or row.get("sentence") or ""}
                for row in ds
                if row.get("text") or row.get("sentence")
            ]
        except Exception as e:
            logger.error("Could not load shard: %s", e)
            raise

    quality_filter = None
    if not args.no_quality_filter:
        from aksu.data.build.quality_filter import QualityFilter
        quality_filter = QualityFilter(lang_threshold=args.lang_threshold)

    v1_test_surfaces = _load_v1_test_surfaces(Path(args.v1_test_surfaces))
    if v1_test_surfaces:
        logger.info("Loaded %d v1 test surfaces for exclusion", len(v1_test_surfaces))

    if args.dry_run:
        logger.info("DRY RUN: first 10 rows from shard:")
        for r in rows[:10]:
            logger.info("  %r", r.get("text", "")[:80])
        return

    stats = preprocess_shard(
        rows,
        source_name=source.name,
        source_license=source.license,
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        quality_filter=quality_filter,
        v1_test_surfaces=v1_test_surfaces,
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
