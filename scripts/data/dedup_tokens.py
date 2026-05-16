"""Deduplicate tokens.jsonl → unique_tokens.jsonl.

Reads data/intermediate/tokens.jsonl, aggregates by token surface form,
and emits one record per unique token with frequency count.

Usage:
    python scripts/data/dedup_tokens.py
    python scripts/data/dedup_tokens.py \\
        --input  data/intermediate/tokens.jsonl \\
        --output data/intermediate/unique_tokens.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def dedup(input_path: Path, output_path: Path) -> dict[str, int]:
    freq: Counter = Counter()
    source_map: dict[str, str] = {}

    logger.info("Reading %s ...", input_path)
    with input_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            token = row["token"]
            freq[token] += 1
            if token not in source_map:
                source_map[token] = row.get("source", "unknown")
            if i % 1_000_000 == 0 and i > 0:
                logger.info("  %d lines read, %d unique tokens so far", i, len(freq))

    logger.info("Writing %d unique tokens to %s ...", len(freq), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for token, count in freq.most_common():
            f.write(
                json.dumps(
                    {
                        "token": token,
                        "frequency": count,
                        "source": source_map[token],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"unique_tokens": len(freq), "total_occurrences": sum(freq.values())}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/intermediate/tokens.jsonl")
    ap.add_argument("--output", default="data/intermediate/unique_tokens.jsonl")
    args = ap.parse_args()

    stats = dedup(Path(args.input), Path(args.output))
    logger.info("Done: %s", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
