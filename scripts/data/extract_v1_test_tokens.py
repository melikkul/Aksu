"""Extract surface forms from the v1 test split for v2 contamination exclusion.

Reads data/splits/test.jsonl and writes every unique surface form to
data/intermediate/v1_test_surface_forms.txt (one per line).

The preprocess pipeline loads this file and skips any token whose surface
matches, keeping v2 clean for downstream evaluation against the v1 test set.

Usage:
    python scripts/data/extract_v1_test_tokens.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(test_path: Path, output_path: Path) -> int:
    surfaces: set[str] = set()
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            s = row.get("surface", "").strip()
            if s:
                surfaces.add(s)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in sorted(surfaces):
            f.write(s + "\n")

    logger.info("Wrote %d unique v1 test surfaces → %s", len(surfaces), output_path)
    return len(surfaces)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-split", default="data/splits/test.jsonl")
    ap.add_argument("--output", default="data/intermediate/v1_test_surface_forms.txt")
    args = ap.parse_args()

    test_path = Path(args.test_split)
    if not test_path.exists():
        ap.error(f"Test split not found: {test_path}")

    extract(test_path, Path(args.output))


if __name__ == "__main__":
    main()
