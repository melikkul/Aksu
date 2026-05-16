"""Export TR-Gold-Morph v2 to HuggingFace Datasets format.

Builds a datasets.DatasetDict with two configs:
  v2-main     — CC-BY-4.0 entries (oscar-tr, mc4-tr, parlamint-tr)
  v2-cc-by-sa — CC-BY-SA-3.0/4.0 entries (wiki-tr, boun-ud gold)

Usage:
    python -m aksu.data.exporters.hf_dataset \\
        --input-dir data/tr_gold_morph/v2 \\
        --output-dir data/tr_gold_morph/v2/hf_format
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# License → sub-config routing
_CC_BY_SA_LICENSES = frozenset([
    "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "cc-by-sa-3.0", "cc-by-sa-4.0",
])

TIER_FILES = [
    "gold.jsonl",
    "silver-confident.jsonl",
    "silver-auto.jsonl",
    "silver-marginal.jsonl",
    "bronze.jsonl",
]


def _route_row(row: dict) -> str:
    """Return 'v2-cc-by-sa' or 'v2-main' based on license tag."""
    lic = row.get("license", row.get("source_lic", ""))
    if lic in _CC_BY_SA_LICENSES:
        return "v2-cc-by-sa"
    return "v2-main"


def build_hf_dataset(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Load per-tier files, split by license, save as HF Datasets.

    Returns counts per config.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        raise RuntimeError(
            "HuggingFace 'datasets' library not installed. "
            "Run: pip install datasets"
        )

    main_rows: list[dict] = []
    cc_by_sa_rows: list[dict] = []
    counts: dict[str, int] = {"v2-main": 0, "v2-cc-by-sa": 0}

    for fname in TIER_FILES:
        fpath = input_dir / fname
        if not fpath.exists():
            logger.warning("Missing tier file: %s", fpath)
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                config = _route_row(row)
                if config == "v2-cc-by-sa":
                    cc_by_sa_rows.append(row)
                    counts["v2-cc-by-sa"] += 1
                else:
                    main_rows.append(row)
                    counts["v2-main"] += 1

    logger.info("v2-main: %d entries, v2-cc-by-sa: %d entries", counts["v2-main"], counts["v2-cc-by-sa"])

    output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_rows(rows: list[dict]) -> list[dict]:
        """Ensure consistent schema for HF Dataset (all rows same keys)."""
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        normalized = []
        for r in rows:
            nr = {k: r.get(k) for k in all_keys}
            normalized.append(nr)
        return normalized

    dd = DatasetDict()

    if main_rows:
        dd["v2-main"] = Dataset.from_list(_normalize_rows(main_rows))

    if cc_by_sa_rows:
        dd["v2-cc-by-sa"] = Dataset.from_list(_normalize_rows(cc_by_sa_rows))

    dd.save_to_disk(str(output_dir))
    logger.info("HF DatasetDict saved to %s", output_dir)

    # Write README card stub
    card_path = output_dir / "README.md"
    card_path.write_text(
        "# TR-Gold-Morph v2\n\n"
        "Turkish morphological atomizer corpus.\n\n"
        "## Configs\n"
        "- `v2-main`: CC-BY-4.0 (OSCAR, mC4, ParlaMint)\n"
        "- `v2-cc-by-sa`: CC-BY-SA-3.0/4.0 (Wikipedia-tr, BOUN gold)\n\n"
        "See `data/external/manifest.json` for full source provenance.\n",
        encoding="utf-8",
    )

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data/tr_gold_morph/v2")
    ap.add_argument("--output-dir", default="data/tr_gold_morph/v2/hf_format")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        ap.error(f"Input dir not found: {input_dir}")

    counts = build_hf_dataset(input_dir, Path(args.output_dir))
    logger.info("Done: %s", counts)


if __name__ == "__main__":
    main()
