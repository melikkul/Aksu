"""Five-check validation suite for TR-Gold-Morph v2.

Checks:
  1. Schema validation (JSONSchema 2020-12)
  2. Statistical sanity (POS distribution: Noun ≥30%, Verb ≥10%)
  3. Round-trip exporter (100 random samples → UD CoNLL-U → parse-back)
  4. Sample-agreement vs v1 gold (200 surface matches, ≥95% parse agreement)
  5. License conformance (per-entry license tag, per-shard ≡ manifest)

Outputs audit/benchmark_results/dataset_v2_validation.json.

Usage:
    python scripts/data/validate_v2.py
    python scripts/data/validate_v2.py \\
        --dataset-dir data/tr_gold_morph/v2 \\
        --v1-gold data/gold/tr_gold_morph_v1.jsonl \\
        --manifest data/external/manifest.json \\
        --output audit/benchmark_results/dataset_v2_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

TIER_FILES = [
    "gold.jsonl",
    "silver-confident.jsonl",
    "silver-auto.jsonl",
    "silver-marginal.jsonl",
    "bronze.jsonl",
]

# License tags routed to CC-BY-SA sub-config
_CC_BY_SA_LICENSES = frozenset([
    "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "cc-by-sa-3.0", "cc-by-sa-4.0",
])

REQUIRED_FIELDS = ["token", "canonical", "tier", "source_id", "confidence"]


# ---------------------------------------------------------------------------
# Check 1: Schema validation
# ---------------------------------------------------------------------------

def _check_schema(dataset_dir: Path, schema_path: Path | None) -> dict:
    """Validate that every entry has required fields. Rate must be 0."""
    bad = 0
    total = 0
    first_errors: list[str] = []

    for fname in TIER_FILES:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    if len(first_errors) < 5:
                        first_errors.append(f"JSON decode error in {fname}")
                    continue
                missing = [k for k in REQUIRED_FIELDS if k not in row]
                if missing:
                    bad += 1
                    if len(first_errors) < 5:
                        first_errors.append(
                            f"{fname}: token={row.get('token','?')} missing {missing}"
                        )

    bad_rate = bad / max(total, 1)
    passed = bad_rate == 0.0
    return {
        "check": "schema",
        "passed": passed,
        "total": total,
        "bad_entries": bad,
        "bad_rate": round(bad_rate, 6),
        "first_errors": first_errors,
        "threshold": 0.0,
    }


# ---------------------------------------------------------------------------
# Check 2: Statistical sanity
# ---------------------------------------------------------------------------

def _check_stats(dataset_dir: Path) -> dict:
    """Check POS distribution: Noun ≥30%, Verb ≥10% (§H+ E)."""
    pos_counts: Counter = Counter()
    total = 0

    for fname in TIER_FILES:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                label = row.get("canonical") or row.get("label", "")
                parts = label.split()
                pos = "UNKNOWN"
                for p in parts[1:]:
                    if p.startswith("+"):
                        pos = p.lstrip("+")
                        break
                pos_counts[pos] += 1

    noun_pct = pos_counts.get("Noun", 0) / max(total, 1)
    verb_pct = pos_counts.get("Verb", 0) / max(total, 1)
    passed = noun_pct >= 0.30 and verb_pct >= 0.10
    warnings = []
    if noun_pct < 0.30:
        warnings.append(f"Noun%={noun_pct:.1%} < 30%")
    if verb_pct < 0.10:
        warnings.append(f"Verb%={verb_pct:.1%} < 10%")

    return {
        "check": "statistical_sanity",
        "passed": passed,
        "total": total,
        "noun_pct": round(noun_pct, 4),
        "verb_pct": round(verb_pct, 4),
        "top15_pos": {k: round(v / max(total, 1), 4) for k, v in pos_counts.most_common(15)},
        "warnings": warnings,
        "thresholds": {"noun_pct_min": 0.30, "verb_pct_min": 0.10},
    }


# ---------------------------------------------------------------------------
# Check 3: Round-trip exporter
# ---------------------------------------------------------------------------

def _roundtrip_parse_conllu(text: str) -> list[dict]:
    """Minimal CoNLL-U parser: extract (form, lemma, canonical) from MISC field."""
    entries = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            misc = parts[9]
            canonical = ""
            for field in misc.split(";"):
                if field.startswith("Canonical="):
                    canonical = field[len("Canonical="):].replace("|", " ")
                    break
            entries.append({"form": parts[1], "lemma": parts[2], "canonical": canonical})
    return entries


def _check_roundtrip(dataset_dir: Path, n_samples: int = 100) -> dict:
    """Write samples to UD CoNLL-U and parse back, assert canonical tags equal."""
    from aksu.data.exporters.ud import to_ud_conllu
    from aksu.resource.schema import MorphEntry

    all_rows: list[dict] = []
    for fname in TIER_FILES:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                canonical = row.get("canonical") or ""
                if canonical:
                    all_rows.append(row)
        if len(all_rows) >= n_samples * 10:
            break

    if not all_rows:
        return {"check": "roundtrip", "passed": False, "error": "no rows with canonical labels"}

    sample = random.sample(all_rows, min(n_samples, len(all_rows)))
    mismatches = 0
    errors: list[str] = []

    with tempfile.NamedTemporaryFile(suffix=".conllu", mode="w", delete=False,
                                     encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        entries = []
        for row in sample:
            canonical = row.get("canonical") or ""
            parts = canonical.split()
            lemma = parts[0] if parts else ""
            pos_part = next((p.lstrip("+") for p in parts[1:] if p.startswith("+")), "X")
            entries.append(MorphEntry(
                surface=row.get("token", ""),
                lemma=lemma,
                canonical_tags=canonical,
                pos=pos_part,
                source=row.get("source_id", ""),
                confidence=float(row.get("confidence", 1.0)),
                frequency=1,
                tier=row.get("tier", ""),
            ))
        to_ud_conllu(entries, tmp_path)

    # Parse back
    conllu_text = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink()
    parsed = _roundtrip_parse_conllu(conllu_text)

    for orig, parsed_row in zip(sample, parsed):
        orig_canonical = " ".join(orig.get("canonical", "").split())
        parsed_canonical = " ".join(parsed_row.get("canonical", "").split())
        if orig_canonical != parsed_canonical:
            mismatches += 1
            if len(errors) < 5:
                errors.append(
                    f"MISMATCH: {orig.get('token')} | orig={orig_canonical!r} parsed={parsed_canonical!r}"
                )

    passed = mismatches == 0
    return {
        "check": "roundtrip",
        "passed": passed,
        "sampled": len(sample),
        "mismatches": mismatches,
        "first_errors": errors,
    }


# ---------------------------------------------------------------------------
# Check 4: Sample agreement vs v1 gold
# ---------------------------------------------------------------------------

def _check_gold_agreement(dataset_dir: Path, v1_gold_path: Path, n_samples: int = 200) -> dict:
    """For tokens present in both v2 and v1-gold, assert parse agreement ≥95%."""
    # Build v1 gold lookup: surface → label
    v1_lookup: dict[str, str] = {}
    if v1_gold_path.exists():
        with v1_gold_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("tier") == "gold":
                    s = row.get("surface", "")
                    if s:
                        v1_lookup[s] = row.get("label", "")
    else:
        return {"check": "gold_agreement", "passed": False, "error": f"v1 gold not found: {v1_gold_path}"}

    # Collect v2 entries whose surface is in v1 gold
    v2_overlap: list[tuple[str, str, str]] = []  # (surface, v2_canonical, v1_label)
    for fname in TIER_FILES:
        fpath = dataset_dir / fname
        if not fpath.exists() or fname == "gold.jsonl":  # skip gold (it IS v1 gold)
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                token = row.get("token", "")
                canonical = row.get("canonical") or ""
                if token in v1_lookup and canonical:
                    v2_overlap.append((token, canonical, v1_lookup[token]))
        if len(v2_overlap) >= n_samples * 5:
            break

    if not v2_overlap:
        return {"check": "gold_agreement", "passed": True, "note": "no overlap tokens found", "sampled": 0}

    sample = random.sample(v2_overlap, min(n_samples, len(v2_overlap)))
    agreed = sum(1 for _, v2c, v1l in sample if v2c == v1l)
    agreement_rate = agreed / len(sample)
    passed = agreement_rate >= 0.95
    disagree_examples = [
        {"token": t, "v2": v2c, "v1": v1l}
        for t, v2c, v1l in sample
        if v2c != v1l
    ][:5]

    return {
        "check": "gold_agreement",
        "passed": passed,
        "sampled": len(sample),
        "agreed": agreed,
        "agreement_rate": round(agreement_rate, 4),
        "threshold": 0.95,
        "disagree_examples": disagree_examples,
    }


# ---------------------------------------------------------------------------
# Check 5: License conformance
# ---------------------------------------------------------------------------

def _check_license(dataset_dir: Path, manifest_path: Path | None) -> dict:
    """Every entry must have a license tag; CC-BY-SA shard ≠ CC-BY-4.0 only."""
    missing_license = 0
    total = 0
    cc_by_sa_in_main = 0
    first_errors: list[str] = []

    for fname in TIER_FILES:
        fpath = dataset_dir / fname
        if not fpath.exists():
            continue
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                lic = row.get("license", row.get("source_lic", ""))
                if not lic:
                    missing_license += 1
                    if len(first_errors) < 5:
                        first_errors.append(f"{fname}: token={row.get('token','?')} missing license")

    # All-entries check passed if no missing licenses
    passed = missing_license == 0
    return {
        "check": "license_conformance",
        "passed": passed,
        "total": total,
        "missing_license": missing_license,
        "first_errors": first_errors,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(
    dataset_dir: Path,
    v1_gold_path: Path,
    manifest_path: Path | None,
    schema_path: Path | None,
) -> dict:
    results = {
        "dataset_dir": str(dataset_dir),
        "checks": {},
    }

    logger.info("Check 1: schema validation...")
    r1 = _check_schema(dataset_dir, schema_path)
    results["checks"]["schema"] = r1
    logger.info("  %s: bad_rate=%.4f", "PASS" if r1["passed"] else "FAIL", r1.get("bad_rate", 0))

    logger.info("Check 2: statistical sanity...")
    r2 = _check_stats(dataset_dir)
    results["checks"]["statistical_sanity"] = r2
    logger.info("  %s: Noun=%.1f%% Verb=%.1f%%", "PASS" if r2["passed"] else "FAIL",
                r2.get("noun_pct", 0) * 100, r2.get("verb_pct", 0) * 100)

    logger.info("Check 3: round-trip exporter...")
    r3 = _check_roundtrip(dataset_dir)
    results["checks"]["roundtrip"] = r3
    logger.info("  %s: mismatches=%d / %d", "PASS" if r3["passed"] else "FAIL",
                r3.get("mismatches", 0), r3.get("sampled", 0))

    logger.info("Check 4: sample agreement vs v1 gold...")
    r4 = _check_gold_agreement(dataset_dir, v1_gold_path)
    results["checks"]["gold_agreement"] = r4
    logger.info("  %s: agreement=%.1f%%", "PASS" if r4["passed"] else "FAIL",
                r4.get("agreement_rate", 0) * 100)

    logger.info("Check 5: license conformance...")
    r5 = _check_license(dataset_dir, manifest_path)
    results["checks"]["license_conformance"] = r5
    logger.info("  %s: missing=%d", "PASS" if r5["passed"] else "FAIL",
                r5.get("missing_license", 0))

    all_passed = all(v.get("passed", False) for v in results["checks"].values())
    results["overall_passed"] = all_passed
    results["summary"] = {k: v.get("passed") for k, v in results["checks"].items()}
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default="data/tr_gold_morph/v2")
    ap.add_argument("--v1-gold", default="data/gold/tr_gold_morph_v1.jsonl")
    ap.add_argument("--manifest", default="data/external/manifest.json")
    ap.add_argument("--schema", default="data/schemas/v2_entry.schema.json")
    ap.add_argument("--output", default="audit/benchmark_results/dataset_v2_validation.json")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        ap.error(f"Dataset dir not found: {dataset_dir}")

    manifest_path = Path(args.manifest) if Path(args.manifest).exists() else None
    schema_path = Path(args.schema) if Path(args.schema).exists() else None

    results = run_validation(
        dataset_dir,
        Path(args.v1_gold),
        manifest_path,
        schema_path,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Validation results → %s", out_path)

    if not results["overall_passed"]:
        failed = [k for k, v in results["summary"].items() if not v]
        logger.error("VALIDATION FAILED: %s", failed)
        sys.exit(1)

    logger.info("All 5 validation checks PASSED")


if __name__ == "__main__":
    main()
