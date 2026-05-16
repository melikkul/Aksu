"""Compute statistics for TR-Gold-Morph v2 dataset.

Reads the per-tier JSONL files in data/tr_gold_morph/v2/ and emits
data/tr_gold_morph/v2/stats.json.

Usage:
    python scripts/data/dataset_stats.py
    python scripts/data/dataset_stats.py --input-dir data/tr_gold_morph/v2 --output data/tr_gold_morph/v2/stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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


def _zipf_exponent(freq_counter: Counter) -> float:
    """Estimate Zipf exponent via linear regression on log-rank vs log-freq."""
    items = freq_counter.most_common()
    if len(items) < 10:
        return float("nan")
    xs, ys = [], []
    for rank, (_, freq) in enumerate(items[:5000], 1):
        if freq > 0:
            xs.append(math.log(rank))
            ys.append(math.log(freq))
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float("nan")
    slope = (n * sxy - sx * sy) / denom
    return -slope  # Zipf exponent is negative slope


def _compute_shard_stats(path: Path) -> dict:
    """Compute per-tier stats from a single JSONL file."""
    total = 0
    with_boundary = 0
    pos_counts: Counter = Counter()
    root_counts: Counter = Counter()
    source_counts: Counter = Counter()
    sent_len_sum = 0
    sent_len_hist: Counter = Counter()
    ambiguous = 0
    confidence_sum = 0.0
    oov_count = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            label = row.get("label") or row.get("canonical", "")
            source = row.get("source", "unknown")
            source_counts[source] += 1

            if row.get("boundaries"):
                with_boundary += 1

            if len(row.get("candidates", [])) > 1:
                ambiguous += 1

            confidence_sum += float(row.get("confidence", 1.0))

            if row.get("method") == "dualhead_pending":
                oov_count += 1

            # POS extraction — first +Tag after lemma
            parts = label.split()
            pos = "UNKNOWN"
            for part in parts[1:]:
                if part.startswith("+"):
                    pos = part.lstrip("+")
                    break
            pos_counts[pos] += 1

            # Root (first token of label)
            if parts:
                root_counts[parts[0]] += 1

            # Sentence-length proxy: # tokens in sentence context
            ctx = row.get("context_sentence", "")
            n_tok = len(ctx.split()) if ctx else 0
            sent_len_sum += n_tok
            bucket = (n_tok // 5) * 5
            sent_len_hist[str(bucket)] += 1

    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "boundary_coverage": round(with_boundary / total, 4),
        "ambiguity_rate": round(ambiguous / total, 4),
        "mean_confidence": round(confidence_sum / total, 4),
        "oov_count": oov_count,
        "pos_distribution": {
            k: round(v / total, 4)
            for k, v in pos_counts.most_common(15)
        },
        "top10_roots": [k for k, _ in root_counts.most_common(10)],
        "zipf_exponent": round(_zipf_exponent(root_counts), 4),
        "source_counts": dict(source_counts),
        "sent_len_histogram": dict(sorted(sent_len_hist.items(), key=lambda x: int(x[0]))),
    }


def compute_dataset_stats(input_dir: Path) -> dict:
    """Compute aggregate stats for all tiers."""
    tier_stats: dict[str, dict] = {}
    total_entries = 0
    total_boundary = 0
    all_pos: Counter = Counter()
    all_source: Counter = Counter()

    # Per-tier stats
    for fname in TIER_FILES:
        fpath = input_dir / fname
        if not fpath.exists():
            logger.warning("Missing tier file: %s", fpath)
            tier = fname.replace(".jsonl", "")
            tier_stats[tier] = {"total": 0}
            continue
        tier = fname.replace(".jsonl", "")
        logger.info("Computing stats for %s ...", fname)
        stats = _compute_shard_stats(fpath)
        tier_stats[tier] = stats
        total_entries += stats.get("total", 0)
        total_boundary += int(stats.get("total", 0) * stats.get("boundary_coverage", 0))
        for pos, frac in stats.get("pos_distribution", {}).items():
            all_pos[pos] += int(frac * stats.get("total", 0))
        for src, cnt in stats.get("source_counts", {}).items():
            all_source[src] += cnt

    # Also account for OOV file
    oov_path = input_dir / "oov.jsonl"
    oov_total = 0
    if oov_path.exists():
        oov_total = sum(1 for l in oov_path.open(encoding="utf-8") if l.strip())

    overall_boundary_cov = total_boundary / total_entries if total_entries else 0.0

    # POS sanity check (§H+ E: Noun ≥30%, Verb ≥10%)
    noun_pct = all_pos.get("Noun", 0) / max(total_entries, 1)
    verb_pct = all_pos.get("Verb", 0) / max(total_entries, 1)
    sanity_ok = noun_pct >= 0.30 and verb_pct >= 0.10
    sanity_warnings = []
    if noun_pct < 0.30:
        sanity_warnings.append(f"Noun%={noun_pct:.1%} < 30% threshold — check POS tagging")
    if verb_pct < 0.10:
        sanity_warnings.append(f"Verb%={verb_pct:.1%} < 10% threshold — check POS tagging")

    # Build v1-compat 3-tier breakdown for reference
    v1_silver_auto = (
        tier_stats.get("silver-confident", {}).get("total", 0)
        + tier_stats.get("silver-auto", {}).get("total", 0)
    )
    v1_silver_agreed = tier_stats.get("silver-marginal", {}).get("total", 0)

    return {
        "total_entries": total_entries,
        "tier_breakdown": {tier: s.get("total", 0) for tier, s in tier_stats.items()},
        "oov_dropped": oov_total,
        "boundary_coverage": round(overall_boundary_cov, 4),
        "boundary_coverage_note": (
            "WARNING: boundary coverage <80%" if overall_boundary_cov < 0.80 else "OK"
        ),
        "source_counts": dict(all_source),
        "pos_sanity": {
            "ok": sanity_ok,
            "noun_pct": round(noun_pct, 4),
            "verb_pct": round(verb_pct, 4),
            "warnings": sanity_warnings,
        },
        "v1_compat_breakdown": {
            "gold": tier_stats.get("gold", {}).get("total", 0),
            "silver-auto": v1_silver_auto,
            "silver-agreed": v1_silver_agreed,
        },
        "bronze_note": "research use only — not part of primary release",
        "tier_stats": tier_stats,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data/tr_gold_morph/v2")
    ap.add_argument("--output", default="data/tr_gold_morph/v2/stats.json")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        ap.error(f"Input dir not found: {input_dir}")

    stats = compute_dataset_stats(input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Stats written to %s", out)
    logger.info("Total entries: %d", stats["total_entries"])
    logger.info("Boundary coverage: %.1f%%", stats["boundary_coverage"] * 100)
    logger.info("Tier breakdown: %s", stats["tier_breakdown"])
    if stats["pos_sanity"]["warnings"]:
        for w in stats["pos_sanity"]["warnings"]:
            logger.warning("SANITY: %s", w)


if __name__ == "__main__":
    main()
