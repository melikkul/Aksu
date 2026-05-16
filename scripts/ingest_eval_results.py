"""Ingest eval_disambiguator.py output into metrics.json.

Handles the nested per-seed + ensemble structure that ingest_metrics.py
cannot address with simple key mapping.

Usage:
    python scripts/ingest_eval_results.py \
        --source models/v6/eval_results.json \
        --target audit/benchmark_results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", default="audit/benchmark_results/metrics.json")
    args = ap.parse_args()

    src = json.loads(Path(args.source).read_text())
    tgt_path = Path(args.target)
    tgt = json.loads(tgt_path.read_text())

    # Collect per-seed em_string values (skip error entries)
    em_string_vals: list[float] = []
    em_argmax_vals: list[float] = []
    for key, val in src.items():
        if key == "ensemble":
            continue
        if "error" in val:
            print(f"[warn] seed {key!r} had error: {val['error']!r} — skipping")
            continue
        em_string_vals.append(val["em_string"])
        em_argmax_vals.append(val["em_argmax"])

    if not em_string_vals:
        print("ERROR: no valid per-seed results found in source JSON")
        raise SystemExit(1)

    ensemble = src.get("ensemble", {})

    updates: dict[str, object] = {
        "em_string_ensemble": ensemble.get("em_string_mean") or statistics.mean(em_string_vals),
        "em_string_single_seed_min": min(em_string_vals),
        "em_string_single_seed_max": max(em_string_vals),
    }

    # Also re-check em_argmax_ensemble and std in case they changed
    if em_argmax_vals:
        measured_em_argmax = ensemble.get("em_argmax_mean") or statistics.mean(em_argmax_vals)
        updates["em_argmax_ensemble"] = measured_em_argmax
        if len(em_argmax_vals) > 1:
            updates["em_argmax_std"] = ensemble.get("em_argmax_std") or statistics.stdev(em_argmax_vals)

    print("Ingesting:")
    for k, v in updates.items():
        old = tgt.get(k)
        if old is not None and old != v:
            print(f"  {k}: {old!r} → {v!r}  (overwrite)")
        else:
            print(f"  {k}: {v!r}")
        tgt[k] = v
        tgt.setdefault("notes", {})[k] = f"measured: SLURM v6 eval from {args.source}"

    # Honesty check: em_string_ensemble vs em_argmax_ensemble
    em_string = tgt.get("em_string_ensemble") or 0.0
    em_argmax = tgt.get("em_argmax_ensemble") or 0.0
    diff_pp = (em_argmax - em_string) * 100
    print(f"\nem_argmax_ensemble={em_argmax:.4f}  em_string_ensemble={em_string:.4f}  diff={diff_pp:.2f}pp")
    if diff_pp > 1.0 and em_string < 0.983:
        print("WARNING: em_string more than 1pp below em_argmax AND below 0.983")
        print("  → Halt condition §7 rule 2 triggered. The README headline must shift to em_string.")
        print("  → Write audit/halt_reports/<date>-em-string-gap.md before final publication.")

    tgt_path.write_text(json.dumps(tgt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {tgt_path}")


if __name__ == "__main__":
    main()
