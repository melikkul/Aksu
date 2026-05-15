# Phase 0.5.1 — TR-Gold-Morph Provenance

## Finding: The 2.5M resource is a SQLite DB generated on TRUBA (not in this checkout)

### Evidence
- `data/resource/tr_gold_morph.db` referenced by `src/resource/training_bridge.py:137`, `src/resource/generation_runner.py:37`
- `data/resource/` directory is **EMPTY** in this checkout
- `scripts/truba/submit_v5_dh.sh:24` references `data/resource/training_export_2.5M.jsonl` (not present)
- `models/vocabs/root_vocab_2.5M.json` (760 KB) and `word_vocab_2.5M.json` (17 MB) ARE in the repo — vocabs derived from the 2.5M DB
- STATUS.md:1107: "Total: 2,512,034 entries (up from 666,741)"
- DVC remote: local_storage at /arf/home/scolakoglu/dvc_cache — no network remote

### What the on-disk file IS
- `data/gold/tr_gold_morph_v1.jsonl` (80,582 rows) is the **gold/silver-annotated training corpus** (labeled subset)
- `stats.json:total_tokens = 80537`

### Impact on D1 verdict
- README claim "2,512,034 entries" refers to the full generated morphological resource DB, NOT the on-disk JSONL
- The JSONL is a different artifact: the labeled training corpus  
- D1: **REFUTED with context** — the 2.5M number is real (per STATUS.md) but refers to a database not in this checkout; the primary on-disk file has 80K entries

## Finding: Ensemble results already stored

`models/v6/ensemble_results.json` contains:
- Individual seeds: {42: 0.9821, 123: 0.9817, 456: 0.9798, 789: 0.9828, 1337: 0.9810}
- Range: 97.98%–98.28%, std ≈ 0.12pp
- Ensemble: **0.9832 (98.32%)** — rounds to README headline 98.3% ✓
- TLF-2 CONFIRMED: headline is ensemble; single seeds cluster at 97.98–98.28%

## DVC status: local_storage at /arf/home/scolakoglu/dvc_cache (no cloud remote)
