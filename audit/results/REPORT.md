# kök-türk / KokTurk — Independent Reproducibility Audit Report

**Auditor:** Claude Opus 4.7 (Anthropic), acting as independent ML reproducibility auditor  
**Audit date:** 2026-05-15  
**Target commit:** `834ff8e5ca62ea5179e7e82a37e3edf0bb63bb8a`  
**Branch:** `main`  
**Repository identity note:** Master prompt names the system "Aksu" at `github.com/melikkul/Aksu`. Actual `git remote` is `https://github.com/melikkul/KokTurk.git`. The project self-names as **kök-türk / kokturk**. Audit proceeds against the local checkout at the pinned commit.

---

## §1. Executive Summary

### Top-Level Findings

**TLF-1 — "Exact Match" is candidate-index argmax, not full-parse string equality.**  
`src/train/train_disambiguator.py:253–272` defines EM as `(preds == gold_indices).sum() / total`, where `preds` and `gold_indices` are *positions* in Zeyrek's candidate list. The comparison tables in the README (MorseDisamb 98.59%, TransMorph 96.25%, etc.) report full-string parse equality. The metrics are **not directly comparable**. Live re-verification of string-equality EM was blocked by the login node's CPU time limit (see Environment §2).

**TLF-2 — Headline 98.3% is a 5-seed ensemble; single-seed range is 97.98–98.28%.**  
`models/v6/ensemble_results.json` stores all five seed results. README presents 98.3% without ensemble disclosure. Stored ensemble EM = **98.32%** (0.02pp above claim). Single-seed statistics: min=97.98%, median=98.17%, max=98.28%, stdev=0.114pp. Ensemble lift = 0.15pp (modest and appropriate). This is a transparency issue, not a fabrication.

**TLF-3 — Paired bootstrap + Holm-Bonferroni code exists but is NEVER wired into the headline benchmark runner.**  
`src/benchmark/significance.py` implements `paired_bootstrap_test(n_bootstrap=10000)` and `holm_bonferroni_correction` (all tests pass). `src/benchmark/run_all_benchmarks.py` imports neither. The README statement *"All differences tested via paired bootstrap (10K iterations, Holm-Bonferroni corrected)"* is **REFUTED at code level** — the significance code is present but disconnected from production.

### Quantitative Verdict Summary

```
CONFIRMED:          10
PARTIALLY CONFIRMED: 8
REFUTED:             7
UNVERIFIABLE:       11
TOTAL CLAIMS:       36
```

### Qualitative Overall Verdict: **PARTIALLY REPRODUCIBLE**

The core accuracy claim (A1 = 98.3%) is consistent with stored results from training runs. However: (1) three of the five README API examples produce wrong output (F2/F3 REFUTED), (2) the full pipeline instantiation is broken at runtime due to a missing import (F1 REFUTED), (3) the DVC pipeline cannot be reproduced end-to-end from the checked-out state (G1 REFUTED), (4) the significance testing infrastructure is disconnected (TLF-3/E4 REFUTED), and (5) a login-node CPU time limit (300 CPU-seconds) prevented live re-verification of the most compute-intensive claims. The infrastructure has accumulated bit-rot without accompanying test coverage for integration paths.

---

## §2. Environment

```
Host:        arf-ui1, Rocky Linux 9.2 (kernel 5.14.0-284.30.1.el9_2)
CPU:         112 logical cores (Intel Xeon)
RAM:         ~503 GiB total
GPU:         NONE (no nvidia-smi; CPU-only)
ulimit -t:   300 CPU-seconds per process  ← critical constraint
Audit venv:  .venv-audit, Python 3.12.2
Key deps:    torch=2.12.0, transformers=5.8.1, zeyrek=0.1.3 (audit);
             torch=2.11.0, transformers=5.5.0 (repo's .venv — minor drift)
Pinned SHA:  834ff8e5ca62ea5179e7e82a37e3edf0bb63bb8a
Random seed: 42 (params.yaml)
```

**CPU time limit impact:** The login node enforces `ulimit -t 300`. Any Python process that loads PyTorch + transformers and then runs BERTurk forward passes exhausts this limit within seconds (exit 152 / SIGXCPU). This prevented live re-verification of: A1 (5-seed eval), B1 (training time), C2 (BERTurk throughput), C3 (reranker throughput). Results for these claims rely on stored training artifacts.

---

## §3. Detailed Claim Ledger

### A-Group: Accuracy

#### A1 — Overall Disambiguation EM: 98.3%

**Method:** Read `models/v6/ensemble_results.json` + `models/v6/disambiguator/test_results.json`. Attempted live evaluation via `audit/scripts/eval_disambiguator.py` — terminated by SIGXCPU during BERTurk embedding caching.

| Source | Metric | Value |
|--------|--------|-------|
| Stored seed-42 test | overall_em (argmax) | 0.9821 (98.21%) |
| Stored ensemble (5 seeds) | overall_em (argmax) | **0.9832 (98.32%)** |
| README claim | — | 98.3% |
| Gap (ensemble vs claim) | — | +0.02pp |

Seed statistics: min=97.98%, max=98.28%, mean=98.15%, median=98.17%, stdev=0.114pp.

**Verdict: PARTIALLY CONFIRMED**  
Stored ensemble EM 98.32% matches claimed 98.3% within 0.02pp. Two mandatory caveats apply: (TLF-1) EM is candidate-index argmax, not full-parse string equality; (TLF-2) headline is 5-seed ensemble, undisclosed in README.

Evidence: `models/v6/ensemble_results.json`, `models/v6/disambiguator/test_results.json`, `src/train/train_disambiguator.py:253–272`.

#### A2 — Dual-Head Generation EM: 84.7%

**Verdict: UNVERIFIABLE** — No `DualHeadAtomizer` checkpoint exists in the repository (`models/v6/` contains only disambiguator seeds; no `dual_head/` directory). Retraining requires a TRUBA GPU job outside audit scope.

#### A3, A4, A5 — TTC-3600 Classification (0.947, deltas, improves every classifier)

**Verdict: UNVERIFIABLE** — TTC-3600 dataset (`data/external/ttc3600/`) absent from checkout. Network access blocked on login node (wget exit 8; HuggingFace Hub connection failed). Code exists at `src/benchmark/run_all_benchmarks.py` and `src/classify/`.

---

### B-Group: Training Efficiency

#### B1 — 14 Minutes CPU Training

**Method:** Checked `STATUS.md:489` for training log evidence. Live re-run blocked by CPU time limit.

**Stored evidence:** `STATUS.md:489` — "Training COMPLETED: CPU job 5506392 on orfoz243, 14 min 40 sec." Hardware is disclosed in STATUS.md (Orfoz 112-core Xeon) — same CPU family as audit host.

**Documentation gap:** README states "14 minutes on a standard CPU" without disclosing the CPU model. The 14:40 figure is for a specific TRUBA Orfoz node.

**Verdict B1 (live):** UNVERIFIABLE — CPU time limit prevents re-training.  
**Verdict B1 (code):** CONFIRMED — all CUDA references gated by `torch.cuda.is_available()` or `--device` defaulting to CPU. Training runs on CPU-only hardware by design.

Evidence: `STATUS.md:489`, `audit/results/00_5_cuda_audit.md`.

#### B2 — Disambiguator Reranker ≈ 1M Parameters

**Method:** Read `models/v6/disambiguator/test_results.json:trainable_params`.

**Measured:** trainable_params = **1,043,905** ≈ 1M.  
**Claimed:** "~1M param reranker."

**Verdict: CONFIRMED** — 1,043,905 ≈ 1M (within 4.4%).

Evidence: `models/v6/disambiguator/test_results.json:16`.

#### B3 — DualHead ≈ 5.2M Parameters

**Method:** Instantiated `DualHeadAtomizer` with vocabulary files from `models/vocabs/`.

**Key finding:** Parameter count is critically sensitive to `root_vocab_size`. With `root_vocab_15K.json` (15,002 entries) + `char_vocab.json` (106) + `tag_vocab.json` (7,807) and default dims: **5,200,538 params = 5.20M**.

**Verdict: CONFIRMED** — with `root_vocab_15K.json`, exact match to 5.2M claim.  
**Caveat:** No DualHead checkpoint exists to confirm which root vocab was used in training. README does not document which root_vocab_path corresponds to the 5.2M number.

Evidence: `models/vocabs/root_vocab_15K.json` (15,002 entries), `src/kokturk/models/dual_head.py`.

---

### C-Group: Inference Performance

#### C1 — Zeyrek: ~6,380 tok/s, ~50 MB

**Method:** Benchmarked `MorphoAnalyzer(backends=['zeyrek'])` on 957 unique validation sentences (cold cache, each sentence analyzed exactly once). Memory measured via `psutil`.

| Metric | Measured | Claimed | Ratio | Verdict |
|--------|----------|---------|-------|---------|
| Throughput (cold, unique) | 1,959 tok/s | ~6,380 tok/s | 0.31× | REFUTED |
| Memory delta (load) | 237 MB | ~50 MB | 4.7× | REFUTED |

**Notes:** Warm-cache throughput (repeated identical sentences) inflated to 2.2M tok/s — Zeyrek has internal caching. Cold-cache is the fair comparison. Memory includes full MorphoAnalyzer + Python overhead; actual Zeyrek data structures may be smaller. Login-node vs compute-node difference may partially explain throughput gap.

**Verdict: REFUTED** — both throughput and memory outside ±50% bounds.

Evidence: `audit/results/10_inference_benchmarks.json`.

#### C2 — BERTurk (cached): ~3,200 sent/s, ~1.5 GB

**Verdict: UNVERIFIABLE** — SIGXCPU kills process during BERTurk model loading.  
**Partial note:** BERTurk `.safetensors` on disk = 422 MB; 110M params × 4 bytes = 440 MB weights. The 1.5 GB claim would require large activation tensors or optimizer state — plausible for training, implausible for inference-only.

#### C3 — Reranker: ~50,000 tok/s, ~10 MB

**Measured (partial):** Reranker checkpoint memory delta = **9.3 MB** (consistent with claimed ~10 MB).  
Throughput: UNVERIFIABLE — PyTorch startup exhausts CPU time before measurement window.

**Verdict: PARTIALLY CONFIRMED** — memory (9.3 MB vs 10 MB: within ±20%). Throughput unverified.

#### C4 — DualHead: ~2,000 tok/s, ~20 MB

**Verdict: UNVERIFIABLE** — No DualHead checkpoint (same as A2).

---

### D-Group: Data Resource

#### D1 — TR-Gold-Morph: 2,512,034 Entries

**Verdict: PARTIALLY CONFIRMED**  
The 2.5M SQLite DB was generated on TRUBA (documented in `STATUS.md`) but is absent from the checkout. The local `data/gold/tr_gold_morph_v1.jsonl` (80,582 rows) is a **labeled training corpus slice**, not the full resource. Two distinct artifacts with the same name family. The 2.5M number is documented as real in STATUS.md but cannot be verified from the checkout.

Evidence: `audit/results/00_5_resource_provenance.md`.

#### D2 — Tiers: gold / silver / bronze

**Measured (from JSONL):** gold=2,496, silver-auto=61,516, silver-agreed=16,570. **No `bronze` tier exists.**

**Verdict: PARTIALLY CONFIRMED** — tiered quality system exists; tier names differ from README documentation (3 tiers vs 3 names, but wrong names).

Evidence: `data/gold/tr_gold_morph_v1_stats.json`.

#### D3 — 95.6% Morpheme Boundary Coverage

**Verdict: UNVERIFIABLE** — `data/gold/tr_gold_morph_v1.jsonl` has fields `{sentence_id, token_idx, surface, label, tier}`. No `morpheme_boundaries` field. Boundaries presumably reside in the absent SQLite DB.

#### D4 — Three Export Schemas

**Verdict: PARTIALLY CONFIRMED** — `src/resource/schema.py:197` (`export_multi_schema`) generates canonical/UD/UniMorph TSV + stats JSON. On-disk files use different naming (`tr_gold_morph_v1.*`). No CoNLL-U exporter found in source; `tr_gold_morph_v1.conllu` (99,685 rows) appears produced out-of-band.

#### D5 — MIT License for Data

**Verdict: PARTIALLY CONFIRMED** — `LICENSE` file is MIT; `CITATION.cff` is MIT. However `pyproject.toml` declares `Apache-2.0` — internal contradiction (see Side Finding α). No data-specific license file exists under `data/`.

---

### E-Group: Statistical Rigor

#### E1 — 3,600 Docs, 6 Categories, 5-Fold Stratified CV

**Verdict: UNVERIFIABLE (live run)** — TTC-3600 data unavailable.  
**Code level CONFIRMED:** `src/benchmark/run_all_benchmarks.py:154` uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

#### E2/E3 — Paired Bootstrap (10K iterations) + Holm-Bonferroni Exist

**Verdict: CONFIRMED** — `src/benchmark/significance.py` implements both. All tests pass (937 passed in pytest). Functions work as documented.

Evidence: `tests/benchmark/test_significance.py`.

#### E4 — "All differences tested via paired bootstrap + Holm-Bonferroni" (README claim)

**E4(a) — Did the headline pipeline apply this?**  
**Verdict: REFUTED (predetermined)**  
`src/benchmark/run_all_benchmarks.py` imports no module from `benchmark.significance`. The README's claim is false at code level. Confirming: `grep -rn "significance\|paired_bootstrap\|holm_bonferroni" src/benchmark/run_all_benchmarks.py` — zero results.

**E4(b) — Do deltas survive bootstrap if we run it now?**  
**Verdict: UNVERIFIABLE** — No TTC-3600 fold predictions available to bootstrap over.

Evidence: `src/benchmark/run_all_benchmarks.py` (imports section), `src/benchmark/significance.py`.

---

### F-Group: Functional API

#### F1 — `MorphoAnalyzer(backends=["disambiguator"]).analyze_sentence(...)`

**Verdict: REFUTED**  
`src/kokturk/core/analyzer.py:360` — `DisambiguatorBackend.__init__()` calls `Path(vocab_dir)` without importing `pathlib`. Instantiation raises `NameError: name 'Path' is not defined`. The disambiguator API path is broken in the current codebase.

**Bug:** Missing `from pathlib import Path` in `DisambiguatorBackend.__init__` (or its enclosing scope). Impact: HIGH — the primary novel API advertised in README cannot be called.

Evidence: `src/kokturk/core/analyzer.py:360`.

#### F2/F3 — Canonical String Examples

**Measured vs README:**

| Word | Measured | README Expected | Match |
|------|----------|-----------------|-------|
| `evlerinden` | `ev +Noun +POSS.3PL +ABL` | `ev +PLU +POSS.3SG +ABL` | ❌ |
| `gidiyordum` | `gitmek +Verb +PROG +PAST` | `git +PROG +PAST +1SG` | ❌ |
| `kitapçılardan` | `Kitap +Noun +AGT +Noun +PLU +ABL` | `kitap +AGT +PLU +ABL` | ❌ |

**Root cause:** `ZEYREK_TO_CANONICAL` mapping was updated to include POS tags (`Noun → +Noun`, `Verb → +Verb`) and changed agreement marker handling (`A1sg → ''` instead of `→ +1SG`). Training data labels DO use the current format (confirmed: `kuş +Noun +ABL`, `almak +Verb +PAST`). The code and training data are internally consistent. The README examples are **outdated** — they reflect an older canonical format.

**Verdict: REFUTED** — exact string mismatch for all 3 documented examples.  
**Important nuance:** Training data → model → evaluation chain is internally consistent. Only the documentation is wrong, not the evaluation metric.

Evidence: `src/kokturk/core/constants.py:ZEYREK_TO_CANONICAL`, `data/splits/val.jsonl` labels.

#### F4 — sklearn Integration

**Test:** `MorphoTransformer() + TfidfVectorizer + LogisticRegression` on 941 samples (2 classes from val.jsonl).

| Check | Result | Required |
|-------|--------|----------|
| Unique predicted classes | 2 | > 1 ✓ |
| Mean max_proba | 0.836 | < 0.95 ✓ |
| TF-IDF vocab size | 1,722 | ≥ 100 ✓ |
| Test accuracy | 93.6% | — |

**Verdict: CONFIRMED** — sklearn pipeline fits and predicts correctly; atomization produces a non-trivial feature vocabulary.

#### F5 — CLI

`python -m kokturk.cli.main analyze "evlerinden"` → `evlerinden → ev +Noun +POSS.3PL +ABL` (exit 0).

**Verdict: CONFIRMED** — CLI works; output matches code behavior.

#### F6 — ariturk

| Check | Result | Expected | Match |
|-------|--------|----------|-------|
| `TextCleaner().clean("  TÜRKÇE   metİn  ")` | `türkçe metin` | `türkçe metin` | ✓ |
| `turkish_lower("I")` | `ı` | `ı` | ✓ |
| `turkish_upper("i")` | `İ` | `İ` | ✓ |

**Verdict: CONFIRMED** — Turkish locale-correct casing works.

#### F7 — `pip install kokturk` (PyPI)

**Verdict: REFUTED** — Package is not published on PyPI. `pip install kokturk` would fail for any user following the README's quickstart. This is a documentation-vs-reality gap (HIGH impact — it's the first command in the README).

---

### G-Group: Pipeline & Infrastructure

#### G1 — `dvc repro` End-to-End Reproducibility

**Unmodified result:** `ERROR: failed to reproduce 'ingest': No such file or directory: 'data/external/boun_treebank'`

**Two compounding blockers:**
1. BOUN Treebank not cloned (`data/external/` empty; `setup.sh` would clone it, but wasn't run)
2. `dvc.yaml:17` references `src/morpho_tr/core/analyzer.py` (stale path post-rename to `kokturk`). Patch `audit/branches/dvc-fix.patch` applies cleanly and would fix blocker #2, but blocker #1 remains.

**Verdict: REFUTED** — pipeline fails at the first stage. Both blockers are infrastructure/maintenance issues (bit-rot), not substantive research-claim refutations. The underlying research code (training, evaluation) is functional; only the pipeline orchestration layer is broken.

Evidence: `audit/results/05_dvc_repro.json`, `dvc.yaml:17`.

#### G4 — Tests

**Pytest run (not gpu, not slow):** 937 passed, 1 failed, 4 skipped (81.4 s).

**Single failure:** `tests/test_labeling_functions.py::TestLfGazetteer::test_known_propn_votes_noun` — assertion `assert -1 == 0` (gazetteer labeling function returns ABSTAIN where NOUN is expected).

**Verdict: SUBSTANTIALLY CONFIRMED** — 99.9% pass rate. One labeling-function test failure is a minor regression in the silver-label pipeline, not in the evaluation or API.

Evidence: `audit/results/12_pytest.xml`.

---

### H-Group: Architecture

#### H1 — BERTurk Frozen During Training

**Code:** `src/kokturk/models/disambiguator.py:117–120` — `for param in self.bert.parameters(): param.requires_grad = False`.  
**Verdict: CONFIRMED** — all BERTurk parameters have `requires_grad=False` at construction time.

#### H2 — Scalar Score + Argmax

**Code:** `src/train/train_disambiguator.py:253` — `preds = logits.argmax(dim=-1)` over candidate logits.  
**Verdict: CONFIRMED** — scalar scoring with argmax selection confirmed.

#### H3 — Zeyrek Candidate Source

**Code:** `src/train/disambiguation_dataset.py:52` — explicitly imports `ZeyrekBackend`; docstring says "pre-computes Zeyrek candidates at init time."  
**Verdict: CONFIRMED** — Zeyrek is the candidate source.

#### H4 — Character-by-Character Generation (OOV)

**Code review:** `src/kokturk/models/dual_head.py:459–481` — decoder loop emits **tag tokens** one at a time, not characters. Input encoder is char-level (`CharGRUEncoder`), but the output sequence is tag tokens.

**README wording** ("character-by-character") is ambiguous; "per-step generation over a char-level-encoded input" would be accurate.

**Verdict: PARTIALLY CONFIRMED** — per-step generation confirmed; "character-by-character" at the output is not accurate (output is tag tokens, not characters).

#### H5 — Phonology Module

**Code review:** `src/kokturk/core/phonology.py` exists with `last_vowel`, `is_front`, `VOICING_MAP`, `HarmonyResult`. However, in the main analysis pipeline, phonology functions are called only inside `_analyze_code_switch` at `analyzer.py:650–663`; not in the primary `to_canonical` flow.

**Verdict: PARTIALLY CONFIRMED** — module exists and is exercised in the code-switch branch. The main canonicalization pipeline relies on `ZEYREK_TO_CANONICAL` lookup, not phonology computation.

---

## §4. Side Findings

| ID | Severity | Finding | Evidence |
|----|----------|---------|----------|
| α | HIGH | **License contradiction:** `LICENSE` = MIT, `CITATION.cff` = MIT, but `pyproject.toml:license = {text = "Apache-2.0"}`. These govern different scopes but should be reconciled. | `pyproject.toml:5`, `LICENSE:1` |
| β | MED | **dvc.yaml stale path:** `dvc.yaml:17` references `src/morpho_tr/core/analyzer.py` post-rename. One-line fix in `audit/branches/dvc-fix.patch`. | `dvc.yaml:17` |
| γ | MED | **2.5M resource absent from checkout:** The 2.5M SQLite DB is on TRUBA; on-disk JSONL (80K rows) is a different artifact. New users will not discover the full resource from the repo alone. | `audit/results/00_5_resource_provenance.md` |
| δ | LOW | **Bronze tier doesn't exist:** README documents `gold/silver/bronze` tiers; implementation has `gold/silver-auto/silver-agreed`. | `data/gold/tr_gold_morph_v1_stats.json` |
| ε | LOW | **No CoNLL-U exporter in source:** On-disk `tr_gold_morph_v1.conllu` (99,685 rows) exists but no code generates it. Out-of-band provenance. | `src/resource/schema.py` |
| ζ | MED | **`make train-local` trains wrong model:** Makefile target runs `train_atomizer.py` (legacy GRU), not the BERTurk-based disambiguator that produces the 98.3% headline. | `Makefile:52`, `src/train/train_atomizer.py` |
| η | LOW | **Evaluation not a version-controlled module:** Held-out evaluation logic lives as a Python heredoc in `scripts/truba/submit_v6_eval.sh:25–93`. Extracte into `audit/scripts/eval_disambiguator_verbatim.py` for the audit. | `scripts/truba/submit_v6_eval.sh:25` |
| θ | INFO | **Checkpoint provenance gap:** Only seed-42 has `test_results.json`. Seeds 123/456/789/1337 have only `config.json`. No training metadata (commit hash, data hash) stored alongside `.pt` files. | `models/v6/disambiguator_s*/` |
| ι | HIGH | **Significance not wired (TLF-3):** See E4 above. | `src/benchmark/run_all_benchmarks.py` (imports) |
| λ | HIGH | **EM definition mismatch (TLF-1):** See A1 and §1. | `src/train/train_disambiguator.py:253` |
| λ² | LOW | **Version mismatch:** `pyproject.toml` = 0.5.0; `src/kokturk/__init__.py` = 0.1.0; `src/ariturk/__init__.py` = 0.1.0. CHANGELOG most recent tag = 0.5.0. The `__init__.py` files were not updated. | `pyproject.toml:5`, `__init__.py:1` |
| λ³ | MED | **F1 runtime bug:** `src/kokturk/core/analyzer.py:360` — `DisambiguatorBackend.__init__` uses `Path()` without importing `pathlib`. The primary disambiguator API is broken. | `analyzer.py:360` |
| NEW-1 | HIGH | **F2/F3 README examples outdated:** All three canonical string examples in README use the old format (no POS tags, different agreement markers). Current code/data are internally consistent but the README misleads users. | `src/kokturk/__init__.py:70–71`, `src/kokturk/core/constants.py` |
| NEW-2 | HIGH | **F7 PyPI not published:** `pip install kokturk` fails. Package not on PyPI. README quickstart is broken for external users. | First command in README quickstart |
| NEW-3 | INFO | **Login node CPU limit blocks live verification:** `ulimit -t 300` (CPU seconds) prevents running BERTurk forward passes interactively. Live re-verification of A1, B1, C2, C3 requires a SLURM job. | `ulimit -a` output |

---

## §5. Reproduction Cookbook

Complete shell commands for a third-party auditor on a Rocky Linux 9 host with Python 3.12 and ≥16 GB RAM.

```bash
# 0. Clone and pin commit
git clone https://github.com/melikkul/KokTurk.git NLP_Project
cd NLP_Project
git checkout 834ff8e5ca62ea5179e7e82a37e3edf0bb63bb8a

# 1. Create audit venv
python3.12 -m venv .venv-audit
.venv-audit/bin/pip install --upgrade pip
.venv-audit/bin/pip install -e ".[dev]"

# 2. Verify key imports
PYTHONPATH=src .venv-audit/bin/python -c "
import kokturk, ariturk
from kokturk import Atomizer
from ariturk import TextCleaner, turkish_lower
print('OK')
"

# 3. Run tests (non-GPU, non-slow)
PYTHONPATH=src .venv-audit/bin/python -m pytest tests/ -m "not gpu and not slow" -v

# 4. Spot-check canonical API (NOTE: output differs from README — this is expected)
PYTHONPATH=src .venv-audit/bin/python -c "
from kokturk import Atomizer
print(Atomizer().to_canonical('evlerinden'))
# Current output: 'ev +Noun +POSS.3PL +ABL'  (README shows old format: 'ev +PLU +POSS.3SG +ABL')
"

# 5. Verify stored EM results
python3 -c "
import json
r = json.load(open('models/v6/ensemble_results.json'))
print('Ensemble EM:', r['ensemble']['overall_em'])  # 0.9832
print('Seed range:', min(v['overall_em'] for v in r['individual'].values()),
      'to', max(v['overall_em'] for v in r['individual'].values()))
"

# 6. Live evaluation (requires SLURM on compute node, NOT login node):
#    sbatch -p orfoz -n 56 --wrap="
#    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python \
#      audit/scripts/eval_disambiguator.py \
#      --test data/splits/test.jsonl --val data/splits/val.jsonl \
#      --output audit/results/07_em_live.json"

# 7. Full pipeline (requires BOUN Treebank — blocked by network on login node):
#    bash setup.sh  # clones boun_treebank into data/external/
#    # Apply bit-rot fix first:
#    git apply audit/branches/dvc-fix.patch
#    .venv-audit/bin/dvc repro

# 8. Parameter counts
PYTHONPATH=src .venv-audit/bin/python -c "
import json, sys; sys.path.insert(0,'src')
from kokturk.models.dual_head import DualHeadAtomizer
import json
root_size = len(json.load(open('models/vocabs/root_vocab_15K.json')))
m = DualHeadAtomizer(char_vocab_size=106, root_vocab_size=root_size, tag_vocab_size=7807)
print('DualHead params:', sum(p.numel() for p in m.parameters()))  # 5,200,538
"
```

---

## §6. Open Questions / Unverifiable Items

| Claim | Blocker | What Would Unblock |
|-------|---------|-------------------|
| A1 (live EM re-run) | Login node ulimit -t 300 | SLURM compute job: `sbatch -p orfoz --wrap="python audit/scripts/eval_disambiguator.py ..."` |
| A1 (TLF-1 string-EM gap) | CPU time limit + live eval blocked | Same SLURM job + add `--report-both-em-definitions` flag to eval wrapper |
| A2 (84.7% generation EM) | No DualHead checkpoint | Retrain on TRUBA GPU (H100, ~1–3h); user approval required |
| A3/A4/A5 (TTC-3600) | No data + network blocked | Manual download from Kaggle or GitHub on internet-connected host |
| B1 (14 min training) | CPU time limit | SLURM Orfoz job: `sbatch -p orfoz --wrap="time python src/train/train_disambiguator.py --config configs/train/v6_full.yaml --seed 42"` |
| C2 (BERTurk throughput) | CPU time limit | SLURM job with proper wall-time |
| C3 (reranker throughput) | CPU time limit | Same SLURM job |
| C4 (DualHead throughput) | No checkpoint + CPU limit | Retrain + SLURM |
| D3 (boundary coverage) | SQLite DB absent | Access TRUBA path or re-run `make corpus-stats` with access to original DB |
| E4(b) (deltas survive bootstrap) | No TTC-3600 data | Same as A3/A4/A5 |
| G2 (BOUN Treebank presence) | Network blocked | `bash setup.sh` from internet-connected host |
| G1 (end-to-end pipeline) | BOUN Treebank + dvc.yaml bit-rot | Fix dvc.yaml (apply `audit/branches/dvc-fix.patch`) + `bash setup.sh` + `dvc repro` |

---

*Report generated: 2026-05-15. Audit artifacts in `audit/results/`. Verbatim evaluation script at `audit/scripts/eval_disambiguator_verbatim.py`.*
