# Pre-Flight — Final Push (CPU-Only Track)

**Date:** 2026-05-16  
**Executed by:** automated workstream (Claude Sonnet 4.6)

---

## Check 1: Git Status

```
Branch: feat/aksu-readme-rewrite
Untracked/modified: tests/data/test_preprocess_offline.py (ruff auto-fix, staged)
Recent commits: 1d946a1 → ad5a1b1 → 2bfe481 → a75f841 → 7496f28
```

**Status: OK**

---

## Check 2: metrics.json Null Values

| Key | Value |
|-----|-------|
| em_argmax_ensemble | 0.983 ✅ |
| em_argmax_std | 0.0011 ✅ |
| em_string_ensemble | **null** ⏳ |
| em_string_single_seed_min | **null** ⏳ |
| em_string_single_seed_max | **null** ⏳ |
| training_wall_clock_min | 16.71 ✅ |
| zeyrek_tok_per_sec | 1537.3 ✅ |
| berturk_sent_per_sec | **null** ⏳ |
| reranker_tok_per_sec | **null** ⏳ |
| dualhead_tok_per_sec | **null** ⏳ |
| dualhead_em | **null** ⏳ |
| dataset_v1_entries | 80537 ✅ |
| dataset_v2_entries | **null** ⏳ |
| dataset_boundary_coverage | **null** ⏳ |
| classification_macro_f1_atomized_berturk | **null** (deferred v1.1) |

**7 null values to fill by this continuation.**

---

## Check 3: SLURM Jobs to Cancel

```
5782172  akya-cuda  aksu-v6-eval         PENDING
5782174  akya-cuda  aksu-dualhead-train  PENDING
```

**Action: Both canceled via `scancel 5782172 5782174`. ✅**

---

## Check 4: OSCAR Pilot Data

```
/arf/scratch/scolakoglu/oscar-tr-pilot.jsonl: MISSING
```

**⚠️ HALT on preprocessing.** The OSCAR pilot file has not been staged.  
User must run Phase A.1 before the preprocessing SLURM job can be submitted:

```bash
cd /arf/home/scolakoglu/NLP_Project
.venv/bin/python scripts/data/download_oscar_pilot.py \
    --max-sentences 500000 \
    --out /arf/scratch/scolakoglu/oscar-tr-pilot.jsonl
```

Expected: ~5–10 min I/O-bound on the login node. Then re-trigger preprocessing:

```bash
sbatch scripts/truba/submit_preprocess_aksu.sh \
    --local-jsonl /arf/scratch/scolakoglu/oscar-tr-pilot.jsonl
```

**Unblocked work:** v6 eval (CPU), DualHead v1 training (CPU), inference benchmark, TTC-3600 attempts — all submitted below without OSCAR pilot.

---

## Check 5: Credentials

| Secret | Status |
|--------|--------|
| HF_TOKEN | ❌ MISSING |
| ZENODO_TOKEN | ❌ MISSING |
| AWS_ACCESS_KEY_ID | ❌ MISSING |
| AWS_SECRET_ACCESS_KEY | ❌ MISSING |

**Phase G (publication) is fully blocked.** No publication steps can run without credentials. See master prompt Phase A.2–A.5 for setup instructions.

---

## Check 6: Checkpoints and Entry Points

| Item | Status |
|------|--------|
| models/v6/disambiguator/best_model.pt | ✅ |
| models/v6/disambiguator_s123/best_model.pt | ✅ |
| models/v6/disambiguator_s456/best_model.pt | ✅ |
| models/v6/disambiguator_s789/best_model.pt | ✅ |
| models/v6/disambiguator_s1337/best_model.pt | ✅ |
| models/berturk/ (safetensors) | ✅ |
| aksu.train.train_v4_master --device flag | ✅ |
| scripts/eval_disambiguator.py | ✅ |

---

## Decision

Proceed with: Phase B (script patches), Phase C Stage 1 (v6 eval CPU, DualHead v1 CPU, inference benchmark CPU), Phase D (TTC-3600 attempts). Skip Phase C preprocessing/autolabel stages until OSCAR pilot is staged.
