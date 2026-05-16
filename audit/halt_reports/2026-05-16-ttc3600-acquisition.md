# Halt Report — TTC-3600 Acquisition (Phase D)

**Date:** 2026-05-16  
**Workstream:** E-Step 2 (TTC-3600 acquisition)  
**Executed by:** automated workstream (Claude Sonnet 4.6)

## What Was Attempted

Three sequential acquisition attempts for the TTC-3600 Turkish text classification dataset (Akin & Akin, 2007 — 3,600 documents across 6 categories: ekonomi, kültür-sanat, sağlık, siyaset, spor, teknoloji):

**Attempt 1 — GitHub mirror (savasy/TurkishTextClassification):**  
`https://raw.githubusercontent.com/savasy/TurkishTextClassification/master/TTC3600.csv` → HTTP 404.  
The repository exists and is public, but `TTC3600.csv` is not in it. Available files: `7allV03.csv`, `TTC4900.csv`, `eval.csv`.

**Attempt 2 — HuggingFace Hub (`savasy/ttc3600`):**  
`https://huggingface.co/api/datasets/savasy/ttc3600` → HTTP 401.  
The dataset exists but is private/gated and requires authentication or approval. `savasy/ttc4900` returns 200 (public).

**Attempt 3 — TTC-4900 as alternative:**  
`TTC4900.csv` is publicly downloadable (GitHub, 4901 rows). However, TTC-4900 is not a drop-in replacement:
- 4,900 documents (7 categories × 700 each) vs. 3,600 (6 categories × 600)
- 7th category "dunya" not present in TTC-3600
- Category labels use unaccented ASCII (saglik, kultur) vs. accented Turkish in the original
- Using TTC-4900 metrics to substantiate TTC-3600 README claims would be methodologically dishonest

## What Halted It

TTC-3600 is no longer publicly accessible via automated means. The original dataset from the Akin & Akin (2007) paper has been superseded by TTC-4900 in the author's public repository, and the HuggingFace copy is restricted.

## Decision: Re-defer to v1.1

Per plan §E-Step 2: ship v1.0.0 **without** the TTC-3600 classification table row. The README section "Text Classification — Why Atomization Matters" will be annotated with a pending footnote.

## User Options

1. **Email acquisition (manual):** Contact the original authors (Güneş Erkan & Olcay Taner Yıldız, or current dataset maintainer) to request the dataset. Subject: "TTC-3600 dataset access for morphological atomization research." This is the path to v1.1 results.

2. **Use TTC-4900 instead (alternative baseline):** Re-run classification benchmarks on TTC-4900; update all README claims to reference TTC-4900 (4,900 docs, 7 categories). This requires updating `src/aksu/classify/prepare_ttc3600.py` to handle the new format, rerunning all 6 feature × classifier configs, and verifying that claims like "atomization improves every classifier" still hold.

3. **Remove the classification table entirely from v1.0.0:** The core value proposition (morphological disambiguation) stands without it. The table can be added in v1.1 once either TTC-3600 or TTC-4900 benchmarks complete.

**Recommendation:** Option 3 (remove) for v1.0.0 honesty; Option 2 (TTC-4900) for v1.1 if email acquisition fails.

## Next Steps (No Blocking Action Required from Executor)

- The classification section in `docs/README.md.j2` will be annotated with `<!-- TTC-3600 DEFERRED — see issue #N -->` and a prose note.
- A GitHub tracking issue should be opened (requires GitHub credentials — deferred to Phase G).
- `audit/results/11_ttc3600.json` updated with this acquisition outcome.
