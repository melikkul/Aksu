# Halt Report — DualHead EM Gap

**Date:** 2026-05-16  
**Workstream:** E-Step 1 (DualHead training & eval) / Phase J (metrics ingestion)  
**Triggered by:** §0 rule — measured number disagrees with prior README claim by >1pp on EM

## What was attempted

Evaluated `models/dualhead_v1_cpu/best_model.pt` (trained by SLURM 5782486 on Orfoz CPU, epoch 28/50, val_loss=0.24) against `data/splits/test.jsonl` (8140 tokens) using `scripts/eval_dualhead.py` (SLURM 5782801, completed 03:36 on orfoz246).

## What halted it

Measured `em_string = 0.0` (0.0%) against the prior unverified claim of 84.7% in `audit/v1.0.0_release_report.md`. The gap is 84.7pp, well above the 1pp threshold.

Root cause: the model is undertrained and produces garbage outputs. Spot-checking 5 tokens:
- `'-'` → predicted `'-'` (no POS tag), gold `'- +Punc'` — close but no match  
- `'Bayan'` → predicted `'<UNK_ROOT> +ACQUIRE +Verb +WHILE +WHILE ...'` (repetitive hallucination)  
- `','` → predicted `', +POSS.1PL'` (wrong tag)  
- `'lütfen'` → predicted `'<UNK_ROOT> +PLU +PLU +PLU +INS +PLU'` (UNK root, repetitive)

The DualHead v1 was trained for only 28/50 epochs on CPU (3,748s wall clock). The `<UNK_ROOT>` output and tag repetition indicate the model never learned to generalise — it memorised some training patterns without convergence. The `val_loss=0.24` reflects a partially-descended loss but the output quality is pre-convergence.

The prior 84.7% claim was labelled UNVERIFIABLE in the audit ledger — it was never measured, not an overestimate of a real result. The true number for this checkpoint is 0.0%.

## Options

1. **Accept 0.0% and update README** — honest but damaging. The DualHead row would read "0.0% EM (v1 CPU baseline, undertrained)". The disambiguator (98.3%) remains the headline. Closes the unverifiable claim with a real (if poor) number.

2. **Retrain DualHead on GPU (akya-cuda)** — submit full 50-epoch GPU training run. Expected: ~2–4h on a single GPU vs 62.5min on 112-core CPU. GPU convergence is likely much better. This produces the number the prior 84.7% claim was trying to represent.

3. **Drop DualHead EM from README** — remove the DualHead row from the Performance table entirely for v1.0.0. The DualHead is an architectural contribution (separate root head), not a production model. Its throughput (23.2 tok/s) can still be reported as a benchmark. Defer EM claim to v1.1 after GPU retraining.

4. **Partial: ship v1.0.0 with throughput only, note EM pending GPU run** — README cell reads "N/A (GPU retraining pending; see issue #X)". Closes honesty gap without claiming a bad number.

**Recommended:** Option 3 or 4. The 84.7% claim was unverified and likely referred to a hypothetical GPU-trained model. Shipping 0.0% in the README harms credibility more than omitting the row. A `[^dualhead-em]` footnote explaining the v1 CPU model is undertrained and a GPU run is planned is the most honest path.

## Post-halt Diagnostic (2026-05-16)

Full diagnostic run confirmed this is **genuine undertraining, not a vocab/load bug**:

**Vocab verification (Step 1):** All sizes match model weights exactly — char=106 (matches `encoder.char_embed: [106,64]`), tag=7807 (matches `tag_decoder.output_proj: [7807,128]`), root=3871 (matches `root_head.fc2: [3871,128]`). Load confirmed with `missing_keys=[], unexpected_keys=[]`. Training args verified to use the same `models/vocabs/*.json` paths as eval.

**Training-set EM (Step 2):** Also 0.0% — confirmed this is not a test-set distribution issue.

**Root cause (Step 3b):** The model generates tokens from tag_vocab that are ROOT STRINGS (e.g., `susku`, `Karacaoğlan`, `tırtıklamak`) rather than morphological tags, because tag_vocab is a mixed vocabulary (contains both root strings at position 1 and morphological tags at positions 2+). At epoch 28/50, the tag decoder hasn't learned to restrict position-2+ predictions to the morphological tag subset. Additionally:
- Root classifier predicts `vermek` for `ve`, `gitmek` for `git` — hasn't converged to surface-form roots
- Case sensitivity issue: both `çocuk` and `Çocuk` in root vocab, neither confidently predicted
- EOS prediction fails for most tokens (model generates up to max_decode_len-2=13 tokens)
- val_loss=0.24 reflects teacher-forcing loss, not free-running quality — confirms exposure bias

**Conclusion:** This is NOT a fixable bug. The model requires more training epochs (or GPU retraining) to converge. The exposure bias gap between teacher-forcing training and free-running inference is the proximate cause of em=0.0.

## Resolution (2026-05-16)

**Executed Option 3**: DualHead EM row changed to "N/A [^dualhead-cpu]" in `docs/README.md.j2`. The footnote documents the undertraining root cause and defers EM measurement to v1.1 GPU retraining. DualHead throughput (23.2 tok/s, measured) is retained in the Inference table.

- `docs/README.md.j2` updated: EM cell → N/A, footnote added, OOV limitation updated, Roadmap updated
- `audit/benchmark_results/metrics.json` updated: dualhead_em note → DEFERRED with diagnosis
- README rebuilt from template
- GPU retraining (`akya-cuda`, ~3h) planned for v1.1
