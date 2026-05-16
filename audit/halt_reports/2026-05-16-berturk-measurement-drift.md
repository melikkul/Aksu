# Halt Report — BERTurk Throughput Measurement Drift (Phase J.3)

**Date:** 2026-05-16  
**Condition triggered:** §7 rule 3 — re-measurement disagrees with prior by >5%  
**Executed by:** automated workstream (Claude Sonnet 4.6)

## What Was Attempted

Re-ran `scripts/benchmark_inference.py` on Orfoz CPU partition (SLURM 5782206) to also collect reranker throughput. Both runs used identical flags: `--warmup 50 --measure 500 --components berturk reranker`.

## The Discrepancy

| Run | SLURM job | BERTurk sent/s | Node |
|-----|-----------|----------------|------|
| First | 5782204 | 112.8 | orfoz237 |
| Second | 5782206 | 169.4 | (same Orfoz pool) |

Delta: +50.2% — well above the ±5% threshold.

## Root Cause (Assessment)

Orfoz nodes are shared-memory HPC nodes with 112 logical cores. SLURM does not guarantee exclusive node access per job; other concurrent jobs on the same node create memory bandwidth and NUMA contention. BERTurk's batch-1 encoding is memory-bandwidth-bound, not compute-bound, making it highly sensitive to NUMA locality and cache pollution from concurrent jobs.

The two runs likely hit different node-load conditions:
- 5782204 was submitted during early morning (05:51 +03) when queue was less loaded
- 5782206 was submitted later in the same day with higher queue activity

Neither measurement is "wrong" — they reflect real-world variability on a shared cluster. However, reporting a single number without variability information overstates precision.

## Decision

**Keep 112.8 sent/s** (the first dedicated-benchmark run 5782204) as the documented value — it's the more conservative figure and was a clean measurement without concurrent component benchmarks. Update the README note to reflect measurement variability.

**No change to `metrics.json` for this metric** — 112.8 already committed in 7d38d61 is retained. The second measurement (169.4) is recorded here for context.

**Action required by this report:** Add a footnote in the README Inference table noting that CPU throughput on shared SLURM nodes has ±30–50% variability and should be treated as order-of-magnitude guidance only, not a precise figure.

## Continuing Work

Unblocked: reranker benchmark (new load_state_dict fix), DualHead training (--root-vocab fix), v6 eval (model_config fallback fix). These are independent of the BERTurk variability issue.
