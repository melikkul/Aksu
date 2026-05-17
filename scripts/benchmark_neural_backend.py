"""NeuralBackend CPU vs GPU throughput benchmark.

Measures tokens/sec across four configurations and writes JSON to --out.
Baseline (fp32, cpu, no-compile) is measured first and stored as
`baseline_v1.0.0a0_cpu_tps` so v1.1 targets can be expressed relatively.

Usage:
    python scripts/benchmark_neural_backend.py --out audit/benchmark_results/v1.1_neural_backend.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------

N_WARMUP = 5
N_BENCHMARK_WORDS = 1000
BATCH_SIZE = 32
SEED = 42


@dataclass
class BenchmarkRow:
    config: str
    device: str
    precision: str
    compile_mode: str | None
    batch_size: int
    tokens_per_sec: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    peak_memory_mb: float | None
    error: str | None = None


@dataclass
class BenchmarkResult:
    baseline_v1_0_0a0_cpu_tps: float | None = None
    corpus_source: str = ""
    rows: list[BenchmarkRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------

def _load_corpus(n: int = N_BENCHMARK_WORDS) -> tuple[list[str], str]:
    """Return (word_list, source_description).

    Checks data/intermediate/ first; falls back to synthetic corpus
    drawn from the bundled lexicon with seed=42.
    """
    data_dir = Path("data/intermediate")
    if data_dir.exists():
        for txt_file in sorted(data_dir.glob("*.txt")):
            try:
                words = txt_file.read_text(encoding="utf-8").split()
                words = [w.strip(".,!?;:\"'()[]{}") for w in words if w.strip(".,!?;:\"'()[]{}")]
                if len(words) >= n:
                    rng = random.Random(SEED)
                    selected = rng.sample(words, n)
                    return selected, f"data/intermediate/{txt_file.name}"
            except Exception:
                continue

    # Fallback: synthetic corpus from bundled lexicon
    lex_path = Path("src/aksu/ariturk/data/turkish_wordlist.txt.gz")
    if lex_path.exists():
        with gzip.open(lex_path, "rt", encoding="utf-8") as fh:
            candidates = [
                w.strip()
                for w in fh
                if w.strip() and w.strip().isalpha() and len(w.strip()) >= 3
            ]
        rng = random.Random(SEED)
        selected = rng.choices(candidates, k=n)
        return selected, "synthetic_from_lexicon_seed42"

    # Last resort: built-in word list
    builtin = ["ev", "okul", "kitap", "araba", "insan", "su", "hava",
               "yol", "dağ", "şehir", "köy", "gelmek", "gitmek", "yapmak"]
    rng = random.Random(SEED)
    selected = rng.choices(builtin, k=n)
    return selected, "synthetic_builtin_fallback"


# ---------------------------------------------------------------------------
# Single-config benchmark
# ---------------------------------------------------------------------------

def _bench_config(
    words: list[str],
    model_path: str,
    vocab_dir: str,
    device: str,
    precision: str,
    compile_mode: str | None,
    batch_size: int,
    config_name: str,
) -> BenchmarkRow:
    """Run one benchmark configuration; return a populated BenchmarkRow."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        return BenchmarkRow(
            config=config_name,
            device=device,
            precision=precision,
            compile_mode=compile_mode,
            batch_size=batch_size,
            tokens_per_sec=None,
            p50_latency_ms=None,
            p95_latency_ms=None,
            peak_memory_mb=None,
            error="CUDA not available on this host",
        )

    try:
        from aksu.kokturk.core.analyzer import NeuralBackend

        backend = NeuralBackend(
            model_path=model_path,
            vocab_dir=vocab_dir,
            device=device,
            precision=precision,
            compile_mode=compile_mode,
            batch_size=batch_size,
        )
    except Exception as exc:
        return BenchmarkRow(
            config=config_name,
            device=device,
            precision=precision,
            compile_mode=compile_mode,
            batch_size=batch_size,
            tokens_per_sec=None,
            p50_latency_ms=None,
            p95_latency_ms=None,
            peak_memory_mb=None,
            error=str(exc),
        )

    # Warm-up
    for _ in range(N_WARMUP):
        backend.predict_batch(words[:batch_size])

    # Benchmark: time individual batch calls to get latency distribution
    latencies: list[float] = []
    tracemalloc.start()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        t0 = time.perf_counter()
        backend.predict_batch(batch)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    t_total = time.perf_counter() - t_start

    _, peak_bytes = tracemalloc.stop()
    tracemalloc.stop()

    if device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_memory_mb = peak_bytes / 1e6

    tokens_per_sec = len(words) / t_total if t_total > 0 else 0.0

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[int(n * 0.50)] if n > 0 else None
    p95 = latencies_sorted[int(n * 0.95)] if n > 0 else None

    return BenchmarkRow(
        config=config_name,
        device=device,
        precision=precision,
        compile_mode=compile_mode,
        batch_size=batch_size,
        tokens_per_sec=round(tokens_per_sec, 2),
        p50_latency_ms=round(p50, 3) if p50 is not None else None,
        p95_latency_ms=round(p95, 3) if p95 is not None else None,
        peak_memory_mb=round(peak_memory_mb, 2),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="NeuralBackend throughput benchmark")
    parser.add_argument("--model-path", default="models/atomizer_v2/best_model.pt")
    parser.add_argument("--vocab-dir", default="models/vocabs")
    parser.add_argument("--out", default="audit/benchmark_results/v1.1_neural_backend.json")
    parser.add_argument("--n-words", type=int, default=N_BENCHMARK_WORDS)
    args = parser.parse_args()

    words, corpus_source = _load_corpus(args.n_words)
    print(f"Corpus: {len(words)} words from '{corpus_source}'")

    result = BenchmarkResult(corpus_source=corpus_source)

    configs: list[dict[str, Any]] = [
        # Baseline FIRST — must be measured before any v1.1 configs
        dict(
            config_name="baseline_v1.0.0a0_cpu_fp32_no_compile",
            device="cpu",
            precision="fp32",
            compile_mode=None,
            batch_size=1,  # original v1.0 analyzed one word at a time
        ),
        dict(
            config_name="v1.1_cpu_fp32_compile",
            device="cpu",
            precision="fp32",
            compile_mode="reduce-overhead",
            batch_size=BATCH_SIZE,
        ),
        dict(
            config_name="v1.1_cuda_bf16_compile",
            device="cuda",
            precision="bf16",
            compile_mode="reduce-overhead",
            batch_size=BATCH_SIZE,
        ),
        dict(
            config_name="v1.1_cuda_bf16_compile_batch64",
            device="cuda",
            precision="bf16",
            compile_mode="reduce-overhead",
            batch_size=64,
        ),
    ]

    for cfg in configs:
        print(f"\nRunning: {cfg['config_name']} ...")
        row = _bench_config(words=words, model_path=args.model_path, vocab_dir=args.vocab_dir, **cfg)
        result.rows.append(row)

        if row.error:
            print(f"  SKIP: {row.error}")
        else:
            print(f"  {row.tokens_per_sec:.1f} tok/s  p50={row.p50_latency_ms:.1f}ms  p95={row.p95_latency_ms:.1f}ms  mem={row.peak_memory_mb:.1f}MB")

        # Record baseline
        if cfg["config_name"].startswith("baseline_") and row.tokens_per_sec is not None:
            result.baseline_v1_0_0a0_cpu_tps = row.tokens_per_sec

    # Acceptance check (J.1 alpha CPU criterion)
    baseline = result.baseline_v1_0_0a0_cpu_tps
    if baseline is not None:
        target = max(baseline * 1.3, 25.0)
        best_cpu_tps = max(
            (r.tokens_per_sec for r in result.rows if r.device == "cpu" and r.tokens_per_sec is not None),
            default=0.0,
        )
        print(f"\n=== Acceptance check ===")
        print(f"  Baseline: {baseline:.1f} tok/s")
        print(f"  Target:   ≥{target:.1f} tok/s (max(1.3×baseline, 25))")
        print(f"  Best CPU: {best_cpu_tps:.1f} tok/s  →  {'PASS ✓' if best_cpu_tps >= target else 'FAIL ✗'}")

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _row_to_dict(row: BenchmarkRow) -> dict[str, Any]:
        d = asdict(row)
        return d

    output: dict[str, Any] = {
        "baseline_v1.0.0a0_cpu_tps": result.baseline_v1_0_0a0_cpu_tps,
        "corpus_source": result.corpus_source,
        "configs": [_row_to_dict(r) for r in result.rows],
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
