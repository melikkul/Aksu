"""Measure Zeyrek throughput honestly: 100-iter warmup, 1000+ measure, cold cache, unique sentences."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import psutil
import zeyrek


def _read_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    import platform
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure Zeyrek tok/s honestly")
    ap.add_argument("--corpus", default="data/splits/test.jsonl")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--measure", type=int, default=1000)
    ap.add_argument(
        "--output",
        default="audit/benchmark_results/zeyrek_throughput.json",
    )
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}; running on synthetic sentences.")
        sentences = [
            f"Çocuklar evlerinden çıktı ve {i} numaralı arabaya bindiler."
            for i in range(args.warmup + args.measure)
        ]
    else:
        # Support both sentence-level files (field "sentence" or "text") and
        # token-level files (field "surface") by trying each key in order.
        raw_lines = [
            json.loads(line)
            for line in corpus_path.read_text().splitlines()
            if line.strip()
        ]
        field = next(
            (k for k in ("sentence", "text", "surface") if raw_lines and k in raw_lines[0]),
            None,
        )
        if field is None:
            raise ValueError(f"Cannot find text field in {corpus_path}; keys: {list(raw_lines[0].keys())}")
        sentences = [row[field] for row in raw_lines][: args.warmup + args.measure]

    if len(sentences) < args.warmup + args.measure:
        # Cycle if corpus is smaller than needed
        factor = (args.warmup + args.measure) // len(sentences) + 1
        sentences = (sentences * factor)[: args.warmup + args.measure]

    analyzer = zeyrek.MorphAnalyzer()

    # Warmup
    for s in sentences[: args.warmup]:
        analyzer.analyze(s)

    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss

    latencies: list[float] = []
    total_tokens = 0
    t0 = time.perf_counter()
    for s in sentences[args.warmup : args.warmup + args.measure]:
        ts = time.perf_counter()
        result = analyzer.analyze(s)
        latencies.append(time.perf_counter() - ts)
        total_tokens += len(result)
    wall = time.perf_counter() - t0

    rss_after = proc.memory_info().rss

    out = {
        "tok_per_sec": round(total_tokens / wall, 1),
        "p50_ms": round(statistics.median(latencies) * 1000, 3),
        "p95_ms": round(statistics.quantiles(latencies, n=20)[-1] * 1000, 3),
        "p99_ms": round(statistics.quantiles(latencies, n=100)[-1] * 1000, 3),
        "peak_rss_mb": round(rss_after / 1024 / 1024, 1),
        "delta_rss_mb": round((rss_after - rss_before) / 1024 / 1024, 1),
        "n_sentences": args.measure,
        "n_tokens": total_tokens,
        "host_cpu": _read_cpu_model(),
        "host_kernel": __import__("platform").release(),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
