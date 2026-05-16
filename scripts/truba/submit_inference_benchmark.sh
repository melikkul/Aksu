#!/bin/bash
# Benchmark BERTurk (C2) and reranker/disambiguator (C3) inference throughput
# on Orfoz CPU.  DualHead (C4) is omitted from the initial run because no
# checkpoint exists yet; re-run after DualHead training:
#
#   sbatch scripts/truba/submit_inference_benchmark.sh --include-dualhead
#
# Usage:
#   sbatch scripts/truba/submit_inference_benchmark.sh [--include-dualhead]

#SBATCH --job-name=aksu-inference-bench-cpu
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/inference_bench_cpu_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/inference_bench_cpu_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
cd "$PROJECT"

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=56

echo "=== Inference Benchmark (CPU/Orfoz) ==="
echo "Date:     $(date)"
echo "Node:     $(hostname)"
lscpu | grep "Model name" | head -1

# Parse optional --include-dualhead argument
INCLUDE_DUALHEAD=0
for arg in "$@"; do
    if [ "$arg" = "--include-dualhead" ]; then
        INCLUDE_DUALHEAD=1
    fi
done

if [ "$INCLUDE_DUALHEAD" = "1" ]; then
    echo "Mode: berturk + reranker + dualhead"
    COMPONENTS="berturk reranker dualhead"
    DUALHEAD_FLAG="--include-dualhead"
else
    echo "Mode: berturk + reranker (DualHead skipped — no checkpoint yet)"
    echo "      Re-run with --include-dualhead after DualHead training."
    COMPONENTS="berturk reranker"
    DUALHEAD_FLAG=""
fi

/usr/bin/time -v python scripts/benchmark_inference.py \
    --components $COMPONENTS \
    --corpus data/splits/val.jsonl \
    --warmup 50 \
    --measure 500 \
    --output audit/benchmark_results/inference_throughput.json \
    $DUALHEAD_FLAG

echo ""
echo "=== Benchmark complete: $(date) ==="
echo "Results written to audit/benchmark_results/inference_throughput.json"
