#!/bin/bash
# Submit Zeyrek throughput benchmark to Orfoz CPU partition.
# Must be submitted from /arf/scratch/scolakoglu/ (TRUBA requirement).
#
# Usage (from /arf/scratch/scolakoglu/):
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_zeyrek_benchmark.sh

#SBATCH --job-name=aksu-zeyrek-bench
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=00:30:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/zeyrek_bench_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/zeyrek_bench_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs

cd "$PROJECT"
python scripts/benchmark_zeyrek.py \
    --corpus data/splits/test.jsonl \
    --warmup 100 \
    --measure 1000 \
    --output audit/benchmark_results/zeyrek_throughput.json

echo "Zeyrek benchmark complete. Results in audit/benchmark_results/zeyrek_throughput.json"
